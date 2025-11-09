"""Gemini RAG Service for file search and retrieval.

This module provides a service wrapper for Google's Gemini File Search API,
handling file uploads, indexing, and querying with proper error handling
and retry logic.
"""

import logging
import os
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Max file size: 100 MB
MAX_FILE_SIZE = 100 * 1024 * 1024


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retrying API calls with exponential backoff."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            "Attempt %d failed for %s: %s. Retrying in %.1fs...",
                            attempt + 1,
                            func.__name__,
                            str(e),
                            current_delay,
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            "All %d attempts failed for %s", max_retries, func.__name__
                        )

            raise last_exception

        return wrapper

    return decorator


class GeminiRAGService:
    """Service for interacting with Gemini File Search RAG API."""

    def __init__(self, api_key: str):
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")

        try:
            self.client = genai.Client(api_key=api_key)
            self.current_store: Optional[str] = None
        except Exception as e:  # pylint: disable=broad-exception-caught
            raise ValueError(f"Failed to initialize Gemini client: {str(e)}") from e

    @staticmethod
    def _extract_store_name(store: Any) -> Tuple[str, str]:
        """Extract store name and display name from API response.

        Returns:
            Tuple of (store_name, display_name)
        """
        if isinstance(store, str):
            return store, "N/A"
        if hasattr(store, "name"):
            return store.name, getattr(store, "display_name", "N/A")
        if isinstance(store, dict):
            return store.get("name", "Unknown"), store.get("display_name", "N/A")
        raise ValueError(f"Invalid store response from API: {type(store)}")

    @staticmethod
    def _parse_error_message(error: Exception) -> Optional[str]:
        """Parse error message and return user-friendly error type.

        Returns:
            User-friendly error message or None if not recognized
        """
        error_msg = str(error).lower()
        if "quota" in error_msg or "limit" in error_msg:
            return "quota"
        if "permission" in error_msg or "unauthorized" in error_msg:
            return "permission"
        if "format" in error_msg or "unsupported" in error_msg:
            return "format"
        if "rate limit" in error_msg:
            return "rate_limit"
        if "safety" in error_msg or "blocked" in error_msg:
            return "safety"
        if "not found" in error_msg:
            return "not_found"
        return None

    def _handle_api_error(self, error: Exception, context: str = "") -> None:
        """Handle API errors with consistent error messages."""
        error_type = self._parse_error_message(error)
        if error_type == "quota":
            raise ValueError(f"Quota exceeded: {str(error)}") from error
        if error_type == "permission":
            raise ValueError(f"Permission denied: {str(error)}") from error
        if error_type == "format":
            raise ValueError(f"Unsupported file format: {str(error)}") from error
        if error_type == "rate_limit":
            raise ValueError(f"Rate limit exceeded: {str(error)}") from error
        if error_type == "safety":
            raise ValueError(
                f"Content blocked by safety filters: {str(error)}"
            ) from error
        if error_type == "not_found":
            raise ValueError(
                f"Store not found{': ' + context if context else ''}"
            ) from error
        raise error

    @retry_on_failure(max_retries=3)
    def create_file_search_store(self, display_name: str) -> str:
        """Create a new file search store and set it as current."""
        if not display_name or not display_name.strip():
            raise ValueError("Display name cannot be empty")

        if len(display_name) > 100:
            raise ValueError("Display name must be 100 characters or less")

        try:
            store = self.client.file_search_stores.create(
                config={"display_name": display_name.strip()}
            )
            store_name, _ = self._extract_store_name(store)
            self.current_store = store_name
            return store_name
        except ValueError:
            raise
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._handle_api_error(e)

    @retry_on_failure(max_retries=3)
    def list_file_search_stores(self) -> List[Dict]:
        """List all available file search stores."""
        try:
            stores = []
            for store in self.client.file_search_stores.list():
                try:
                    store_name, display_name = self._extract_store_name(store)
                    stores.append({"name": store_name, "display_name": display_name})
                except ValueError:
                    logger.warning("Unexpected store type: %s", type(store))
                    continue
            return stores
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error listing stores: %s", str(e))
            return []

    def _validate_file(self, file_path: str):
        """Validate file before upload."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValueError("File is empty")

        if file_size > MAX_FILE_SIZE:
            raise ValueError(
                f"File size ({file_size / (1024*1024):.2f} MB) exceeds maximum "
                f"allowed size ({MAX_FILE_SIZE / (1024*1024)} MB)"
            )

    @retry_on_failure(max_retries=3)
    def upload_and_index_file(
        self,
        file_path: str,
        display_name: Optional[str] = None,
        chunking_config: Optional[Dict] = None,
    ) -> Dict:
        """Upload a file directly to the current file search store."""
        if not self.current_store:
            raise ValueError("No file search store selected. Create one first")

        self._validate_file(file_path)

        config = {}
        if display_name:
            config["display_name"] = display_name.strip() if display_name else None
        if chunking_config:
            # Validate chunking config
            if "white_space_config" in chunking_config:
                ws_config = chunking_config["white_space_config"]
                if "max_tokens_per_chunk" in ws_config:
                    if not 100 <= ws_config["max_tokens_per_chunk"] <= 2000:
                        raise ValueError(
                            "max_tokens_per_chunk must be between 100 and 2000"
                        )
                if "max_overlap_tokens" in ws_config:
                    if ws_config["max_overlap_tokens"] < 0:
                        raise ValueError("max_overlap_tokens must be non-negative")
            config["chunking_config"] = chunking_config

        try:
            operation = self.client.file_search_stores.upload_to_file_search_store(
                file=file_path,
                file_search_store_name=self.current_store,
                config=config if config else None,
            )

            # Store the operation object itself (not just the name)
            # The API expects the operation object to be passed to operations.get()
            if isinstance(operation, str):
                # If it's a string, we'll need to handle it differently
                operation_name = operation
                operation_obj = None
                done = False
            elif hasattr(operation, "name"):
                operation_name = operation.name
                operation_obj = operation  # Keep the object
                done = getattr(operation, "done", False)
            else:
                raise ValueError(
                    f"Invalid operation response from API: {type(operation)}"
                )

            if not operation_name:
                raise ValueError("Operation name is empty")

            return {
                "operation_name": operation_name,
                "operation_obj": operation_obj,
                "done": done,
            }
        except ValueError:
            raise
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._handle_api_error(e)

    def wait_for_operation(
        self,
        operation_name: str,
        operation_obj: Optional[Any] = None,
        *,
        check_interval: int = 5,
        max_wait_time: int = 600,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Wait for an operation to complete with timeout and progress tracking.

        Args:
            operation_name: The operation name (for logging/fallback)
            operation_obj: The operation object returned from the API (preferred)
            check_interval: Seconds between status checks
            max_wait_time: Maximum time to wait in seconds
            progress_callback: Optional callback for progress updates
        """
        if not operation_name:
            raise ValueError("Operation name cannot be empty")

        # According to the API docs, operations.get() expects the operation object, not the name
        if operation_obj is None:
            # Fallback: if we only have the name, try to use it
            # But this might fail, so we'll use a simple wait
            logger.warning(
                "No operation object provided for %s, using fallback wait method",
                operation_name,
            )
            if progress_callback:
                progress_callback("Indexing in progress (estimated 30-60 seconds)...")
            time.sleep(30)
            if progress_callback:
                progress_callback(
                    "Indexing should be complete. You can try querying now."
                )
            return True

        start_time = time.time()
        check_count = 0

        # Use the operation object directly, as per API documentation
        # Pattern from docs: while not operation.done: operation = client.operations.get(operation)
        operation = operation_obj

        try:

            def is_done(op):
                """Check if operation is complete."""
                if op is None:
                    return False
                if isinstance(op, dict):
                    return op.get("done", False)
                return getattr(op, "done", False)

            # Poll the operation status
            # Per API docs: pass the operation object itself to operations.get()
            logger.info("Starting to wait for operation: %s", operation_name)
            while not is_done(operation):
                elapsed = time.time() - start_time
                if elapsed > max_wait_time:
                    raise TimeoutError(
                        f"Operation {operation_name} timed out after {max_wait_time} seconds"
                    )

                if progress_callback:
                    progress_callback(
                        f"Waiting for operation... ({int(elapsed)}s elapsed)"
                    )

                time.sleep(check_interval)
                check_count += 1

                # Get updated operation status
                # Pass the operation object itself, not the name (per API docs)
                try:
                    operation = self.client.operations.get(operation)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    if check_count < 3:
                        logger.warning(
                            "Error getting status for operation %s: %s. Retrying...",
                            operation_name,
                            str(e),
                        )
                        time.sleep(check_interval)
                        continue
                    # If we can't check status after retries, assume it's processing
                    logger.warning(
                        "Cannot verify completion for operation %s, assuming it will complete",
                        operation_name,
                    )
                    return True

            # Check for operation errors
            error = None
            if isinstance(operation, dict):
                error = operation.get("error")
            elif hasattr(operation, "error"):
                error = operation.error

            if error:
                error_msg = str(error)
                raise RuntimeError(f"Operation {operation_name} failed: {error_msg}")

            logger.info("Operation %s completed successfully", operation_name)
            return is_done(operation)
        except TimeoutError:
            raise
        except Exception as e:
            logger.error("Error waiting for operation %s: %s", operation_name, str(e))
            raise

    @retry_on_failure(max_retries=2, delay=2.0)
    def query(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash",
        metadata_filter: Optional[str] = None,
    ) -> Dict:
        """Query the RAG system with a prompt."""
        if not self.current_store:
            raise ValueError("No file search store selected.")

        if not prompt or not prompt.strip():
            raise ValueError("Query prompt cannot be empty")

        # Validate model
        valid_models = ["gemini-2.5-flash", "gemini-2.5-pro"]
        if model not in valid_models:
            raise ValueError(
                f"Invalid model. Must be one of: {', '.join(valid_models)}"
            )

        try:
            file_search_config = types.FileSearch(
                file_search_store_names=[self.current_store]
            )

            if metadata_filter and metadata_filter.strip():
                file_search_config.metadata_filter = metadata_filter.strip()

            response = self.client.models.generate_content(
                model=model,
                contents=prompt.strip(),
                config=types.GenerateContentConfig(
                    tools=[types.Tool(file_search=file_search_config)]
                ),
            )

            if not response:
                raise ValueError("Empty response from API")

            result = {
                "text": (
                    response.text
                    if hasattr(response, "text") and response.text
                    else "No response generated"
                ),
                "citations": [],
            }

            # Extract citations safely
            try:
                candidate = response.candidates[0]
                grounding = candidate.grounding_metadata
                chunks = grounding.retrieval_metadata.chunks
                for chunk in chunks:
                    chunk_data = chunk.chunk
                    result["citations"].append(
                        {
                            "file_name": getattr(chunk_data, "file_name", "Unknown"),
                            "chunk_index": getattr(chunk_data, "chunk_index", None),
                        }
                    )
            except (AttributeError, IndexError):
                pass  # No citations available

            return result
        except ValueError:
            raise
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._handle_api_error(e)

    def set_current_store(self, store_name: str):
        """Set the active file search store."""
        if not store_name or not store_name.strip():
            raise ValueError("Store name cannot be empty")
        self.current_store = store_name.strip()

    @staticmethod
    def _validate_non_empty(value: str, field_name: str) -> str:
        """Validate that a string value is not empty."""
        if not value or not value.strip():
            raise ValueError(f"{field_name} cannot be empty")
        return value.strip()

    @retry_on_failure(max_retries=2)
    def delete_store(self, store_name: str, force: bool = True):
        """Delete a file search store."""
        store_name = self._validate_non_empty(store_name, "Store name")

        try:
            self.client.file_search_stores.delete(
                name=store_name, config={"force": force}
            )
            if self.current_store == store_name:
                self.current_store = None
        except ValueError:
            raise
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._handle_api_error(e, context=store_name)
