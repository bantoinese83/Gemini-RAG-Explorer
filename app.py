"""Streamlit application for Gemini RAG Explorer.

This application provides a web interface for uploading documents,
creating file search stores, and querying documents using Gemini's
File Search RAG capabilities.
"""

import os
import tempfile
import time
import traceback
from pathlib import Path

import streamlit as st

from src.config import get_api_key
from src.rag_service import GeminiRAGService

st.set_page_config(page_title="Gemini RAG Explorer", page_icon="🔍", layout="wide")

# Initialize session state
if "current_store_name" not in st.session_state:
    st.session_state.current_store_name = None
if "upload_progress" not in st.session_state:
    st.session_state.upload_progress = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []


@st.cache_resource
def get_rag_service():
    """Initialize and cache the RAG service."""
    try:
        api_key = get_api_key()
        service = GeminiRAGService(api_key)
        return service
    except ValueError as e:
        st.error(f"❌ Configuration Error: {str(e)}")
        st.info("💡 Make sure you've created a `.env` file with your `GEMINI_API_KEY`")
        st.stop()
    except (RuntimeError, ConnectionError, OSError) as e:
        st.error(f"❌ Failed to initialize service: {str(e)}")
        st.stop()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: File size in bytes

    Returns:
        Formatted string with appropriate unit (B, KB, MB, GB, TB)
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def main():
    """Main application entry point.

    Sets up the Streamlit UI and handles user interactions for
    file upload, store management, and document querying.
    """
    st.title("🔍 Gemini RAG Explorer")
    st.markdown(
        "Upload documents and query them using Gemini's File Search RAG capabilities"
    )

    rag_service = get_rag_service()

    # Sync session state with service
    if st.session_state.current_store_name:
        try:
            rag_service.set_current_store(st.session_state.current_store_name)
        except (ValueError, AttributeError):
            st.session_state.current_store_name = None

    # Sidebar for store management
    with st.sidebar:
        st.header("📚 File Search Stores")

        # List existing stores with error handling
        try:
            stores = rag_service.list_file_search_stores()
        except (RuntimeError, ConnectionError, AttributeError) as e:
            st.error(f"Failed to load stores: {str(e)}")
            stores = []

        if stores:
            st.subheader("Existing Stores")
            store_names = [s["name"] for s in stores]
            store_display = [
                f"{s['display_name']} ({s['name'].split('/')[-1]})" for s in stores
            ]

            # Determine selected index
            selected_idx = 0
            current_store = (
                st.session_state.current_store_name or rag_service.current_store
            )
            if current_store and current_store in store_names:
                selected_idx = store_names.index(current_store)

            selected_display = st.selectbox(
                "Select Store",
                options=range(len(store_display)),
                format_func=lambda x: store_display[x],
                index=selected_idx,
                key="store_selector",
            )

            if st.button("Set as Active", use_container_width=True):
                try:
                    selected_store = store_names[selected_display]
                    rag_service.set_current_store(selected_store)
                    st.session_state.current_store_name = selected_store
                    st.success(f"✅ Active: {store_display[selected_display]}")
                    st.rerun()
                except (ValueError, AttributeError) as e:
                    st.error(f"Failed to set store: {str(e)}")
        else:
            st.info("No stores found. Create one below.")

        st.divider()

        # Create new store
        st.subheader("Create New Store")
        new_store_name = st.text_input(
            "Store Display Name",
            placeholder="my-documents",
            max_chars=100,
            help="Maximum 100 characters",
        )
        if st.button("Create Store", use_container_width=True):
            if new_store_name and new_store_name.strip():
                try:
                    store_name = rag_service.create_file_search_store(
                        new_store_name.strip()
                    )
                    st.session_state.current_store_name = store_name
                    st.success(f"✅ Created: {store_name}")
                    time.sleep(0.5)  # Brief delay for user feedback
                    st.rerun()
                except ValueError as e:
                    st.error(f"❌ {str(e)}")
                except (RuntimeError, ConnectionError) as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
            else:
                st.warning("⚠️ Please enter a store name")

    # Main content area
    tab1, tab2, tab3 = st.tabs(["📤 Upload Files", "💬 Query", "⚙️ Settings"])

    with tab1:
        st.header("Upload & Index Documents")

        current_store = st.session_state.current_store_name or rag_service.current_store
        if not current_store:
            st.warning(
                "⚠️ No active file search store. Create one in the sidebar first."
            )
        else:
            st.info(f"📦 Active store: `{current_store}`")

            uploaded_file = st.file_uploader(
                "Choose a file to upload",
                type=[
                    "txt",
                    "pdf",
                    "md",
                    "docx",
                    "csv",
                    "json",
                    "html",
                    "xlsx",
                    "pptx",
                ],
                help="Supported: PDF, DOCX, TXT, MD, CSV, JSON, HTML, and more. Max size: 100 MB",
            )

            if uploaded_file:
                # Show file info
                file_size = len(uploaded_file.getvalue())
                file_size_mb = file_size / (1024 * 1024)

                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.metric("File Size", format_file_size(file_size))
                with col_info2:
                    st.metric("File Type", uploaded_file.type or "Unknown")

                if file_size > 100 * 1024 * 1024:
                    st.error(
                        f"⚠️ File size ({file_size_mb:.2f} MB) exceeds 100 MB limit"
                    )
                elif file_size == 0:
                    st.error("⚠️ File is empty")
                else:
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        display_name = st.text_input(
                            "Display Name (for citations)",
                            value=uploaded_file.name,
                            max_chars=200,
                            help="Name shown in citations",
                        )

                    with col2:
                        st.write("**Chunking Config**")
                        max_tokens = st.number_input(
                            "Max tokens per chunk",
                            min_value=100,
                            max_value=2000,
                            value=512,
                            step=100,
                            help="Larger chunks = more context, but slower retrieval",
                        )
                        overlap_tokens = st.number_input(
                            "Overlap tokens",
                            min_value=0,
                            max_value=100,
                            value=20,
                            step=5,
                            help="Overlap between chunks for better context",
                        )

                    if st.button(
                        "🚀 Upload & Index", type="primary", use_container_width=True
                    ):
                        tmp_path = None
                        try:
                            # Validate before proceeding
                            if file_size == 0:
                                raise ValueError("File is empty")
                            if file_size > 100 * 1024 * 1024:
                                raise ValueError("File exceeds 100 MB limit")

                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=Path(uploaded_file.name).suffix
                            ) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_path = tmp_file.name

                            chunking_config = {
                                "white_space_config": {
                                    "max_tokens_per_chunk": max_tokens,
                                    "max_overlap_tokens": overlap_tokens,
                                }
                            }

                            # Progress tracking
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            status_text.info("📤 Uploading file...")
                            progress_bar.progress(20)

                            operation = rag_service.upload_and_index_file(
                                file_path=tmp_path,
                                display_name=(
                                    display_name.strip()
                                    if display_name
                                    else uploaded_file.name
                                ),
                                chunking_config=chunking_config,
                            )

                            progress_bar.progress(40)
                            status_text.info(
                                "⏳ Indexing in progress... This may take a few minutes."
                            )

                            # Wait for completion with progress updates
                            def update_progress(msg: str):
                                status_text.info(f"⏳ {msg}")

                            rag_service.wait_for_operation(
                                operation["operation_name"],
                                operation_obj=operation.get("operation_obj"),
                                progress_callback=update_progress,
                            )

                            progress_bar.progress(100)
                            status_text.success(
                                "✅ File uploaded and indexed successfully!"
                            )

                            time.sleep(1)
                            st.rerun()

                        except ValueError as e:
                            st.error(f"❌ Validation Error: {str(e)}")
                        except TimeoutError as e:
                            st.error(f"⏱️ {str(e)}")
                            st.info(
                                "The file may still be processing. Check back later."
                            )
                        except (RuntimeError, ConnectionError, OSError) as e:
                            st.error(f"❌ Error: {str(e)}")
                            with st.expander("🔍 Error Details"):
                                st.code(traceback.format_exc())
                        finally:
                            # Clean up temp file
                            if tmp_path and os.path.exists(tmp_path):
                                try:
                                    os.unlink(tmp_path)
                                except OSError:
                                    pass

    with tab2:
        st.header("Query Your Documents")

        current_store = st.session_state.current_store_name or rag_service.current_store
        if not current_store:
            st.warning(
                "⚠️ No active file search store. Create one and upload files first."
            )
        else:
            st.info(f"🔍 Querying store: `{current_store}`")

            col_model, col_advanced = st.columns([1, 1])

            with col_model:
                model_choice = st.selectbox(
                    "Model",
                    ["gemini-2.5-flash", "gemini-2.5-pro"],
                    index=0,
                    help="Flash is faster, Pro is more capable",
                )

            with col_advanced:
                show_advanced = st.checkbox("Advanced Options", value=False)

            query = st.text_area(
                "Enter your question",
                height=120,
                placeholder="What information can you find about...",
                help="Ask questions about your uploaded documents",
            )

            metadata_filter = None
            if show_advanced:
                metadata_filter = st.text_input(
                    "Metadata Filter (optional)",
                    placeholder='author="Robert Graves"',
                    help="Use AIP-160 filter syntax to search specific documents",
                )

            # Query history
            if st.session_state.query_history:
                with st.expander("📜 Recent Queries", expanded=False):
                    for i, (q, _) in enumerate(
                        reversed(st.session_state.query_history[-5:]), 1
                    ):
                        if st.button(
                            f"{i}. {q[:50]}...",
                            key=f"history_{i}",
                            use_container_width=True,
                        ):
                            query = q

            col_query, col_clear = st.columns([3, 1])

            with col_query:
                query_button = st.button(
                    "🔍 Query", type="primary", use_container_width=True
                )

            with col_clear:
                if st.button("Clear", use_container_width=True):
                    query = ""
                    st.rerun()

            if query_button:
                if query and query.strip():
                    try:
                        with st.spinner(
                            "🔍 Searching documents and generating response..."
                        ):
                            result = rag_service.query(
                                prompt=query.strip(),
                                model=model_choice,
                                metadata_filter=(
                                    metadata_filter.strip()
                                    if metadata_filter and metadata_filter.strip()
                                    else None
                                ),
                            )

                            # Save to history
                            st.session_state.query_history.append(
                                (query.strip(), result["text"])
                            )
                            if len(st.session_state.query_history) > 20:
                                st.session_state.query_history = (
                                    st.session_state.query_history[-20:]
                                )

                            st.subheader("💬 Response")
                            st.markdown(result["text"])

                            if result["citations"]:
                                st.divider()
                                st.subheader(
                                    f"📎 Citations ({len(result['citations'])})"
                                )

                                # Group citations by file
                                citations_by_file = {}
                                for citation in result["citations"]:
                                    file_name = citation.get("file_name", "Unknown")
                                    if file_name not in citations_by_file:
                                        citations_by_file[file_name] = []
                                    citations_by_file[file_name].append(citation)

                                for file_name, citations in citations_by_file.items():
                                    with st.expander(
                                        f"📄 {file_name} ({len(citations)} chunks)"
                                    ):
                                        for citation in citations:
                                            st.write(
                                                f"**Chunk {citation.get('chunk_index', 'N/A')}**"
                                            )
                            else:
                                st.info("ℹ️ No citations found in this response")

                    except ValueError as e:
                        st.error(f"❌ {str(e)}")
                    except (RuntimeError, ConnectionError) as e:
                        st.error(f"❌ Query error: {str(e)}")
                        with st.expander("🔍 Error Details"):
                            st.code(traceback.format_exc())
                else:
                    st.warning("⚠️ Please enter a query")

    with tab3:
        st.header("Settings & Info")

        current_store = st.session_state.current_store_name or rag_service.current_store

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Active Store", "Set" if current_store else "None")
        with col2:
            st.metric("Total Stores", len(stores))

        st.divider()

        st.subheader("Current Configuration")
        with st.container():
            st.code(f"Active Store: {current_store or 'None'}", language=None)
            st.code(f"Total Stores: {len(stores)}", language=None)
            st.code(
                f"API Key: {'✅ Configured' if get_api_key() else '❌ Missing'}",
                language=None,
            )

        st.divider()

        st.subheader("📋 Supported File Types")
        col_docs, col_code = st.columns(2)

        with col_docs:
            st.markdown(
                """
            **Documents**
            - PDF, DOCX, TXT, MD, RTF
            - XLSX, PPTX
            - HTML, XML
            """
            )

        with col_code:
            st.markdown(
                """
            **Code & Data**
            - Python, JavaScript, TypeScript
            - Java, C++, Go, Rust
            - CSV, JSON, YAML
            - 100+ file types supported
            """
            )

        st.divider()

        st.subheader("⚡ Rate Limits & Pricing")
        st.markdown(
            """
        - **Max file size**: 100 MB per document
        - **Free tier**: 1 GB total storage
        - **Recommended**: Keep stores under 20 GB for optimal performance
        - **Embeddings**: $0.15 per 1M tokens (indexing time)
        - **Storage**: Free
        - **Query embeddings**: Free
        """
        )

        st.divider()

        if stores:
            st.subheader("🗑️ Manage Stores")
            st.warning("⚠️ Deleting a store permanently removes all indexed documents!")

            store_to_delete = st.selectbox(
                "Select store to delete",
                options=[s["name"] for s in stores],
                key="delete_store_selector",
            )

            col_del1, col_del2 = st.columns([1, 1])
            with col_del1:
                confirm_delete = st.checkbox(
                    "I understand this action cannot be undone", key="confirm_delete"
                )
            with col_del2:
                if st.button(
                    "🗑️ Delete Store",
                    type="secondary",
                    use_container_width=True,
                    disabled=not confirm_delete,
                ):
                    try:
                        rag_service.delete_store(store_to_delete)
                        if st.session_state.current_store_name == store_to_delete:
                            st.session_state.current_store_name = None
                        st.success("✅ Store deleted successfully")
                        time.sleep(0.5)
                        st.rerun()
                    except ValueError as e:
                        st.error(f"❌ {str(e)}")
                    except (RuntimeError, ConnectionError) as e:
                        st.error(f"❌ Error deleting store: {str(e)}")
                        with st.expander("🔍 Error Details"):
                            st.code(traceback.format_exc())

        st.divider()

        st.subheader("🔄 Session Management")
        if st.button("Clear Session State", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key != "query_history":  # Keep query history
                    del st.session_state[key]
            st.session_state.current_store_name = None
            st.success("✅ Session state cleared")
            st.rerun()


if __name__ == "__main__":
    main()
