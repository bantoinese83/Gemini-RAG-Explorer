"""Configuration management for Gemini RAG Explorer.

This module handles loading environment variables and API key configuration.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def get_api_key() -> str:
    """Get Gemini API key from environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Please set it in your .env file or environment."
        )
    return api_key
