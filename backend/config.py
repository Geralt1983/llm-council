"""Configuration for the LLM Council."""

import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for storage
DATA_DIR = "data"


def get_council_models() -> List[str]:
    """Get current council models from database or defaults."""
    try:
        from .storage import get_council_config
        config = get_council_config()
        return config.get("council_models", _DEFAULT_COUNCIL_MODELS)
    except Exception:
        return _DEFAULT_COUNCIL_MODELS


def get_chairman_model() -> str:
    """Get current chairman model from database or default."""
    try:
        from .storage import get_council_config
        config = get_council_config()
        return config.get("chairman_model", _DEFAULT_CHAIRMAN_MODEL)
    except Exception:
        return _DEFAULT_CHAIRMAN_MODEL


# Default council configuration (used as fallback)
_DEFAULT_COUNCIL_MODELS = [
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-2.0-flash",
    "x-ai/grok-2",
]

_DEFAULT_CHAIRMAN_MODEL = "google/gemini-2.0-flash"

# For backwards compatibility - these now call the dynamic functions
# but modules can import them directly
COUNCIL_MODELS = property(lambda self: get_council_models())
CHAIRMAN_MODEL = property(lambda self: get_chairman_model())
