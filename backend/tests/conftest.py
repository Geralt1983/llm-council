"""Pytest fixtures and configuration."""

import pytest
import asyncio
import os
import tempfile
from unittest.mock import patch, AsyncMock, MagicMock

# Set test environment
os.environ["OPENROUTER_API_KEY"] = "test-api-key"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_openrouter_response():
    """Mock successful OpenRouter API response."""
    return {
        "choices": [{
            "message": {
                "content": "This is a test response from the model.",
                "role": "assistant"
            }
        }]
    }


@pytest.fixture
def mock_httpx_client(mock_openrouter_response):
    """Mock httpx async client."""
    mock_response = MagicMock()
    mock_response.json.return_value = mock_openrouter_response
    mock_response.raise_for_status.return_value = None

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    return mock_client


@pytest.fixture
def sample_stage1_results():
    """Sample Stage 1 results for testing."""
    return [
        {"model": "openai/gpt-4o", "response": "GPT-4 response about the topic."},
        {"model": "anthropic/claude-3.5-sonnet", "response": "Claude response with details."},
        {"model": "google/gemini-2.5-flash", "response": "Gemini response explanation."},
    ]


@pytest.fixture
def sample_stage2_results():
    """Sample Stage 2 ranking results."""
    return [
        {
            "model": "openai/gpt-4o",
            "ranking": """Response A is good but lacks depth.
Response B provides comprehensive coverage.
Response C is accurate but brief.

FINAL RANKING:
1. Response B
2. Response A
3. Response C""",
            "parsed_ranking": ["Response B", "Response A", "Response C"]
        },
        {
            "model": "anthropic/claude-3.5-sonnet",
            "ranking": """All responses are informative.
Response B stands out.
Response A is also strong.

FINAL RANKING:
1. Response B
2. Response C
3. Response A""",
            "parsed_ranking": ["Response B", "Response C", "Response A"]
        },
    ]


@pytest.fixture
def sample_label_to_model():
    """Sample label to model mapping."""
    return {
        "Response A": "openai/gpt-4o",
        "Response B": "anthropic/claude-3.5-sonnet",
        "Response C": "google/gemini-2.5-flash"
    }


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    os.unlink(db_path)


@pytest.fixture
def mock_council_models():
    """Mock council model configuration."""
    return [
        "openai/gpt-4o",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-2.5-flash"
    ]
