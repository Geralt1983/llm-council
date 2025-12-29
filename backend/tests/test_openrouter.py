"""Tests for OpenRouter API client."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import httpx


class TestQueryModel:
    """Tests for query_model function."""

    @pytest.mark.asyncio
    async def test_query_model_success(self, mock_openrouter_response):
        """Test successful model query."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_openrouter_response
        mock_response.raise_for_status.return_value = None

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            from backend.openrouter import query_model

            result = await query_model(
                "openai/gpt-4o",
                [{"role": "user", "content": "Hello"}]
            )

            assert result is not None
            assert result["content"] == "This is a test response from the model."

    @pytest.mark.asyncio
    async def test_query_model_failure(self):
        """Test model query failure returns None."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.HTTPStatusError(
                "Error", request=MagicMock(), response=MagicMock()
            )
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            from backend.openrouter import query_model

            result = await query_model(
                "openai/gpt-4o",
                [{"role": "user", "content": "Hello"}]
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_query_models_parallel(self, mock_openrouter_response):
        """Test parallel model queries."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_openrouter_response
        mock_response.raise_for_status.return_value = None

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            from backend.openrouter import query_models_parallel

            models = ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"]
            messages = [{"role": "user", "content": "Hello"}]

            results = await query_models_parallel(models, messages)

            assert len(results) == 2
            assert "openai/gpt-4o" in results
            assert "anthropic/claude-3.5-sonnet" in results


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_is_closed(self):
        """Test circuit breaker starts in CLOSED state."""
        from backend.openrouter import CircuitBreaker, _circuit_breaker_state

        # Clear state for test
        _circuit_breaker_state.clear()

        state = CircuitBreaker.get_state("test-model")
        assert state["state"] == "CLOSED"
        assert state["failures"] == 0

    def test_can_execute_when_closed(self):
        """Test requests allowed when circuit is closed."""
        from backend.openrouter import CircuitBreaker, _circuit_breaker_state

        _circuit_breaker_state.clear()

        assert CircuitBreaker.can_execute("test-model") is True

    def test_circuit_opens_after_failures(self):
        """Test circuit opens after threshold failures."""
        from backend.openrouter import CircuitBreaker, _circuit_breaker_state

        _circuit_breaker_state.clear()

        # Record failures up to threshold
        for _ in range(CircuitBreaker.FAILURE_THRESHOLD):
            CircuitBreaker.record_failure("test-model")

        assert CircuitBreaker.get_status("test-model") == "OPEN"

    def test_cannot_execute_when_open(self):
        """Test requests blocked when circuit is open."""
        from backend.openrouter import CircuitBreaker, _circuit_breaker_state

        _circuit_breaker_state.clear()

        # Open the circuit
        for _ in range(CircuitBreaker.FAILURE_THRESHOLD):
            CircuitBreaker.record_failure("test-model")

        assert CircuitBreaker.can_execute("test-model") is False

    def test_success_resets_failures(self):
        """Test successful request resets failure count."""
        from backend.openrouter import CircuitBreaker, _circuit_breaker_state

        _circuit_breaker_state.clear()

        # Record some failures (below threshold)
        CircuitBreaker.record_failure("test-model")
        CircuitBreaker.record_failure("test-model")

        # Record success
        CircuitBreaker.record_success("test-model")

        state = CircuitBreaker.get_state("test-model")
        assert state["failures"] == 0
        assert state["state"] == "CLOSED"


class TestStreamModel:
    """Tests for stream_model function."""

    @pytest.mark.asyncio
    async def test_stream_model_yields_tokens(self):
        """Test streaming yields token events."""
        mock_lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            'data: [DONE]'
        ]

        async def mock_aiter_lines():
            for line in mock_lines:
                yield line

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.aiter_lines = mock_aiter_lines

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_stream_context = MagicMock()
            mock_stream_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream_context.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream.return_value = mock_stream_context
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            from backend.openrouter import stream_model

            tokens = []
            async for chunk in stream_model(
                "openai/gpt-4o",
                [{"role": "user", "content": "Hello"}]
            ):
                tokens.append(chunk)

            # Should have 2 token events and 1 done event
            assert len(tokens) == 3
            assert tokens[0]["type"] == "token"
            assert tokens[0]["content"] == "Hello"
            assert tokens[1]["type"] == "token"
            assert tokens[1]["content"] == " world"
            assert tokens[2]["type"] == "done"
