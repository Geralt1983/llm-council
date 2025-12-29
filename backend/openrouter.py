"""OpenRouter API client for making LLM requests."""

import httpx
import json
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
from .config import OPENROUTER_API_KEY, OPENROUTER_API_URL


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via OpenRouter API.

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content', optional 'reasoning_details', and metrics, or None if failed
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    start_time = time.time()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            elapsed_ms = int((time.time() - start_time) * 1000)
            data = response.json()
            message = data['choices'][0]['message']

            # Extract usage data if available
            usage = data.get('usage', {})

            return {
                'content': message.get('content'),
                'reasoning_details': message.get('reasoning_details'),
                'metrics': {
                    'response_time_ms': elapsed_ms,
                    'input_tokens': usage.get('prompt_tokens'),
                    'output_tokens': usage.get('completion_tokens'),
                    'total_tokens': usage.get('total_tokens'),
                }
            }

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        print(f"Error querying model {model}: {e}")
        return {
            'content': None,
            'error': str(e),
            'metrics': {
                'response_time_ms': elapsed_ms,
                'error_type': type(e).__name__,
            }
        }


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of OpenRouter model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio

    # Create tasks for all models
    tasks = [query_model(model, messages) for model in models]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}


async def stream_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream tokens from a single model via OpenRouter API.

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Yields:
        Dict with 'type' (token, done, error) and 'content'
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                OPENROUTER_API_URL,
                headers=headers,
                json=payload
            ) as response:
                response.raise_for_status()

                full_content = ""
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        yield {"type": "done", "content": full_content}
                        return

                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_content += content
                            yield {"type": "token", "content": content}
                    except json.JSONDecodeError:
                        continue

    except httpx.HTTPStatusError as e:
        yield {"type": "error", "content": f"HTTP {e.response.status_code}: {str(e)}"}
    except Exception as e:
        yield {"type": "error", "content": str(e)}


# Circuit breaker state
_circuit_breaker_state: Dict[str, Dict[str, Any]] = {}


class CircuitBreaker:
    """
    Circuit breaker pattern for model API calls.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing, reject requests immediately
    - HALF_OPEN: Testing if service recovered
    """

    FAILURE_THRESHOLD = 3  # failures before opening circuit
    RECOVERY_TIMEOUT = 60  # seconds before trying again
    SUCCESS_THRESHOLD = 2  # successes to close circuit

    @classmethod
    def get_state(cls, model: str) -> Dict[str, Any]:
        """Get circuit state for a model."""
        import time
        if model not in _circuit_breaker_state:
            _circuit_breaker_state[model] = {
                "state": "CLOSED",
                "failures": 0,
                "successes": 0,
                "last_failure_time": 0
            }
        return _circuit_breaker_state[model]

    @classmethod
    def can_execute(cls, model: str) -> bool:
        """Check if request can proceed."""
        import time
        state = cls.get_state(model)

        if state["state"] == "CLOSED":
            return True

        if state["state"] == "OPEN":
            # Check if recovery timeout has passed
            if time.time() - state["last_failure_time"] >= cls.RECOVERY_TIMEOUT:
                state["state"] = "HALF_OPEN"
                state["successes"] = 0
                return True
            return False

        # HALF_OPEN: allow request to test recovery
        return True

    @classmethod
    def record_success(cls, model: str):
        """Record successful request."""
        state = cls.get_state(model)

        if state["state"] == "HALF_OPEN":
            state["successes"] += 1
            if state["successes"] >= cls.SUCCESS_THRESHOLD:
                state["state"] = "CLOSED"
                state["failures"] = 0
                state["successes"] = 0
        else:
            state["failures"] = 0

    @classmethod
    def record_failure(cls, model: str):
        """Record failed request."""
        import time
        state = cls.get_state(model)

        state["failures"] += 1
        state["last_failure_time"] = time.time()

        if state["state"] == "HALF_OPEN":
            state["state"] = "OPEN"
        elif state["failures"] >= cls.FAILURE_THRESHOLD:
            state["state"] = "OPEN"

    @classmethod
    def get_status(cls, model: str) -> str:
        """Get current circuit state."""
        return cls.get_state(model)["state"]


async def query_model_with_circuit_breaker(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a model with circuit breaker protection.

    Returns None if circuit is open or request fails.
    """
    if not CircuitBreaker.can_execute(model):
        print(f"Circuit breaker OPEN for {model}, skipping request")
        return {
            'content': None,
            'error': 'Circuit breaker OPEN',
            'metrics': {
                'response_time_ms': 0,
                'error_type': 'CircuitBreakerOpen',
            }
        }

    result = await query_model(model, messages, timeout)

    if result is None or result.get('content') is None:
        CircuitBreaker.record_failure(model)
    else:
        CircuitBreaker.record_success(model)

    return result


async def query_models_parallel_with_circuit_breaker(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel with circuit breaker protection.
    """
    import asyncio

    tasks = [query_model_with_circuit_breaker(model, messages) for model in models]
    responses = await asyncio.gather(*tasks)

    return {model: response for model, response in zip(models, responses)}
