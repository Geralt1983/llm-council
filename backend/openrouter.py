"""OpenRouter API client for making LLM requests."""

import httpx
import json
import time
import re
from typing import List, Dict, Any, Optional, AsyncGenerator
from .config import OPENROUTER_API_KEY, OPENROUTER_API_URL

# OpenRouter models endpoint
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Cache for models list
_models_cache: Optional[Dict[str, Any]] = None
_models_cache_time: float = 0
MODELS_CACHE_TTL = 300  # 5 minutes


# Patterns to identify reasoning models
REASONING_MODEL_PATTERNS = [
    r'openai/o1',
    r'openai/o3',
    r'o1-preview',
    r'o1-mini',
    r'o3-mini',
    r'reasoning',
    r'thinking',
    r'deepseek.*reasoner',
]


def is_reasoning_model(model: str) -> bool:
    """
    Check if a model is a reasoning model that requires special handling.

    Reasoning models (o1, o3, etc.) have different API requirements:
    - No system messages (must be converted to user messages)
    - Different temperature handling
    - May return reasoning_content in addition to content
    """
    model_lower = model.lower()
    return any(re.search(pattern, model_lower) for pattern in REASONING_MODEL_PATTERNS)


async def fetch_available_models(
    force_refresh: bool = False
) -> List[Dict[str, Any]]:
    """
    Fetch available models from OpenRouter API.

    Returns a list of model objects with id, name, description, pricing, etc.
    Results are cached for 5 minutes unless force_refresh is True.
    """
    global _models_cache, _models_cache_time

    # Check cache
    if not force_refresh and _models_cache is not None:
        if time.time() - _models_cache_time < MODELS_CACHE_TTL:
            return _models_cache.get("data", [])

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                OPENROUTER_MODELS_URL,
                headers=headers
            )
            response.raise_for_status()

            data = response.json()
            _models_cache = data
            _models_cache_time = time.time()

            return data.get("data", [])

    except Exception as e:
        print(f"Error fetching models from OpenRouter: {e}")
        # Return cached data if available, otherwise empty list
        if _models_cache is not None:
            return _models_cache.get("data", [])
        return []


def format_model_for_display(model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a model from OpenRouter API for frontend display.

    Extracts key information and adds computed fields like is_reasoning.
    """
    model_id = model.get("id", "")
    pricing = model.get("pricing", {})

    # Calculate cost per 1M tokens
    prompt_cost = float(pricing.get("prompt", 0)) * 1_000_000
    completion_cost = float(pricing.get("completion", 0)) * 1_000_000

    return {
        "id": model_id,
        "name": model.get("name", model_id),
        "description": model.get("description", ""),
        "context_length": model.get("context_length", 0),
        "is_reasoning": is_reasoning_model(model_id),
        "pricing": {
            "prompt_per_million": prompt_cost,
            "completion_per_million": completion_cost,
        },
        "top_provider": model.get("top_provider", {}),
    }


def prepare_messages_for_model(
    model: str,
    messages: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Prepare messages for a specific model, handling reasoning model requirements.

    For reasoning models:
    - Convert system messages to user messages (prepended with context)
    - Merge consecutive user messages
    """
    if not is_reasoning_model(model):
        return messages

    # Convert system messages to user messages for reasoning models
    converted = []
    for msg in messages:
        if msg['role'] == 'system':
            # Convert system message to user message with clear labeling
            converted.append({
                'role': 'user',
                'content': f"[System Context]\n{msg['content']}"
            })
        else:
            converted.append(msg.copy())

    # Merge consecutive user messages (reasoning models may require alternating roles)
    merged = []
    for msg in converted:
        if merged and merged[-1]['role'] == msg['role'] == 'user':
            merged[-1]['content'] += '\n\n' + msg['content']
        else:
            merged.append(msg)

    return merged


def get_reasoning_model_params(model: str) -> Dict[str, Any]:
    """
    Get reasoning model-specific parameters for the API request.

    Returns parameters dict to merge with the payload for reasoning models.
    """
    params = {}

    if is_reasoning_model(model):
        # Reasoning models don't support temperature
        # Some may support reasoning_effort
        params['reasoning_effort'] = 'high'

    return params


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via OpenRouter API.

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds
        temperature: Optional temperature for response generation (0.0-2.0)
        max_tokens: Optional maximum tokens in response

    Returns:
        Response dict with 'content', optional 'reasoning_details', and metrics, or None if failed
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    # Prepare messages for this specific model (handles reasoning models)
    prepared_messages = prepare_messages_for_model(model, messages)

    # Build payload with model-specific parameters
    payload = {
        "model": model,
        "messages": prepared_messages,
    }

    # Add reasoning model-specific parameters (e.g., reasoning_effort for o1/o3)
    reasoning_params = get_reasoning_model_params(model)

    # For reasoning models, don't add temperature (they don't support it)
    if not is_reasoning_model(model):
        if temperature is not None:
            payload["temperature"] = temperature

    # Add max_tokens if specified
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    payload.update(reasoning_params)

    is_reasoning = is_reasoning_model(model)
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

            # Build response with reasoning model support
            result = {
                'content': message.get('content'),
                'reasoning_details': message.get('reasoning_details'),
                'is_reasoning_model': is_reasoning,
                'metrics': {
                    'response_time_ms': elapsed_ms,
                    'input_tokens': usage.get('prompt_tokens'),
                    'output_tokens': usage.get('completion_tokens'),
                    'total_tokens': usage.get('total_tokens'),
                }
            }

            # For reasoning models, also capture reasoning_content if available
            if is_reasoning:
                result['reasoning_content'] = message.get('reasoning_content')
                result['reasoning_tokens'] = usage.get('reasoning_tokens')

            return result

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        print(f"Error querying model {model}: {e}")
        return {
            'content': None,
            'error': str(e),
            'is_reasoning_model': is_reasoning,
            'metrics': {
                'response_time_ms': elapsed_ms,
                'error_type': type(e).__name__,
            }
        }


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]],
    model_parameters: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of OpenRouter model identifiers
        messages: List of message dicts to send to each model
        model_parameters: Optional dict mapping model_id to {temperature, max_tokens}

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio

    # Create tasks for all models with their specific parameters
    tasks = []
    for model in models:
        params = {}
        if model_parameters and model in model_parameters:
            mp = model_parameters[model]
            if "temperature" in mp:
                params["temperature"] = mp["temperature"]
            if "max_tokens" in mp:
                params["max_tokens"] = mp["max_tokens"]
        tasks.append(query_model(model, messages, **params))

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}


async def stream_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream tokens from a single model via OpenRouter API.

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds
        temperature: Optional temperature for response generation (0.0-2.0)
        max_tokens: Optional maximum tokens in response

    Yields:
        Dict with 'type' (token, reasoning, done, error) and 'content'
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    # Prepare messages for this specific model (handles reasoning models)
    prepared_messages = prepare_messages_for_model(model, messages)

    payload = {
        "model": model,
        "messages": prepared_messages,
        "stream": True,
    }

    # Add reasoning model-specific parameters
    reasoning_params = get_reasoning_model_params(model)

    # For reasoning models, don't add temperature (they don't support it)
    if not is_reasoning_model(model):
        if temperature is not None:
            payload["temperature"] = temperature

    # Add max_tokens if specified
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    payload.update(reasoning_params)

    is_reasoning = is_reasoning_model(model)

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
                full_reasoning = ""
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        result = {
                            "type": "done",
                            "content": full_content,
                            "is_reasoning_model": is_reasoning
                        }
                        if is_reasoning and full_reasoning:
                            result["reasoning_content"] = full_reasoning
                        yield result
                        return

                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})

                        # Handle regular content
                        content = delta.get("content", "")
                        if content:
                            full_content += content
                            yield {"type": "token", "content": content}

                        # Handle reasoning content for reasoning models
                        if is_reasoning:
                            reasoning = delta.get("reasoning_content", "")
                            if reasoning:
                                full_reasoning += reasoning
                                yield {"type": "reasoning", "content": reasoning}
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
    timeout: float = 120.0,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
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

    result = await query_model(model, messages, timeout, temperature, max_tokens)

    if result is None or result.get('content') is None:
        CircuitBreaker.record_failure(model)
    else:
        CircuitBreaker.record_success(model)

    return result


async def query_models_parallel_with_circuit_breaker(
    models: List[str],
    messages: List[Dict[str, str]],
    model_parameters: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel with circuit breaker protection.
    """
    import asyncio

    tasks = []
    for model in models:
        params = {}
        if model_parameters and model in model_parameters:
            mp = model_parameters[model]
            if "temperature" in mp:
                params["temperature"] = mp["temperature"]
            if "max_tokens" in mp:
                params["max_tokens"] = mp["max_tokens"]
        tasks.append(query_model_with_circuit_breaker(model, messages, **params))

    responses = await asyncio.gather(*tasks)

    return {model: response for model, response in zip(models, responses)}
