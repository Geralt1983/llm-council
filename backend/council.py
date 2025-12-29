"""3-stage LLM Council orchestration."""

from typing import List, Dict, Any, Tuple, Optional, AsyncGenerator
from .openrouter import (
    query_models_parallel,
    query_model,
    query_models_parallel_with_circuit_breaker,
    query_model_with_circuit_breaker,
    stream_model,
    CircuitBreaker
)
from .config import get_council_models, get_chairman_model


# OpenRouter pricing per 1M tokens (approximate, varies by model)
# Format: model_prefix -> (input_price, output_price) per 1M tokens
MODEL_PRICING = {
    'openai/gpt-4o': (2.50, 10.00),
    'openai/gpt-4': (30.00, 60.00),
    'openai/gpt-3.5': (0.50, 1.50),
    'anthropic/claude-3.5': (3.00, 15.00),
    'anthropic/claude-3': (3.00, 15.00),
    'anthropic/claude-sonnet': (3.00, 15.00),
    'google/gemini-2.5': (0.15, 0.60),
    'google/gemini-2': (0.10, 0.40),
    'google/gemini-1.5': (0.075, 0.30),
    'x-ai/grok': (5.00, 15.00),
    'meta-llama/llama': (0.20, 0.20),
    'mistralai/': (0.25, 0.25),
    'default': (1.00, 3.00),  # Fallback pricing
}


def estimate_cost(model: str, metrics: Dict[str, Any]) -> float:
    """
    Estimate the cost of an API call based on token usage.

    Args:
        model: The model identifier
        metrics: Dict with input_tokens and output_tokens

    Returns:
        Estimated cost in USD
    """
    input_tokens = metrics.get('input_tokens', 0) or 0
    output_tokens = metrics.get('output_tokens', 0) or 0

    # Find matching pricing
    input_price, output_price = MODEL_PRICING['default']
    for prefix, (inp, outp) in MODEL_PRICING.items():
        if prefix != 'default' and model.startswith(prefix):
            input_price, output_price = inp, outp
            break

    # Calculate cost (prices are per 1M tokens)
    cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000

    return round(cost, 6)


def build_messages_with_history(
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """
    Build messages list including conversation history.

    Args:
        user_query: The current user query
        conversation_history: Previous messages in the conversation

    Returns:
        List of message dicts for the API
    """
    messages = []

    # Add conversation history if provided
    if conversation_history:
        for msg in conversation_history:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })

    # Add current query
    messages.append({"role": "user", "content": user_query})

    return messages


async def stage1_collect_responses(
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    use_circuit_breaker: bool = True
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Stage 1: Collect individual responses from all council models.

    Args:
        user_query: The user's question
        conversation_history: Previous messages for multi-turn context
        use_circuit_breaker: Whether to use circuit breaker protection

    Returns:
        Tuple of (results list, metrics list)
    """
    messages = build_messages_with_history(user_query, conversation_history)

    # Query all models in parallel
    council_models = get_council_models()

    if use_circuit_breaker:
        responses = await query_models_parallel_with_circuit_breaker(council_models, messages)
    else:
        responses = await query_models_parallel(council_models, messages)

    # Format results and collect metrics
    stage1_results = []
    stage1_metrics = []

    for model, response in responses.items():
        if response is not None:
            content = response.get('content')
            metrics = response.get('metrics', {})

            # Build metrics record
            metric_record = {
                'model_id': model,
                'stage': 'stage1',
                'response_time_ms': metrics.get('response_time_ms'),
                'input_tokens': metrics.get('input_tokens'),
                'output_tokens': metrics.get('output_tokens'),
                'total_tokens': metrics.get('total_tokens'),
                'error_type': metrics.get('error_type'),
            }

            # Calculate estimated cost
            if metrics.get('total_tokens'):
                metric_record['cost_usd'] = estimate_cost(model, metrics)

            stage1_metrics.append(metric_record)

            # Only include in results if we got content
            if content:
                stage1_results.append({
                    "model": model,
                    "response": content
                })

    return stage1_results, stage1_metrics


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    use_circuit_breaker: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]]]:
    """
    Stage 2: Each model ranks the anonymized responses.

    Args:
        user_query: The original user query
        stage1_results: Results from Stage 1
        use_circuit_breaker: Whether to use circuit breaker protection

    Returns:
        Tuple of (rankings list, label_to_model mapping, metrics list)
    """
    # Create anonymized labels for responses (Response A, Response B, etc.)
    labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...

    # Create mapping from label to model name
    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, stage1_results)
    }

    # Build the ranking prompt
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, stage1_results)
    ])

    ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""

    messages = [{"role": "user", "content": ranking_prompt}]

    # Get rankings from all council models in parallel
    council_models = get_council_models()

    if use_circuit_breaker:
        responses = await query_models_parallel_with_circuit_breaker(council_models, messages)
    else:
        responses = await query_models_parallel(council_models, messages)

    # Format results and collect metrics
    stage2_results = []
    stage2_metrics = []

    for model, response in responses.items():
        if response is not None:
            full_text = response.get('content', '')
            metrics = response.get('metrics', {})

            # Build metrics record
            metric_record = {
                'model_id': model,
                'stage': 'stage2',
                'response_time_ms': metrics.get('response_time_ms'),
                'input_tokens': metrics.get('input_tokens'),
                'output_tokens': metrics.get('output_tokens'),
                'total_tokens': metrics.get('total_tokens'),
                'error_type': metrics.get('error_type'),
            }

            if metrics.get('total_tokens'):
                metric_record['cost_usd'] = estimate_cost(model, metrics)

            stage2_metrics.append(metric_record)

            if full_text:
                parsed = parse_ranking_from_text(full_text)
                stage2_results.append({
                    "model": model,
                    "ranking": full_text,
                    "parsed_ranking": parsed
                })

    return stage2_results, label_to_model, stage2_metrics


def build_chairman_prompt(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]]
) -> str:
    """Build the chairman synthesis prompt."""
    stage1_text = "\n\n".join([
        f"Model: {result['model']}\nResponse: {result['response']}"
        for result in stage1_results
    ])

    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results
    ])

    return f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question: {user_query}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    use_circuit_breaker: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Stage 3: Chairman synthesizes final response.

    Args:
        user_query: The original user query
        stage1_results: Individual model responses from Stage 1
        stage2_results: Rankings from Stage 2
        use_circuit_breaker: Whether to use circuit breaker protection

    Returns:
        Tuple of (result dict, metrics dict)
    """
    chairman_prompt = build_chairman_prompt(user_query, stage1_results, stage2_results)
    messages = [{"role": "user", "content": chairman_prompt}]

    # Query the chairman model
    chairman_model = get_chairman_model()

    if use_circuit_breaker:
        response = await query_model_with_circuit_breaker(chairman_model, messages)
    else:
        response = await query_model(chairman_model, messages)

    # Build metrics
    metrics = response.get('metrics', {}) if response else {}
    stage3_metrics = {
        'model_id': chairman_model,
        'stage': 'stage3',
        'response_time_ms': metrics.get('response_time_ms'),
        'input_tokens': metrics.get('input_tokens'),
        'output_tokens': metrics.get('output_tokens'),
        'total_tokens': metrics.get('total_tokens'),
        'error_type': metrics.get('error_type'),
    }

    if metrics.get('total_tokens'):
        stage3_metrics['cost_usd'] = estimate_cost(chairman_model, metrics)

    if response is None or not response.get('content'):
        # Fallback if chairman fails
        return {
            "model": chairman_model,
            "response": "Error: Unable to generate final synthesis."
        }, stage3_metrics

    return {
        "model": chairman_model,
        "response": response.get('content', '')
    }, stage3_metrics


async def stage3_synthesize_streaming(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]]
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stage 3: Chairman synthesizes final response with token streaming.

    Yields:
        Dict with 'type' (token, done, error), 'model', and 'content'
    """
    chairman_prompt = build_chairman_prompt(user_query, stage1_results, stage2_results)
    messages = [{"role": "user", "content": chairman_prompt}]

    chairman_model = get_chairman_model()

    # Check circuit breaker
    if not CircuitBreaker.can_execute(chairman_model):
        yield {
            "type": "error",
            "model": chairman_model,
            "content": "Circuit breaker open - model temporarily unavailable"
        }
        return

    async for chunk in stream_model(chairman_model, messages):
        yield {
            "type": chunk["type"],
            "model": chairman_model,
            "content": chunk["content"]
        }

        # Record success/failure based on streaming result
        if chunk["type"] == "done":
            CircuitBreaker.record_success(chairman_model)
        elif chunk["type"] == "error":
            CircuitBreaker.record_failure(chairman_model)


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    Parse the FINAL RANKING section from the model's response.

    Args:
        ranking_text: The full text response from the model

    Returns:
        List of response labels in ranked order
    """
    import re

    # Look for "FINAL RANKING:" section
    if "FINAL RANKING:" in ranking_text:
        # Extract everything after "FINAL RANKING:"
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            # Try to extract numbered list format (e.g., "1. Response A")
            # This pattern looks for: number, period, optional space, "Response X"
            numbered_matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)
            if numbered_matches:
                # Extract just the "Response X" part
                return [re.search(r'Response [A-Z]', m).group() for m in numbered_matches]

            # Fallback: Extract all "Response X" patterns in order
            matches = re.findall(r'Response [A-Z]', ranking_section)
            return matches

    # Fallback: try to find any "Response X" patterns in order
    matches = re.findall(r'Response [A-Z]', ranking_text)
    return matches


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from anonymous labels to model names

    Returns:
        List of dicts with model name and average rank, sorted best to worst
    """
    from collections import defaultdict

    # Track positions for each model
    model_positions = defaultdict(list)

    for ranking in stage2_results:
        ranking_text = ranking['ranking']

        # Parse the ranking from the structured format
        parsed_ranking = parse_ranking_from_text(ranking_text)

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)

    # Calculate average position for each model
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x['average_rank'])

    return aggregate


async def generate_conversation_title(user_query: str) -> str:
    """
    Generate a short title for a conversation based on the first user message.

    Args:
        user_query: The first user message

    Returns:
        A short title (3-5 words)
    """
    title_prompt = f"""Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

    messages = [{"role": "user", "content": title_prompt}]

    # Use gemini-2.5-flash for title generation (fast and cheap)
    response = await query_model("google/gemini-2.5-flash", messages, timeout=30.0)

    if response is None:
        # Fallback to a generic title
        return "New Conversation"

    title = response.get('content', 'New Conversation').strip()

    # Clean up the title - remove quotes, limit length
    title = title.strip('"\'')

    # Truncate if too long
    if len(title) > 50:
        title = title[:47] + "..."

    return title


async def run_full_council(user_query: str) -> Tuple[List, List, Dict, Dict, List]:
    """
    Run the complete 3-stage council process.

    Args:
        user_query: The user's question

    Returns:
        Tuple of (stage1_results, stage2_results, stage3_result, metadata, all_metrics)
    """
    all_metrics = []

    # Stage 1: Collect individual responses
    stage1_results, stage1_metrics = await stage1_collect_responses(user_query)
    all_metrics.extend(stage1_metrics)

    # If no models responded successfully, return error
    if not stage1_results:
        return [], [], {
            "model": "error",
            "response": "All models failed to respond. Please try again."
        }, {}, all_metrics

    # Stage 2: Collect rankings
    stage2_results, label_to_model, stage2_metrics = await stage2_collect_rankings(user_query, stage1_results)
    all_metrics.extend(stage2_metrics)

    # Calculate aggregate rankings and add ranking positions to metrics
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    # Add ranking positions to stage1 metrics
    for ranking in aggregate_rankings:
        for metric in all_metrics:
            if metric['model_id'] == ranking['model'] and metric['stage'] == 'stage1':
                metric['ranking_position'] = ranking['average_rank']

    # Stage 3: Synthesize final answer
    stage3_result, stage3_metrics = await stage3_synthesize_final(
        user_query,
        stage1_results,
        stage2_results
    )
    all_metrics.append(stage3_metrics)

    # Prepare metadata
    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings
    }

    return stage1_results, stage2_results, stage3_result, metadata, all_metrics
