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
from .config import get_council_models, get_chairman_model, get_model_parameters


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


def parse_confidence_from_response(response: str) -> Tuple[str, Optional[float]]:
    """
    Parse confidence score from a response.

    Looks for patterns like:
    - "Confidence: 8/10"
    - "Confidence: 85%"
    - "Confidence: High"
    - "[CONFIDENCE: 0.8]"

    Args:
        response: The model's response text

    Returns:
        Tuple of (cleaned response without confidence marker, confidence score 0-1)
    """
    import re

    if not response:
        return response, None

    # Pattern 1: [CONFIDENCE: X] at the end (preferred format)
    match = re.search(r'\[CONFIDENCE:\s*(\d*\.?\d+)\]\s*$', response)
    if match:
        score = float(match.group(1))
        if score > 1:
            score = score / 100 if score <= 100 else score / 10
        cleaned = re.sub(r'\s*\[CONFIDENCE:\s*\d*\.?\d+\]\s*$', '', response).strip()
        return cleaned, min(1.0, max(0.0, score))

    # Pattern 2: "Confidence: X/10" or "Confidence: X%"
    match = re.search(r'[Cc]onfidence:\s*(\d+(?:\.\d+)?)\s*[/%]?\s*(?:10|100)?', response)
    if match:
        value = float(match.group(1))
        if value > 1:
            score = value / 100 if value <= 100 else value / 10
        else:
            score = value
        return response, min(1.0, max(0.0, score))

    # Pattern 3: Word-based confidence
    confidence_words = {
        'very high': 0.95, 'extremely high': 0.95,
        'high': 0.8, 'confident': 0.8,
        'moderate': 0.6, 'medium': 0.6,
        'low': 0.4, 'uncertain': 0.3,
        'very low': 0.2, 'not confident': 0.2,
    }
    for word, score in confidence_words.items():
        if re.search(rf'[Cc]onfidence:\s*{word}', response, re.IGNORECASE):
            return response, score

    return response, None


async def stage1_collect_responses(
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    use_circuit_breaker: bool = True,
    enable_confidence: bool = False
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Stage 1: Collect individual responses from all council models.

    Args:
        user_query: The user's question
        conversation_history: Previous messages for multi-turn context
        use_circuit_breaker: Whether to use circuit breaker protection
        enable_confidence: Whether to ask models for confidence scores

    Returns:
        Tuple of (results list, metrics list)
    """
    messages = build_messages_with_history(user_query, conversation_history)

    # Add confidence prompt if enabled
    if enable_confidence:
        # Find the user message and append confidence instruction
        for msg in messages:
            if msg['role'] == 'user':
                msg['content'] = msg['content'] + '''

Please end your response with a confidence score in this exact format:
[CONFIDENCE: X.X]
Where X.X is a number between 0.0 (not confident) and 1.0 (very confident).'''
                break

    # Query all models in parallel
    council_models = get_council_models()
    model_params = get_model_parameters()

    if use_circuit_breaker:
        responses = await query_models_parallel_with_circuit_breaker(council_models, messages, model_params)
    else:
        responses = await query_models_parallel(council_models, messages, model_params)

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
                result = {
                    "model": model,
                    "response": content
                }

                # Parse confidence score if enabled
                if enable_confidence:
                    cleaned_response, confidence = parse_confidence_from_response(content)
                    result["response"] = cleaned_response
                    result["confidence"] = confidence

                stage1_results.append(result)

    return stage1_results, stage1_metrics


def build_ranking_prompt(
    user_query: str,
    responses_text: str,
    ranking_criteria: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Build the Stage 2 ranking prompt with optional custom criteria.

    Args:
        user_query: The original user question
        responses_text: Formatted anonymized responses
        ranking_criteria: Optional list of criteria to evaluate against

    Returns:
        The formatted ranking prompt
    """
    # Build criteria section if custom criteria provided
    criteria_text = ""
    if ranking_criteria:
        enabled_criteria = [c for c in ranking_criteria if c.get('enabled', True)]
        if enabled_criteria:
            criteria_list = "\n".join([
                f"- **{c['name']}**: {c.get('description', '')}"
                for c in enabled_criteria
            ])
            criteria_text = f"""
Evaluate each response based on these criteria:
{criteria_list}

"""

    return f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually.{criteria_text} For each response, explain what it does well and what it does poorly.
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


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    use_circuit_breaker: bool = True,
    ranking_criteria: Optional[List[Dict[str, Any]]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]]]:
    """
    Stage 2: Each model ranks the anonymized responses.

    Args:
        user_query: The original user query
        stage1_results: Results from Stage 1
        use_circuit_breaker: Whether to use circuit breaker protection
        ranking_criteria: Optional custom ranking criteria

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

    ranking_prompt = build_ranking_prompt(user_query, responses_text, ranking_criteria)

    messages = [{"role": "user", "content": ranking_prompt}]

    # Get rankings from all council models in parallel
    council_models = get_council_models()
    model_params = get_model_parameters()

    if use_circuit_breaker:
        responses = await query_models_parallel_with_circuit_breaker(council_models, messages, model_params)
    else:
        responses = await query_models_parallel(council_models, messages, model_params)

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
    model_params = get_model_parameters()

    # Get chairman-specific parameters
    chairman_params = {}
    if chairman_model in model_params:
        mp = model_params[chairman_model]
        if "temperature" in mp:
            chairman_params["temperature"] = mp["temperature"]
        if "max_tokens" in mp:
            chairman_params["max_tokens"] = mp["max_tokens"]

    if use_circuit_breaker:
        response = await query_model_with_circuit_breaker(chairman_model, messages, **chairman_params)
    else:
        response = await query_model(chairman_model, messages, **chairman_params)

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
    model_params = get_model_parameters()

    # Get chairman-specific parameters for streaming
    stream_params = {}
    if chairman_model in model_params:
        mp = model_params[chairman_model]
        if "temperature" in mp:
            stream_params["temperature"] = mp["temperature"]
        if "max_tokens" in mp:
            stream_params["max_tokens"] = mp["max_tokens"]

    # Check circuit breaker
    if not CircuitBreaker.can_execute(chairman_model):
        yield {
            "type": "error",
            "model": chairman_model,
            "content": "Circuit breaker open - model temporarily unavailable"
        }
        return

    async for chunk in stream_model(chairman_model, messages, **stream_params):
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
    label_to_model: Dict[str, str],
    model_weights: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models with optional weighted voting.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from anonymous labels to model names
        model_weights: Optional dict mapping model_id to voting weight (default 1.0)

    Returns:
        List of dicts with model name and average rank, sorted best to worst
    """
    from collections import defaultdict

    # Track positions for each model with weights
    model_positions = defaultdict(list)
    model_weighted_positions = defaultdict(list)

    for ranking in stage2_results:
        ranking_text = ranking['ranking']
        ranker_model = ranking.get('model', '')

        # Get weight for the ranking model (default 1.0)
        weight = 1.0
        if model_weights and ranker_model in model_weights:
            weight = model_weights[ranker_model]

        # Parse the ranking from the structured format
        parsed_ranking = parse_ranking_from_text(ranking_text)

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)
                model_weighted_positions[model_name].append((position, weight))

    # Calculate average position for each model
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            # Calculate unweighted average
            avg_rank = sum(positions) / len(positions)

            # Calculate weighted average if weights provided
            weighted_positions = model_weighted_positions[model]
            total_weight = sum(w for _, w in weighted_positions)
            weighted_avg = sum(p * w for p, w in weighted_positions) / total_weight if total_weight > 0 else avg_rank

            aggregate.append({
                "model": model,
                "average_rank": round(weighted_avg, 2),
                "unweighted_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x['average_rank'])

    return aggregate


def calculate_dissent_metrics(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> Dict[str, Any]:
    """
    Calculate disagreement/dissent metrics from Stage 2 rankings.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from anonymous labels to model names

    Returns:
        Dict with dissent metrics: agreement_score, controversies, unanimous_winner
    """
    from collections import defaultdict

    if len(stage2_results) < 2:
        return {"agreement_score": 1.0, "controversies": [], "unanimous_winner": None}

    # Track who each model ranked first and last
    first_place_votes = defaultdict(list)  # model -> list of rankers who put them first
    last_place_votes = defaultdict(list)   # model -> list of rankers who put them last
    all_rankings = []  # list of (ranker, [ranked models in order])

    for ranking in stage2_results:
        ranker = ranking.get('model', 'unknown')
        parsed = parse_ranking_from_text(ranking['ranking'])

        if parsed:
            # Convert labels to model names
            ranked_models = [label_to_model.get(label, label) for label in parsed]
            all_rankings.append((ranker, ranked_models))

            if ranked_models:
                first_place_votes[ranked_models[0]].append(ranker)
                last_place_votes[ranked_models[-1]].append(ranker)

    # Calculate agreement score (0-1, 1 = perfect agreement)
    # Based on Kendall's W (coefficient of concordance) simplified
    num_rankers = len(all_rankings)
    if num_rankers < 2:
        return {"agreement_score": 1.0, "controversies": [], "unanimous_winner": None}

    # Check for unanimous first place
    unanimous_winner = None
    for model, voters in first_place_votes.items():
        if len(voters) == num_rankers:
            unanimous_winner = model
            break

    # Find controversial responses (models with both first and last place votes)
    controversies = []
    for model in set(first_place_votes.keys()) | set(last_place_votes.keys()):
        first_votes = len(first_place_votes.get(model, []))
        last_votes = len(last_place_votes.get(model, []))
        if first_votes > 0 and last_votes > 0:
            controversies.append({
                "model": model,
                "first_place_votes": first_votes,
                "last_place_votes": last_votes,
                "controversy_score": round((first_votes + last_votes) / num_rankers, 2)
            })

    # Sort controversies by controversy score
    controversies.sort(key=lambda x: x['controversy_score'], reverse=True)

    # Calculate simple agreement score based on position variance
    # Lower variance = higher agreement
    from collections import defaultdict
    position_lists = defaultdict(list)
    for ranker, ranked_models in all_rankings:
        for pos, model in enumerate(ranked_models, start=1):
            position_lists[model].append(pos)

    if position_lists:
        variances = []
        for model, positions in position_lists.items():
            if len(positions) > 1:
                mean = sum(positions) / len(positions)
                variance = sum((p - mean) ** 2 for p in positions) / len(positions)
                variances.append(variance)

        if variances:
            avg_variance = sum(variances) / len(variances)
            # Normalize: variance of 0 = agreement 1.0, high variance = lower agreement
            # Max possible variance for n items is roughly (n-1)^2 / 4
            max_variance = (len(label_to_model) - 1) ** 2 / 4 if len(label_to_model) > 1 else 1
            agreement_score = max(0, 1 - (avg_variance / max_variance)) if max_variance > 0 else 1.0
        else:
            agreement_score = 1.0
    else:
        agreement_score = 1.0

    return {
        "agreement_score": round(agreement_score, 2),
        "controversies": controversies,
        "unanimous_winner": unanimous_winner
    }


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

    if response is None or response.get('content') is None:
        # Fallback to a generic title
        return "New Conversation"

    title = response.get('content', 'New Conversation').strip()

    # Clean up the title - remove quotes, limit length
    title = title.strip('"\'')

    # Truncate if too long
    if len(title) > 50:
        title = title[:47] + "..."

    return title


async def run_full_council(
    user_query: str,
    ranking_criteria: Optional[List[Dict[str, Any]]] = None,
    model_weights: Optional[Dict[str, float]] = None,
    enable_dissent_tracking: bool = True,
    enable_confidence: bool = False
) -> Tuple[List, List, Dict, Dict, List]:
    """
    Run the complete 3-stage council process.

    Args:
        user_query: The user's question
        ranking_criteria: Optional custom ranking criteria for Stage 2
        model_weights: Optional model voting weights for aggregate rankings
        enable_dissent_tracking: Whether to calculate dissent metrics
        enable_confidence: Whether to ask models for confidence scores

    Returns:
        Tuple of (stage1_results, stage2_results, stage3_result, metadata, all_metrics)
    """
    all_metrics = []

    # Stage 1: Collect individual responses
    stage1_results, stage1_metrics = await stage1_collect_responses(
        user_query,
        enable_confidence=enable_confidence
    )
    all_metrics.extend(stage1_metrics)

    # If no models responded successfully, return error
    if not stage1_results:
        return [], [], {
            "model": "error",
            "response": "All models failed to respond. Please try again."
        }, {}, all_metrics

    # Stage 2: Collect rankings with optional custom criteria
    stage2_results, label_to_model, stage2_metrics = await stage2_collect_rankings(
        user_query,
        stage1_results,
        ranking_criteria=ranking_criteria
    )
    all_metrics.extend(stage2_metrics)

    # Calculate aggregate rankings with optional weighted voting
    aggregate_rankings = calculate_aggregate_rankings(
        stage2_results,
        label_to_model,
        model_weights=model_weights
    )

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

    # Calculate dissent metrics if enabled
    if enable_dissent_tracking and len(stage2_results) >= 2:
        dissent_metrics = calculate_dissent_metrics(stage2_results, label_to_model)
        metadata["dissent"] = dissent_metrics

    return stage1_results, stage2_results, stage3_result, metadata, all_metrics
