"""
Dialectic Chain: A revolutionary workflow for life-changing AI responses.

Instead of parallel voting (which produces averaged/generic output), this uses
a sequential chain where each model builds on, challenges, and deepens the previous.

The 4-Stage Dialectic:
1. First Responder: Comprehensive initial answer
2. Devil's Advocate: Challenges assumptions, finds gaps, offers alternatives
3. Deep Insight: Synthesizes the debate into non-obvious wisdom
4. Action Coach: Transforms insight into specific, actionable steps

This produces output that is genuinely useful because:
- Each stage forces DEEPER thinking, not just agreement
- The challenge stage prevents generic "safe" answers
- The action stage ensures practical value
"""

from typing import List, Dict, Any, Tuple, Optional, AsyncGenerator
from .openrouter import query_model, stream_model, CircuitBreaker
from .config import get_council_models, get_chairman_model, get_model_parameters


# =============================================================================
# DIALECTIC CHAIN PROMPTS
# =============================================================================

FIRST_RESPONDER_PROMPT = """You are the First Responder in a dialectic chain. Your job is to provide a comprehensive, well-reasoned initial answer.

## Your Mandate
- Go DEEP, not wide. Better to thoroughly address the core issue than superficially cover everything.
- Be SPECIFIC. Use concrete examples, numbers, and real-world references.
- Take a STANCE. Don't hedge. Give your actual opinion based on evidence.
- Anticipate objections and address them proactively.

## What Makes a Great First Response
✓ Specific recommendations, not vague suggestions
✓ Real examples or case studies
✓ Clear reasoning for why this approach beats alternatives
✓ Acknowledgment of trade-offs with specific guidance on when each applies

## What to Avoid
✗ "It depends" without specifying on what
✗ Lists of generic options without ranking them
✗ Hedging language ("you might consider", "some people think")
✗ Surface-level overviews that don't help the user decide

Remember: Your response will be CHALLENGED by a Devil's Advocate. Write something worth defending.

---

USER'S QUESTION:
{user_query}

YOUR COMPREHENSIVE RESPONSE:"""


DEVILS_ADVOCATE_PROMPT = """You are the Devil's Advocate in a dialectic chain. Your job is to CHALLENGE the previous response and push thinking deeper.

## The First Responder Said:
{previous_response}

## Your Mandate
- Find the WEAKNESSES in their argument. What did they miss? What's wrong?
- Offer ALTERNATIVE perspectives they didn't consider.
- Ask the questions they should have addressed but didn't.
- Push back on any vague or generic advice - demand specificity.

## Your Analysis Should Cover:

### 1. Critical Gaps
What crucial considerations did they completely miss? What would change their answer?

### 2. Flawed Assumptions
What are they assuming that might not be true for this user? What context would change everything?

### 3. Better Alternatives
Is there a completely different approach that might work better? Why?

### 4. The Hard Truth
What uncomfortable reality are they dancing around? What would a truly honest expert say?

## Important
- Be constructive, not just contrarian. Every critique should point toward a better answer.
- If they got something genuinely right, acknowledge it briefly, then move on to what needs work.
- Your challenge should make the final synthesis BETTER, not just different.

---

ORIGINAL QUESTION: {user_query}

YOUR CRITICAL ANALYSIS:"""


DEEP_INSIGHT_PROMPT = """You are the Deep Insight Synthesizer. You've witnessed a debate and must now extract the NON-OBVIOUS wisdom.

## The Exchange So Far:

### First Responder's Answer:
{first_response}

### Devil's Advocate's Challenge:
{challenge}

## Your Mandate
Synthesize this into an insight that neither response achieved alone. You're looking for:

### 1. The Synthesis
Where do both perspectives point to a deeper truth? What emerges from their tension?

### 2. The Non-Obvious Insight
What would an expert with 20 years experience say that would surprise even the previous responders?

### 3. The Key Leverage Point
If the user could only focus on ONE thing, what would create the most value? Why?

### 4. The Trap to Avoid
What's the most common mistake people make here that would be disastrous?

## Your Output Should Be
- Genuinely insightful (not just restating what was said)
- Memorable (something they'll actually remember tomorrow)
- Unconventional (at least partially - if it's totally obvious, dig deeper)

---

ORIGINAL QUESTION: {user_query}

YOUR SYNTHESIZED INSIGHT:"""


ACTION_COACH_PROMPT = """You are the Action Coach. You transform insight into SPECIFIC, ACTIONABLE steps.

## The Insight So Far:
{synthesis}

## Your Mandate
Make this REAL and ACTIONABLE. The user should finish reading your response knowing EXACTLY what to do.

### Your Output Must Include:

## Immediate Action (Do This Today)
One specific thing they can do in the next 24 hours. Be precise:
- Not "start exercising" but "Do a 20-minute walk at 7am tomorrow"
- Not "network more" but "Send a LinkedIn message to 3 people in your target role"
- Not "learn more" but "Read chapters 1-3 of [specific book] this weekend"

## The 7-Day Sprint
A concrete mini-plan for the next week. Day by day if helpful.

## Success Metrics
How will they KNOW if this is working? Give them specific indicators to watch for.

## The One Thing That Will Make This Fail
The most common failure mode - and how to avoid it.

## Resources (Optional)
Only include if genuinely helpful. Specific tools, books, or references - not generic categories.

---

ORIGINAL QUESTION: {user_query}

YOUR ACTION PLAN:"""


FINAL_SYNTHESIS_PROMPT = """You are the Final Synthesizer. Combine the entire dialectic chain into one powerful, life-changing response.

## The Complete Dialectic:

### First Response:
{first_response}

### Devil's Advocate Challenge:
{challenge}

### Deep Insight:
{insight}

### Action Plan:
{action_plan}

## Your Task
Create a SINGLE, UNIFIED response that:

1. **Opens with the key insight** - The non-obvious truth that makes this valuable
2. **Provides the strategic context** - Why this matters and the bigger picture
3. **Gives the specific action plan** - What to do, starting today
4. **Closes with the crucial warning** - What to avoid

## Style Guidelines
- Write directly to the user (not about the process)
- Be confident and direct - you've earned it through rigorous analysis
- Make it memorable - they should think about this tomorrow
- Be specific - vague advice is worthless advice

---

ORIGINAL QUESTION: {user_query}

YOUR LIFE-CHANGING RESPONSE:"""


# =============================================================================
# DIALECTIC CHAIN IMPLEMENTATION
# =============================================================================

async def run_dialectic_chain(
    user_query: str,
    models: Optional[List[str]] = None,
    stream_final: bool = True
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Run the full dialectic chain with streaming progress updates.

    Yields events:
        - {"type": "stage_start", "stage": "first_responder"|"devils_advocate"|"deep_insight"|"action_coach"|"final_synthesis"}
        - {"type": "stage_complete", "stage": str, "content": str, "model": str}
        - {"type": "final_token", "content": str}
        - {"type": "complete", "final_response": str}
        - {"type": "error", "message": str}
    """
    council_models = models or get_council_models()
    chairman_model = get_chairman_model()
    model_params = get_model_parameters()

    # Assign roles - use different models for different perspectives
    # If we have 4+ models, use them; otherwise cycle through what we have
    if len(council_models) >= 4:
        first_responder = council_models[0]
        devils_advocate = council_models[1]
        deep_insight_model = council_models[2]
        action_coach = council_models[3]
    else:
        # Cycle through available models
        first_responder = council_models[0]
        devils_advocate = council_models[1 % len(council_models)]
        deep_insight_model = council_models[2 % len(council_models)]
        action_coach = council_models[3 % len(council_models)]

    final_synthesizer = chairman_model

    try:
        # Stage 1: First Responder
        yield {"type": "stage_start", "stage": "first_responder", "model": first_responder}

        first_response = await _query_stage(
            first_responder,
            FIRST_RESPONDER_PROMPT.format(user_query=user_query),
            model_params
        )

        if not first_response:
            yield {"type": "error", "message": "First responder failed to generate response"}
            return

        yield {"type": "stage_complete", "stage": "first_responder", "content": first_response, "model": first_responder}

        # Stage 2: Devil's Advocate
        yield {"type": "stage_start", "stage": "devils_advocate", "model": devils_advocate}

        challenge = await _query_stage(
            devils_advocate,
            DEVILS_ADVOCATE_PROMPT.format(
                user_query=user_query,
                previous_response=first_response
            ),
            model_params
        )

        if not challenge:
            yield {"type": "error", "message": "Devil's advocate failed to generate response"}
            return

        yield {"type": "stage_complete", "stage": "devils_advocate", "content": challenge, "model": devils_advocate}

        # Stage 3: Deep Insight
        yield {"type": "stage_start", "stage": "deep_insight", "model": deep_insight_model}

        insight = await _query_stage(
            deep_insight_model,
            DEEP_INSIGHT_PROMPT.format(
                user_query=user_query,
                first_response=first_response,
                challenge=challenge
            ),
            model_params
        )

        if not insight:
            yield {"type": "error", "message": "Deep insight synthesis failed"}
            return

        yield {"type": "stage_complete", "stage": "deep_insight", "content": insight, "model": deep_insight_model}

        # Stage 4: Action Coach
        yield {"type": "stage_start", "stage": "action_coach", "model": action_coach}

        action_plan = await _query_stage(
            action_coach,
            ACTION_COACH_PROMPT.format(
                user_query=user_query,
                synthesis=insight
            ),
            model_params
        )

        if not action_plan:
            yield {"type": "error", "message": "Action coach failed to generate response"}
            return

        yield {"type": "stage_complete", "stage": "action_coach", "content": action_plan, "model": action_coach}

        # Stage 5: Final Synthesis (streamed)
        yield {"type": "stage_start", "stage": "final_synthesis", "model": final_synthesizer}

        final_prompt = FINAL_SYNTHESIS_PROMPT.format(
            user_query=user_query,
            first_response=first_response,
            challenge=challenge,
            insight=insight,
            action_plan=action_plan
        )

        full_response = ""

        if stream_final:
            # Stream the final synthesis token by token
            async for chunk in stream_model(final_synthesizer, [{"role": "user", "content": final_prompt}]):
                if chunk["type"] == "token":
                    full_response += chunk["content"]
                    yield {"type": "final_token", "content": chunk["content"]}
                elif chunk["type"] == "done":
                    full_response = chunk["content"]
                elif chunk["type"] == "error":
                    yield {"type": "error", "message": chunk["content"]}
                    return
        else:
            # Non-streaming final synthesis
            response = await query_model(final_synthesizer, [{"role": "user", "content": final_prompt}])
            if response:
                full_response = response.get("content", "")

        yield {"type": "stage_complete", "stage": "final_synthesis", "content": full_response, "model": final_synthesizer}
        yield {"type": "complete", "final_response": full_response}

    except Exception as e:
        yield {"type": "error", "message": str(e)}


async def _query_stage(model: str, prompt: str, model_params: Dict[str, Any]) -> Optional[str]:
    """Query a single stage in the dialectic chain."""

    # Check circuit breaker
    if not CircuitBreaker.can_execute(model):
        return None

    # Get model-specific parameters
    params = {}
    if model in model_params:
        mp = model_params[model]
        if "temperature" in mp:
            params["temperature"] = mp["temperature"]
        if "max_tokens" in mp:
            params["max_tokens"] = mp["max_tokens"]

    try:
        response = await query_model(model, [{"role": "user", "content": prompt}], **params)

        if response and response.get("content"):
            CircuitBreaker.record_success(model)
            return response["content"]
        else:
            CircuitBreaker.record_failure(model)
            return None

    except Exception as e:
        CircuitBreaker.record_failure(model)
        return None


async def run_dialectic_simple(user_query: str) -> Dict[str, Any]:
    """
    Run dialectic chain and return complete result (non-streaming).

    Returns:
        Dict with stages and final_response
    """
    result = {
        "stages": {},
        "final_response": "",
        "error": None
    }

    async for event in run_dialectic_chain(user_query, stream_final=False):
        if event["type"] == "stage_complete":
            result["stages"][event["stage"]] = {
                "content": event["content"],
                "model": event["model"]
            }
        elif event["type"] == "complete":
            result["final_response"] = event["final_response"]
        elif event["type"] == "error":
            result["error"] = event["message"]
            break

    return result
