"""FastAPI backend for LLM Council."""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import json
import asyncio

from . import storage
from .council import (
    run_full_council,
    generate_conversation_title,
    stage1_collect_responses,
    stage2_collect_rankings,
    stage3_synthesize_final,
    stage3_synthesize_streaming,
    calculate_aggregate_rankings
)
from .openrouter import CircuitBreaker
from .db.session import init_db, migrate_from_json

# Initialize database on module load
init_db()

# Migrate existing JSON conversations if any
migrate_from_json()

app = FastAPI(title="LLM Council API", version="1.1.0")

# CORS origins - allow localhost for dev and configured frontend URL for production
cors_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]

# Add production frontend URL if configured
frontend_url = os.environ.get("FRONTEND_URL")
if frontend_url:
    cors_origins.append(frontend_url)
    # Also allow without trailing slash
    cors_origins.append(frontend_url.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models

class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    pass


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    content: str


class ConversationMetadata(BaseModel):
    """Conversation metadata for list view."""
    id: str
    created_at: str
    title: str
    message_count: int


class Conversation(BaseModel):
    """Full conversation with all messages."""
    id: str
    created_at: str
    title: str
    messages: List[Dict[str, Any]]


class CouncilConfigRequest(BaseModel):
    """Request to update council configuration."""
    council_models: Optional[List[str]] = None
    chairman_model: Optional[str] = None
    theme: Optional[str] = None


class CouncilConfigResponse(BaseModel):
    """Council configuration response."""
    council_models: List[str]
    chairman_model: str
    theme: str


class ExportRequest(BaseModel):
    """Request to export a conversation."""
    format: str = "markdown"  # markdown, json


# Health Check

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "LLM Council API", "version": "1.1.0"}


# Conversation Endpoints

@app.get("/api/conversations", response_model=List[ConversationMetadata])
async def list_conversations():
    """List all conversations (metadata only)."""
    return storage.list_conversations()


@app.post("/api/conversations", response_model=Conversation)
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    conversation = storage.create_conversation(conversation_id)
    return conversation


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """Get a specific conversation with all its messages."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    deleted = storage.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted", "id": conversation_id}


@app.post("/api/conversations/{conversation_id}/message")
async def send_message(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and run the 3-stage council process.
    Returns the complete response with all stages.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    # Add user message
    storage.add_user_message(conversation_id, request.content)

    # If this is the first message, generate a title
    if is_first_message:
        title = await generate_conversation_title(request.content)
        storage.update_conversation_title(conversation_id, title)

    # Run the 3-stage council process
    stage1_results, stage2_results, stage3_result, metadata, all_metrics = await run_full_council(
        request.content
    )

    # Add assistant message with all stages
    storage.add_assistant_message(
        conversation_id,
        stage1_results,
        stage2_results,
        stage3_result,
        metadata
    )

    # Save metrics
    message_id = storage.get_last_message_id(conversation_id)
    if message_id and all_metrics:
        storage.save_model_metrics(message_id, all_metrics)

    # Return the complete response with metadata
    return {
        "stage1": stage1_results,
        "stage2": stage2_results,
        "stage3": stage3_result,
        "metadata": metadata
    }


def get_conversation_history(conversation: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract conversation history for multi-turn context.

    Returns list of {role, content} dicts suitable for LLM API.
    """
    history = []
    for msg in conversation.get("messages", []):
        if msg["role"] == "user":
            history.append({
                "role": "user",
                "content": msg.get("content", "")
            })
        elif msg["role"] == "assistant":
            # Use the stage3 synthesis as the assistant response
            if msg.get("stage3"):
                history.append({
                    "role": "assistant",
                    "content": msg["stage3"].get("response", "")
                })
    return history


@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and stream the 3-stage council process.
    Returns Server-Sent Events as each stage completes.
    Supports multi-turn conversation context.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    # Get conversation history for multi-turn context
    conversation_history = get_conversation_history(conversation)

    async def event_generator():
        try:
            all_metrics = []

            # Add user message
            storage.add_user_message(conversation_id, request.content)

            # Start title generation in parallel (don't await yet)
            title_task = None
            if is_first_message:
                title_task = asyncio.create_task(generate_conversation_title(request.content))

            # Stage 1: Collect responses (with conversation history)
            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
            stage1_results, stage1_metrics = await stage1_collect_responses(
                request.content,
                conversation_history=conversation_history
            )
            all_metrics.extend(stage1_metrics)
            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"

            # Stage 2: Collect rankings
            yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
            stage2_results, label_to_model, stage2_metrics = await stage2_collect_rankings(request.content, stage1_results)
            all_metrics.extend(stage2_metrics)
            aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

            # Add ranking positions to stage1 metrics
            for ranking in aggregate_rankings:
                for metric in all_metrics:
                    if metric['model_id'] == ranking['model'] and metric.get('stage') == 'stage1':
                        metric['ranking_position'] = ranking['average_rank']

            yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results, 'metadata': {'label_to_model': label_to_model, 'aggregate_rankings': aggregate_rankings}})}\n\n"

            # Stage 3: Synthesize final answer
            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
            stage3_result, stage3_metrics = await stage3_synthesize_final(request.content, stage1_results, stage2_results)
            all_metrics.append(stage3_metrics)
            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result})}\n\n"

            # Wait for title generation if it was started
            if title_task:
                title = await title_task
                storage.update_conversation_title(conversation_id, title)
                yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"

            # Save complete assistant message with metadata
            metadata = {
                "label_to_model": label_to_model,
                "aggregate_rankings": aggregate_rankings
            }
            storage.add_assistant_message(
                conversation_id,
                stage1_results,
                stage2_results,
                stage3_result,
                metadata
            )

            # Save metrics
            message_id = storage.get_last_message_id(conversation_id)
            if message_id and all_metrics:
                storage.save_model_metrics(message_id, all_metrics)

            # Send completion event
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            # Send error event
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/api/conversations/{conversation_id}/message/stream-tokens")
async def send_message_stream_tokens(conversation_id: str, request: SendMessageRequest):
    """
    Send a message with full token-by-token streaming for Stage 3.
    Returns Server-Sent Events including individual tokens during synthesis.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    is_first_message = len(conversation["messages"]) == 0
    conversation_history = get_conversation_history(conversation)

    async def event_generator():
        try:
            all_metrics = []
            storage.add_user_message(conversation_id, request.content)

            title_task = None
            if is_first_message:
                title_task = asyncio.create_task(generate_conversation_title(request.content))

            # Stage 1
            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
            stage1_results, stage1_metrics = await stage1_collect_responses(
                request.content,
                conversation_history=conversation_history
            )
            all_metrics.extend(stage1_metrics)
            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"

            # Stage 2
            yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
            stage2_results, label_to_model, stage2_metrics = await stage2_collect_rankings(request.content, stage1_results)
            all_metrics.extend(stage2_metrics)
            aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

            # Add ranking positions to stage1 metrics
            for ranking in aggregate_rankings:
                for metric in all_metrics:
                    if metric['model_id'] == ranking['model'] and metric.get('stage') == 'stage1':
                        metric['ranking_position'] = ranking['average_rank']

            yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results, 'metadata': {'label_to_model': label_to_model, 'aggregate_rankings': aggregate_rankings}})}\n\n"

            # Stage 3 with token streaming
            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"

            full_response = ""
            chairman_model = ""

            async for chunk in stage3_synthesize_streaming(
                request.content, stage1_results, stage2_results
            ):
                chairman_model = chunk.get("model", "")

                if chunk["type"] == "token":
                    full_response += chunk["content"]
                    yield f"data: {json.dumps({'type': 'stage3_token', 'data': {'content': chunk['content']}})}\n\n"
                elif chunk["type"] == "done":
                    full_response = chunk["content"]
                elif chunk["type"] == "error":
                    yield f"data: {json.dumps({'type': 'error', 'message': chunk['content']})}\n\n"
                    return

            stage3_result = {
                "model": chairman_model,
                "response": full_response
            }
            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result})}\n\n"

            # Title generation
            if title_task:
                title = await title_task
                storage.update_conversation_title(conversation_id, title)
                yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"

            # Save message
            metadata = {
                "label_to_model": label_to_model,
                "aggregate_rankings": aggregate_rankings
            }
            storage.add_assistant_message(
                conversation_id,
                stage1_results,
                stage2_results,
                stage3_result,
                metadata
            )

            # Save metrics (stage3 metrics from streaming are minimal)
            stage3_metrics = {
                'model_id': chairman_model,
                'stage': 'stage3',
            }
            all_metrics.append(stage3_metrics)

            message_id = storage.get_last_message_id(conversation_id)
            if message_id and all_metrics:
                storage.save_model_metrics(message_id, all_metrics)

            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/circuit-breaker/status")
async def get_circuit_breaker_status():
    """Get circuit breaker status for all models."""
    from .config import get_council_models, get_chairman_model

    models = get_council_models() + [get_chairman_model()]
    status = {}

    for model in set(models):
        status[model] = CircuitBreaker.get_status(model)

    return {"circuit_breaker_status": status}


# Settings Endpoints

@app.get("/api/settings/council", response_model=CouncilConfigResponse)
async def get_council_config():
    """Get the current council configuration."""
    config = storage.get_council_config()
    return CouncilConfigResponse(
        council_models=config.get("council_models", []),
        chairman_model=config.get("chairman_model", ""),
        theme=config.get("theme", "light")
    )


@app.put("/api/settings/council", response_model=CouncilConfigResponse)
async def update_council_config(request: CouncilConfigRequest):
    """Update the council configuration."""
    config = storage.update_council_config(
        council_models=request.council_models,
        chairman_model=request.chairman_model,
        theme=request.theme
    )
    return CouncilConfigResponse(
        council_models=config.get("council_models", []),
        chairman_model=config.get("chairman_model", ""),
        theme=config.get("theme", "light")
    )


# Export Endpoints

@app.post("/api/conversations/{conversation_id}/export")
async def export_conversation(conversation_id: str, request: ExportRequest):
    """Export a conversation to the specified format."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if request.format == "json":
        return conversation

    elif request.format == "markdown":
        md = generate_markdown_export(conversation)
        return StreamingResponse(
            iter([md]),
            media_type="text/markdown",
            headers={
                "Content-Disposition": f"attachment; filename={conversation_id}.md"
            }
        )

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")


def generate_markdown_export(conversation: Dict[str, Any]) -> str:
    """Generate markdown export of a conversation."""
    lines = [
        f"# {conversation['title']}",
        f"",
        f"*Exported from LLM Council*",
        f"*Created: {conversation['created_at']}*",
        f"",
        "---",
        ""
    ]

    for msg in conversation["messages"]:
        if msg["role"] == "user":
            lines.extend([
                "## User",
                "",
                msg["content"],
                "",
                "---",
                ""
            ])
        else:
            lines.extend([
                "## LLM Council Response",
                "",
                "### Stage 1: Individual Responses",
                ""
            ])

            if msg.get("stage1"):
                for resp in msg["stage1"]:
                    lines.extend([
                        f"#### {resp['model']}",
                        "",
                        resp["response"],
                        ""
                    ])

            lines.extend([
                "### Stage 2: Peer Rankings",
                ""
            ])

            if msg.get("stage2"):
                for ranking in msg["stage2"]:
                    lines.extend([
                        f"#### {ranking['model']}",
                        "",
                        ranking["ranking"],
                        ""
                    ])

            lines.extend([
                "### Stage 3: Final Synthesis",
                ""
            ])

            if msg.get("stage3"):
                lines.extend([
                    f"*Chairman: {msg['stage3']['model']}*",
                    "",
                    msg["stage3"]["response"],
                    ""
                ])

            lines.extend([
                "---",
                ""
            ])

    return "\n".join(lines)


# Analytics Endpoints

@app.get("/api/analytics/summary")
async def get_analytics_summary():
    """Get aggregate analytics summary across all conversations."""
    return storage.get_analytics_summary()


@app.get("/api/analytics/models/{model_id:path}")
async def get_model_analytics(model_id: str):
    """Get detailed analytics for a specific model."""
    return storage.get_model_analytics(model_id)


@app.get("/api/analytics/recent")
async def get_recent_metrics(limit: int = 100):
    """Get recent metrics for timeline display."""
    return storage.get_recent_metrics(limit)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
