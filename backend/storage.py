"""Storage layer for LLM Council - SQLite implementation."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy import func

from .db.session import get_db_context
from .db.models import Conversation, Message, Settings, ModelMetric, Preset


def create_conversation(conversation_id: str) -> Dict[str, Any]:
    """
    Create a new conversation.

    Args:
        conversation_id: Unique identifier for the conversation

    Returns:
        New conversation dict
    """
    with get_db_context() as db:
        conversation = Conversation(
            id=conversation_id,
            title="New Conversation"
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        return conversation.to_dict()


def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a conversation from storage.

    Args:
        conversation_id: Unique identifier for the conversation

    Returns:
        Conversation dict or None if not found
    """
    with get_db_context() as db:
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()

        if conversation is None:
            return None

        return conversation.to_dict()


def list_conversations() -> List[Dict[str, Any]]:
    """
    List all conversations (metadata only).

    Returns:
        List of conversation metadata dicts
    """
    with get_db_context() as db:
        conversations = db.query(Conversation).order_by(
            Conversation.created_at.desc()
        ).all()

        return [conv.to_metadata() for conv in conversations]


def add_user_message(conversation_id: str, content: str):
    """
    Add a user message to a conversation.

    Args:
        conversation_id: Conversation identifier
        content: User message content
    """
    with get_db_context() as db:
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()

        if conversation is None:
            raise ValueError(f"Conversation {conversation_id} not found")

        message = Message(
            conversation_id=conversation_id,
            role="user",
            content=content
        )
        db.add(message)

        # Update conversation timestamp
        conversation.updated_at = datetime.utcnow()

        db.commit()


def add_assistant_message(
    conversation_id: str,
    stage1: List[Dict[str, Any]],
    stage2: List[Dict[str, Any]],
    stage3: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Add an assistant message with all 3 stages to a conversation.

    Args:
        conversation_id: Conversation identifier
        stage1: List of individual model responses
        stage2: List of model rankings
        stage3: Final synthesized response
        metadata: Optional metadata (label_to_model, aggregate_rankings)
    """
    with get_db_context() as db:
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()

        if conversation is None:
            raise ValueError(f"Conversation {conversation_id} not found")

        message = Message(
            conversation_id=conversation_id,
            role="assistant",
            stage1=stage1,
            stage2=stage2,
            stage3=stage3,
            metadata_json=metadata
        )
        db.add(message)

        # Update conversation timestamp
        conversation.updated_at = datetime.utcnow()

        db.commit()


def update_conversation_title(conversation_id: str, title: str):
    """
    Update the title of a conversation.

    Args:
        conversation_id: Conversation identifier
        title: New title for the conversation
    """
    with get_db_context() as db:
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()

        if conversation is None:
            raise ValueError(f"Conversation {conversation_id} not found")

        conversation.title = title
        db.commit()


def delete_conversation(conversation_id: str) -> bool:
    """
    Delete a conversation and all its messages.

    Args:
        conversation_id: Conversation identifier

    Returns:
        True if deleted, False if not found
    """
    with get_db_context() as db:
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()

        if conversation is None:
            return False

        db.delete(conversation)
        db.commit()
        return True


# Settings Management

def get_settings(key: str) -> Optional[Any]:
    """
    Get a settings value by key.

    Args:
        key: Settings key

    Returns:
        Settings value or None if not found
    """
    with get_db_context() as db:
        setting = db.query(Settings).filter(Settings.key == key).first()
        return setting.value if setting else None


def set_settings(key: str, value: Any):
    """
    Set a settings value.

    Args:
        key: Settings key
        value: Settings value (must be JSON serializable)
    """
    with get_db_context() as db:
        setting = db.query(Settings).filter(Settings.key == key).first()

        if setting:
            setting.value = value
        else:
            setting = Settings(key=key, value=value)
            db.add(setting)

        db.commit()


def get_council_config() -> Dict[str, Any]:
    """
    Get the current council configuration.

    Returns:
        Council configuration dict
    """
    config = get_settings("council_config")
    if config is None:
        return Settings.get_default_council_config()
    return config


def update_council_config(
    council_models: Optional[List[str]] = None,
    chairman_model: Optional[str] = None,
    theme: Optional[str] = None,
    ranking_criteria: Optional[List[Dict[str, Any]]] = None,
    model_weights: Optional[Dict[str, float]] = None,
    model_parameters: Optional[Dict[str, Dict[str, Any]]] = None,
    enable_confidence: Optional[bool] = None,
    enable_dissent_tracking: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Update the council configuration.

    Args:
        council_models: List of model identifiers for the council
        chairman_model: Model identifier for the chairman
        theme: UI theme ('light' or 'dark')
        ranking_criteria: List of ranking criteria with weights
        model_weights: Dict mapping model_id to voting weight
        model_parameters: Dict mapping model_id to {temperature, max_tokens}
        enable_confidence: Whether to request confidence scores
        enable_dissent_tracking: Whether to track disagreements

    Returns:
        Updated configuration dict
    """
    config = get_council_config()

    if council_models is not None:
        config["council_models"] = council_models
    if chairman_model is not None:
        config["chairman_model"] = chairman_model
    if theme is not None:
        config["theme"] = theme
    if ranking_criteria is not None:
        config["ranking_criteria"] = ranking_criteria
    if model_weights is not None:
        config["model_weights"] = model_weights
    if model_parameters is not None:
        config["model_parameters"] = model_parameters
    if enable_confidence is not None:
        config["enable_confidence"] = enable_confidence
    if enable_dissent_tracking is not None:
        config["enable_dissent_tracking"] = enable_dissent_tracking

    set_settings("council_config", config)
    return config


# Analytics Functions

def save_model_metrics(
    message_id: int,
    metrics_list: List[Dict[str, Any]]
):
    """
    Save model metrics for a message.

    Args:
        message_id: The message ID to associate metrics with
        metrics_list: List of metric dicts with model_id, response_time_ms, tokens, etc.
    """
    with get_db_context() as db:
        for metrics in metrics_list:
            metric = ModelMetric(
                message_id=message_id,
                model_id=metrics.get('model_id', ''),
                response_time_ms=metrics.get('response_time_ms'),
                input_tokens=metrics.get('input_tokens'),
                output_tokens=metrics.get('output_tokens'),
                total_tokens=metrics.get('total_tokens'),
                ranking_position=metrics.get('ranking_position'),
                cost_usd=metrics.get('cost_usd'),
                error_type=metrics.get('error_type'),
            )
            db.add(metric)
        db.commit()


def get_last_message_id(conversation_id: str) -> Optional[int]:
    """Get the ID of the last message in a conversation."""
    with get_db_context() as db:
        message = db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.id.desc()).first()
        return message.id if message else None


def get_analytics_summary() -> Dict[str, Any]:
    """
    Get aggregate analytics across all conversations.

    Returns:
        Summary dict with total queries, model stats, etc.
    """
    with get_db_context() as db:
        # Total metrics
        total_metrics = db.query(func.count(ModelMetric.id)).scalar() or 0

        # Aggregate by model
        model_stats = db.query(
            ModelMetric.model_id,
            func.count(ModelMetric.id).label('query_count'),
            func.avg(ModelMetric.response_time_ms).label('avg_response_time'),
            func.sum(ModelMetric.total_tokens).label('total_tokens'),
            func.sum(ModelMetric.cost_usd).label('total_cost'),
            func.avg(ModelMetric.ranking_position).label('avg_ranking'),
        ).group_by(ModelMetric.model_id).all()

        models = []
        for stat in model_stats:
            models.append({
                'model_id': stat.model_id,
                'query_count': stat.query_count,
                'avg_response_time_ms': round(stat.avg_response_time or 0, 0),
                'total_tokens': stat.total_tokens or 0,
                'total_cost_usd': round(stat.total_cost or 0, 4),
                'avg_ranking': round(stat.avg_ranking or 0, 2) if stat.avg_ranking else None,
            })

        # Sort by query count descending
        models.sort(key=lambda x: x['query_count'], reverse=True)

        # Total conversations and messages
        total_conversations = db.query(func.count(Conversation.id)).scalar() or 0
        total_messages = db.query(func.count(Message.id)).scalar() or 0

        # Total tokens and cost
        totals = db.query(
            func.sum(ModelMetric.total_tokens).label('tokens'),
            func.sum(ModelMetric.cost_usd).label('cost'),
        ).first()

        return {
            'total_conversations': total_conversations,
            'total_messages': total_messages,
            'total_api_calls': total_metrics,
            'total_tokens': totals.tokens or 0,
            'total_cost_usd': round(totals.cost or 0, 4),
            'models': models,
        }


def get_model_analytics(model_id: str) -> Dict[str, Any]:
    """
    Get detailed analytics for a specific model.

    Args:
        model_id: The model identifier

    Returns:
        Detailed stats for the model
    """
    with get_db_context() as db:
        metrics = db.query(ModelMetric).filter(
            ModelMetric.model_id == model_id
        ).all()

        if not metrics:
            return {'model_id': model_id, 'query_count': 0}

        response_times = [m.response_time_ms for m in metrics if m.response_time_ms]
        tokens = [m.total_tokens for m in metrics if m.total_tokens]
        rankings = [m.ranking_position for m in metrics if m.ranking_position]
        costs = [m.cost_usd for m in metrics if m.cost_usd]
        errors = [m for m in metrics if m.error_type]

        return {
            'model_id': model_id,
            'query_count': len(metrics),
            'response_time': {
                'avg_ms': round(sum(response_times) / len(response_times), 0) if response_times else 0,
                'min_ms': min(response_times) if response_times else 0,
                'max_ms': max(response_times) if response_times else 0,
            },
            'tokens': {
                'total': sum(tokens) if tokens else 0,
                'avg_per_query': round(sum(tokens) / len(tokens), 0) if tokens else 0,
            },
            'ranking': {
                'avg_position': round(sum(rankings) / len(rankings), 2) if rankings else None,
                'times_ranked': len(rankings),
            },
            'cost': {
                'total_usd': round(sum(costs), 4) if costs else 0,
                'avg_per_query': round(sum(costs) / len(costs), 6) if costs else 0,
            },
            'errors': {
                'count': len(errors),
                'rate': round(len(errors) / len(metrics) * 100, 1),
            },
        }


def get_recent_metrics(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get recent metrics for timeline display.

    Args:
        limit: Maximum number of metrics to return

    Returns:
        List of recent metric records
    """
    with get_db_context() as db:
        metrics = db.query(ModelMetric).order_by(
            ModelMetric.created_at.desc()
        ).limit(limit).all()

        return [{
            'id': m.id,
            'model_id': m.model_id,
            'response_time_ms': m.response_time_ms,
            'total_tokens': m.total_tokens,
            'cost_usd': m.cost_usd,
            'ranking_position': m.ranking_position,
            'error_type': m.error_type,
            'created_at': m.created_at.isoformat() if m.created_at else None,
        } for m in metrics]


# Preset Management

def list_presets() -> List[Dict[str, Any]]:
    """
    List all presets.

    Returns:
        List of preset dicts
    """
    with get_db_context() as db:
        presets = db.query(Preset).order_by(Preset.name).all()
        return [p.to_dict() for p in presets]


def get_preset(preset_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a preset by ID.

    Args:
        preset_id: Preset identifier

    Returns:
        Preset dict or None if not found
    """
    with get_db_context() as db:
        preset = db.query(Preset).filter(Preset.id == preset_id).first()
        return preset.to_dict() if preset else None


def create_preset(
    name: str,
    council_models: List[str],
    chairman_model: str,
    description: Optional[str] = None,
    model_weights: Optional[Dict[str, float]] = None,
    model_parameters: Optional[Dict[str, Dict[str, Any]]] = None,
    ranking_criteria: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Create a new preset.

    Args:
        name: Preset name
        council_models: List of model identifiers
        chairman_model: Chairman model identifier
        description: Optional description
        model_weights: Optional model weight overrides
        model_parameters: Optional per-model parameters
        ranking_criteria: Optional ranking criteria

    Returns:
        Created preset dict
    """
    with get_db_context() as db:
        preset = Preset(
            name=name,
            description=description,
            council_models=council_models,
            chairman_model=chairman_model,
            model_weights=model_weights,
            model_parameters=model_parameters,
            ranking_criteria=ranking_criteria
        )
        db.add(preset)
        db.commit()
        db.refresh(preset)
        return preset.to_dict()


def update_preset(
    preset_id: int,
    name: Optional[str] = None,
    description: Optional[str] = None,
    council_models: Optional[List[str]] = None,
    chairman_model: Optional[str] = None,
    model_weights: Optional[Dict[str, float]] = None,
    model_parameters: Optional[Dict[str, Dict[str, Any]]] = None,
    ranking_criteria: Optional[List[Dict[str, Any]]] = None
) -> Optional[Dict[str, Any]]:
    """
    Update an existing preset.

    Args:
        preset_id: Preset identifier
        name: New name (optional)
        description: New description (optional)
        council_models: New council models (optional)
        chairman_model: New chairman model (optional)
        model_weights: New model weights (optional)
        model_parameters: New per-model parameters (optional)
        ranking_criteria: New ranking criteria (optional)

    Returns:
        Updated preset dict or None if not found
    """
    with get_db_context() as db:
        preset = db.query(Preset).filter(Preset.id == preset_id).first()
        if preset is None:
            return None

        if name is not None:
            preset.name = name
        if description is not None:
            preset.description = description
        if council_models is not None:
            preset.council_models = council_models
        if chairman_model is not None:
            preset.chairman_model = chairman_model
        if model_weights is not None:
            preset.model_weights = model_weights
        if model_parameters is not None:
            preset.model_parameters = model_parameters
        if ranking_criteria is not None:
            preset.ranking_criteria = ranking_criteria

        db.commit()
        db.refresh(preset)
        return preset.to_dict()


def delete_preset(preset_id: int) -> bool:
    """
    Delete a preset.

    Args:
        preset_id: Preset identifier

    Returns:
        True if deleted, False if not found
    """
    with get_db_context() as db:
        preset = db.query(Preset).filter(Preset.id == preset_id).first()
        if preset is None:
            return False

        db.delete(preset)
        db.commit()
        return True


def set_default_preset(preset_id: int) -> bool:
    """
    Set a preset as the default (active) preset.

    Args:
        preset_id: Preset identifier

    Returns:
        True if set, False if preset not found
    """
    with get_db_context() as db:
        # First, unset any existing default
        db.query(Preset).filter(Preset.is_default == 1).update({"is_default": 0})

        # Set the new default
        preset = db.query(Preset).filter(Preset.id == preset_id).first()
        if preset is None:
            return False

        preset.is_default = 1
        db.commit()
        return True


def get_default_preset() -> Optional[Dict[str, Any]]:
    """
    Get the default (active) preset.

    Returns:
        Default preset dict or None if not set
    """
    with get_db_context() as db:
        preset = db.query(Preset).filter(Preset.is_default == 1).first()
        return preset.to_dict() if preset else None


def apply_preset(preset_id: int) -> Optional[Dict[str, Any]]:
    """
    Apply a preset to the current council configuration.

    Args:
        preset_id: Preset identifier

    Returns:
        Updated council config or None if preset not found
    """
    preset = get_preset(preset_id)
    if preset is None:
        return None

    # Set as default
    set_default_preset(preset_id)

    # Update council config with preset values
    return update_council_config(
        council_models=preset['council_models'],
        chairman_model=preset['chairman_model'],
        model_weights=preset.get('model_weights'),
        ranking_criteria=preset.get('ranking_criteria')
    )
