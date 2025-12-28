"""Storage layer for LLM Council - SQLite implementation."""

from typing import List, Dict, Any, Optional
from datetime import datetime

from .db.session import get_db_context
from .db.models import Conversation, Message, Settings


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
    theme: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update the council configuration.

    Args:
        council_models: List of model identifiers for the council
        chairman_model: Model identifier for the chairman
        theme: UI theme ('light' or 'dark')

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

    set_settings("council_config", config)
    return config
