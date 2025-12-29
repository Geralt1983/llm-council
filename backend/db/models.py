"""SQLAlchemy models for LLM Council."""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, Text, DateTime, Integer, Float, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Conversation(Base):
    """Conversation model storing chat sessions."""

    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True)
    title = Column(String(255), default="New Conversation")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at"
    )

    def to_dict(self):
        """Convert to dictionary for API response."""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "messages": [msg.to_dict() for msg in self.messages]
        }

    def to_metadata(self):
        """Convert to metadata dictionary for list view."""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "message_count": len(self.messages)
        }


class Message(Base):
    """Message model storing individual messages in conversations."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=True)  # For user messages
    stage1 = Column(JSON, nullable=True)  # Stage 1 responses
    stage2 = Column(JSON, nullable=True)  # Stage 2 rankings
    stage3 = Column(JSON, nullable=True)  # Stage 3 synthesis
    metadata_json = Column(JSON, nullable=True)  # Additional metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    metrics = relationship(
        "ModelMetric",
        back_populates="message",
        cascade="all, delete-orphan"
    )

    def to_dict(self):
        """Convert to dictionary for API response."""
        if self.role == "user":
            return {
                "role": "user",
                "content": self.content
            }
        else:
            result = {
                "role": "assistant",
                "stage1": self.stage1,
                "stage2": self.stage2,
                "stage3": self.stage3
            }
            # Include metadata if present (label_to_model, aggregate_rankings, dissent, etc.)
            if self.metadata_json:
                result["metadata"] = self.metadata_json
            return result


class ModelMetric(Base):
    """Model metrics for analytics tracking."""

    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=False)
    model_id = Column(String(100), nullable=False)
    response_time_ms = Column(Integer, nullable=True)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)
    ranking_position = Column(Float, nullable=True)
    cost_usd = Column(Float, nullable=True)
    error_type = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    message = relationship("Message", back_populates="metrics")


class Settings(Base):
    """Application settings stored in database."""

    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(JSON, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @classmethod
    def get_default_council_config(cls):
        """Return default council configuration."""
        return {
            "council_models": [
                "openai/gpt-4o",
                "anthropic/claude-3.5-sonnet",
                "google/gemini-2.5-flash",
                "x-ai/grok-3",
            ],
            "chairman_model": "google/gemini-2.5-flash",
            "theme": "light",
            # Phase 4: Advanced Deliberation
            "ranking_criteria": [
                {"id": "accuracy", "name": "Accuracy", "description": "Factual correctness and precision", "weight": 1.0, "enabled": True},
                {"id": "completeness", "name": "Completeness", "description": "Thoroughness and coverage of the topic", "weight": 1.0, "enabled": True},
                {"id": "clarity", "name": "Clarity", "description": "Clear and easy to understand", "weight": 1.0, "enabled": True},
            ],
            "model_weights": {},  # model_id -> weight (1.0 default)
            "enable_confidence": False,  # Ask models for confidence scores
            "enable_dissent_tracking": True,  # Track disagreements
        }
