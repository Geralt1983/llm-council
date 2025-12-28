"""Database session management for LLM Council."""

import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

from .models import Base, Settings

# Database path - configurable via environment variable
# Railway provides a persistent volume at /data in production
DB_PATH = Path(os.environ.get("DATABASE_PATH", "data/llmcouncil.db"))
DATA_DIR = DB_PATH.parent

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Create SQLite engine
DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for SQLite
    echo=False  # Set to True for SQL debugging
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize the database and create all tables."""
    Base.metadata.create_all(bind=engine)

    # Initialize default settings if not exist
    with get_db_context() as db:
        existing = db.query(Settings).filter(Settings.key == "council_config").first()
        if not existing:
            default_config = Settings(
                key="council_config",
                value=Settings.get_default_council_config()
            )
            db.add(default_config)
            db.commit()


def get_db() -> Session:
    """Get a database session. Use with dependency injection."""
    db = SessionLocal()
    try:
        return db
    finally:
        pass  # Caller is responsible for closing


@contextmanager
def get_db_context():
    """Context manager for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def migrate_from_json():
    """Migrate existing JSON conversations to SQLite."""
    import json
    from datetime import datetime

    json_dir = DATA_DIR / "conversations"
    if not json_dir.exists():
        return

    with get_db_context() as db:
        for json_file in json_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Check if already migrated
                existing = db.query(
                    __import__('backend.db.models', fromlist=['Conversation']).Conversation
                ).filter_by(id=data["id"]).first()

                if existing:
                    continue

                # Import here to avoid circular imports
                from .models import Conversation, Message

                # Create conversation
                conv = Conversation(
                    id=data["id"],
                    title=data.get("title", "New Conversation"),
                    created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"]
                )
                db.add(conv)

                # Create messages
                for msg_data in data.get("messages", []):
                    if msg_data["role"] == "user":
                        msg = Message(
                            conversation_id=conv.id,
                            role="user",
                            content=msg_data["content"]
                        )
                    else:
                        msg = Message(
                            conversation_id=conv.id,
                            role="assistant",
                            stage1=msg_data.get("stage1"),
                            stage2=msg_data.get("stage2"),
                            stage3=msg_data.get("stage3")
                        )
                    db.add(msg)

                db.commit()
                print(f"Migrated: {json_file.name}")

            except Exception as e:
                print(f"Error migrating {json_file.name}: {e}")
                db.rollback()
