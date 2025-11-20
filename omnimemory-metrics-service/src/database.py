"""
Database configuration and models for OmniMemory Metrics Service
Supports both SQLite (development) and PostgreSQL (production)
"""

import os
import uuid
from datetime import datetime
from typing import Optional
from contextlib import contextmanager

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    DateTime,
    ForeignKey,
    JSON,
    Text,
    Index,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.types import TypeDecorator, CHAR
import logging

logger = logging.getLogger(__name__)

# ============================================================
# Database Configuration
# ============================================================

# Detect database type from environment
DB_TYPE = os.getenv("OMNIMEMORY_DB_TYPE", "sqlite").lower()

if DB_TYPE == "postgresql":
    # PostgreSQL configuration
    DATABASE_URL = os.getenv(
        "DATABASE_URL", "postgresql://omnimemory:omnimemory@localhost:5432/omnimemory"
    )
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,  # Verify connections before using
        pool_size=10,  # Connection pool size
        max_overflow=20,  # Max overflow connections
        echo=False,  # Set to True for SQL query logging
    )
    logger.info(f"Using PostgreSQL database: {DATABASE_URL.split('@')[1]}")
else:
    # SQLite configuration (default for development)
    # Use config default if env var not set
    from pathlib import Path

    default_path = str(Path.home() / ".omnimemory" / "dashboard.db")
    db_path = os.getenv("SQLITE_DB_PATH", default_path)
    DATABASE_URL = f"sqlite:///{db_path}"
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},  # Allow multi-threaded access
        poolclass=None,  # Disable connection pooling for SQLite
        echo=False,  # Set to True for SQL query logging
    )
    logger.info(f"Using SQLite database: {db_path}")

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all models
Base = declarative_base()


# ============================================================
# Custom Types for Cross-Database Compatibility
# ============================================================


class GUID(TypeDecorator):
    """
    Platform-independent GUID type.
    Uses PostgreSQL's UUID type, otherwise uses CHAR(36), storing as stringified hex values.
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID())
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == "postgresql":
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return str(uuid.UUID(value))
            else:
                return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return uuid.UUID(value)
            return value


class JSONType(TypeDecorator):
    """
    Platform-independent JSON type.
    Uses PostgreSQL's JSONB type, otherwise uses JSON for SQLite.
    """

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB)
        else:
            return dialect.type_descriptor(JSON)


# ============================================================
# Database Models
# ============================================================


class Project(Base):
    """Represents a project workspace with per-project context isolation"""

    __tablename__ = "projects"

    # Primary Key
    project_id = Column(String(64), primary_key=True)  # Hash of workspace_path

    # Identification
    workspace_path = Column(String(512), unique=True, nullable=False, index=True)
    project_name = Column(String(256))

    # Technology Stack Detection
    language = Column(String(50))  # Primary language: python, javascript, go, etc.
    framework = Column(String(100))  # Framework: django, react, nextjs, etc.

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_accessed = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Statistics
    total_sessions = Column(Integer, default=0)

    # Settings (JSON)
    settings_json = Column(JSONType)  # Auto-save config, memory limits, etc.

    # Multi-tenancy
    tenant_id = Column(GUID, nullable=True)

    # Relationships
    sessions = relationship("ToolSession", back_populates="project")
    memories = relationship(
        "ProjectMemory", back_populates="project", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Project(project_id={self.project_id}, workspace_path={self.workspace_path})>"


class ProjectMemory(Base):
    """Stores project-specific memory with VisionDrop compression"""

    __tablename__ = "project_memories"

    # Primary Key
    memory_id = Column(String(64), primary_key=True)  # "mem_{hex}"

    # Foreign Key
    project_id = Column(
        String(64), ForeignKey("projects.project_id"), nullable=False, index=True
    )

    # Memory Data
    memory_key = Column(
        String(256), nullable=False, index=True
    )  # e.g., "architecture", "api_endpoints"
    memory_value = Column(Text)  # Full content
    compressed_value = Column(Text)  # VisionDrop compressed version

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_accessed = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Usage Tracking
    accessed_count = Column(Integer, default=0)

    # TTL (Time To Live)
    ttl_seconds = Column(Integer, nullable=True)  # NULL = never expires
    expires_at = Column(DateTime, nullable=True)  # Calculated from ttl_seconds

    # Metadata (JSON)
    metadata_json = Column(JSONType)  # Tags, source, importance, etc.

    # Multi-tenancy
    tenant_id = Column(GUID, nullable=True)

    # Relationships
    project = relationship("Project", back_populates="memories")

    def __repr__(self):
        return f"<ProjectMemory(memory_id={self.memory_id}, key={self.memory_key})>"

    # Indexes
    __table_args__ = (
        Index("idx_project_memories_project_key", "project_id", "memory_key"),
        Index("idx_project_memories_expires", "expires_at"),
    )


class ToolSession(Base):
    """Represents a tool session (e.g., Claude Code session)"""

    __tablename__ = "tool_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(GUID, unique=True, nullable=False, default=uuid.uuid4)
    tool_id = Column(String(50), nullable=False)
    tool_version = Column(String(50))
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime)
    last_activity = Column(DateTime, default=datetime.utcnow, nullable=False)
    total_compressions = Column(Integer, default=0)
    total_embeddings = Column(Integer, default=0)
    total_workflows = Column(Integer, default=0)
    tokens_saved = Column(Integer, default=0)
    process_id = Column(Integer, nullable=True)  # Process ID for deduplication

    # Project relationship (Phase 1)
    project_id = Column(
        String(64), ForeignKey("projects.project_id"), nullable=True, index=True
    )
    workspace_path = Column(String(512), nullable=True)  # Store workspace path

    # Relationships
    operations = relationship(
        "ToolOperation", back_populates="session", cascade="all, delete-orphan"
    )
    project = relationship("Project", back_populates="sessions")

    def __repr__(self):
        return f"<ToolSession(session_id={self.session_id}, tool_id={self.tool_id})>"


class ToolOperation(Base):
    """Tracks individual tool operations (read/search) with token metrics"""

    __tablename__ = "tool_operations"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    session_id = Column(
        GUID, ForeignKey("tool_sessions.session_id"), nullable=False, index=True
    )

    # Operation details
    tool_name = Column(String(50), nullable=False, index=True)  # "read" or "search"
    operation_mode = Column(
        String(50), nullable=False, index=True
    )  # "full", "overview", "symbol", "semantic", "tri_index"
    parameters = Column(JSONType)  # {compress: true, symbol: "auth", limit: 5}
    file_path = Column(String(512))  # nullable for search operations

    # Token metrics
    tokens_original = Column(Integer, nullable=False)
    tokens_actual = Column(Integer, nullable=False)
    tokens_prevented = Column(Integer, nullable=False)  # calculated

    # Performance metrics
    response_time_ms = Column(Float, nullable=False)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    tool_id = Column(String(50), nullable=False)  # "claude-code", "cursor", etc.

    # Relationships
    session = relationship("ToolSession", back_populates="operations")

    def __repr__(self):
        return f"<ToolOperation(id={self.id}, tool_name={self.tool_name}, mode={self.operation_mode})>"

    # Indexes
    __table_args__ = (
        Index("idx_session_created", "session_id", "created_at"),
        Index("idx_tool_operation", "tool_name", "operation_mode"),
    )


# ============================================================
# Database Utilities
# ============================================================


def get_db() -> Session:
    """
    Dependency for FastAPI routes.
    Yields a database session and ensures it's closed after use.

    Usage:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """
    Context manager for database sessions.

    Usage:
        with get_db_context() as db:
            db.query(ToolOperation).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database tables.
    Should be called on application startup.
    """
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully")


def get_db_info() -> dict:
    """
    Get information about the current database configuration.

    Returns:
        dict: Database configuration details
    """
    return {
        "db_type": DB_TYPE,
        "database_url": DATABASE_URL.split("@")[1]
        if "@" in DATABASE_URL
        else DATABASE_URL,
        "engine": str(engine.url),
        "pool_size": engine.pool.size() if hasattr(engine.pool, "size") else None,
    }


# ============================================================
# Migration Helpers
# ============================================================


def check_connection() -> bool:
    """
    Check if database connection is working.

    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            conn.commit()
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test database connection
    print("Testing database connection...")
    print(f"Database type: {DB_TYPE}")
    print(f"Database URL: {DATABASE_URL}")

    if check_connection():
        print("✅ Connection successful!")
        print("\nDatabase info:")
        for key, value in get_db_info().items():
            print(f"  {key}: {value}")
    else:
        print("❌ Connection failed!")
