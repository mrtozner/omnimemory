-- OmniMemory: Memories Table Schema
-- Version: 1.0.0
-- Database: PostgreSQL 12+ or SQLite with JSON support

-- Drop existing table if reimplementing
-- DROP TABLE IF EXISTS memories CASCADE;

CREATE TABLE IF NOT EXISTS memories (
    -- Identity
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),  -- UUID equivalent for SQLite
    scope TEXT NOT NULL CHECK (scope IN ('shared', 'private')),

    -- Authentication (foreign key to api_keys table)
    api_key_id TEXT NOT NULL,

    -- Content
    content TEXT NOT NULL,                   -- Compressed content
    original_content TEXT,                   -- Original uncompressed (optional)

    -- Compression metrics
    compressed INTEGER DEFAULT 0,            -- Boolean (0/1)
    compression_ratio REAL,                  -- 0.0-1.0
    original_tokens INTEGER,
    compressed_tokens INTEGER,
    tokens_saved INTEGER,
    cost_saved_usd REAL,                     -- Decimal

    -- Indexing
    indexed INTEGER DEFAULT 0,               -- Boolean (0/1)
    index_time_ms INTEGER,

    -- Metadata
    user_id TEXT,
    agent_id TEXT,
    tags TEXT,                               -- JSON array as TEXT (SQLite)
    metadata TEXT DEFAULT '{}',              -- JSON as TEXT (SQLite)

    -- Session
    session_id TEXT,
    tool_id TEXT,

    -- Versioning
    version INTEGER DEFAULT 1,

    -- Timestamps
    created_at TEXT DEFAULT (datetime('now')),     -- ISO 8601
    updated_at TEXT DEFAULT (datetime('now')),     -- ISO 8601
    expires_at TEXT,                               -- ISO 8601 (optional)

    -- Access tracking
    accessed_count INTEGER DEFAULT 0,
    last_accessed TEXT,                      -- ISO 8601
    accessed_by TEXT DEFAULT '[]'            -- JSON array as TEXT (SQLite)
);

-- Indexes for performance (SQLite compatible)
CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope);
CREATE INDEX IF NOT EXISTS idx_memories_api_key_id ON memories(api_key_id);
CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_agent_id ON memories(agent_id);
CREATE INDEX IF NOT EXISTS idx_memories_session_id ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_tool_id ON memories(tool_id);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_expires_at ON memories(expires_at) WHERE expires_at IS NOT NULL;

-- Foreign key constraint (if using with api_keys table)
-- Note: SQLite requires PRAGMA foreign_keys = ON;
-- CREATE INDEX IF NOT EXISTS idx_memories_fk_api_key ON memories(api_key_id);

-- Full-text search virtual table (SQLite FTS5)
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory_id UNINDEXED,
    content,
    tags,
    tokenize = 'porter'
);

-- Trigger to keep FTS table in sync
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(memory_id, content, tags)
    VALUES (new.id, new.content, COALESCE(new.tags, '[]'));
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    UPDATE memories_fts
    SET content = new.content, tags = COALESCE(new.tags, '[]')
    WHERE memory_id = new.id;
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    DELETE FROM memories_fts WHERE memory_id = old.id;
END;

-- Trigger to auto-update updated_at timestamp
CREATE TRIGGER IF NOT EXISTS memories_updated_at AFTER UPDATE ON memories
FOR EACH ROW
BEGIN
    UPDATE memories
    SET updated_at = datetime('now')
    WHERE id = NEW.id;
END;

-- View for easy querying with computed fields
CREATE VIEW IF NOT EXISTS memories_with_stats AS
SELECT
    m.*,
    (m.accessed_count > 0) as has_been_accessed,
    (m.expires_at IS NOT NULL AND datetime(m.expires_at) < datetime('now')) as is_expired,
    CAST((julianday(datetime('now')) - julianday(m.created_at)) * 24 AS INTEGER) as age_hours
FROM memories m;

-- Migration helper: Check if table needs to be created
SELECT CASE
    WHEN EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories')
    THEN 'Table memories already exists'
    ELSE 'Table memories created successfully'
END as status;
