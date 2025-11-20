# Database Setup Guide

This guide explains how to set up and use the database for OmniMemory Metrics Service with support for both SQLite (development) and PostgreSQL (production).

## Overview

The metrics service uses SQLAlchemy 2.0+ with Alembic for database migrations. It supports:

- **SQLite** (development): Lightweight, file-based database
- **PostgreSQL** (production): Scalable, production-ready database

## Quick Start

### Development (SQLite)

```bash
# Install dependencies
uv pip install -r requirements.txt

# Run migrations
alembic upgrade head

# The database file will be created at: ./omnimemory_metrics.db
```

### Production (PostgreSQL)

```bash
# Set environment variables
export OMNIMEMORY_DB_TYPE=postgresql
export DATABASE_URL="postgresql://user:password@localhost:5432/omnimemory"

# Run migrations
alembic upgrade head
```

## Database Schema

### tool_sessions

Tracks individual tool sessions (e.g., Claude Code sessions).

| Column            | Type         | Description                          |
|-------------------|--------------|--------------------------------------|
| id                | Integer      | Auto-incrementing primary key        |
| session_id        | UUID/CHAR(36)| Unique session identifier            |
| tool_id           | String(50)   | Tool identifier (e.g., "claude-code")|
| tool_version      | String(50)   | Tool version                         |
| started_at        | DateTime     | Session start timestamp              |
| ended_at          | DateTime     | Session end timestamp (nullable)     |
| last_activity     | DateTime     | Last activity timestamp              |
| total_compressions| Integer      | Total compression operations         |
| total_embeddings  | Integer      | Total embedding operations           |
| total_workflows   | Integer      | Total workflow operations            |
| tokens_saved      | Integer      | Total tokens saved                   |

### tool_operations

Tracks individual read/search operations with token metrics.

| Column            | Type         | Description                          |
|-------------------|--------------|--------------------------------------|
| id                | UUID/CHAR(36)| Primary key                          |
| session_id        | UUID/CHAR(36)| Foreign key to tool_sessions         |
| tool_name         | String(50)   | Operation type ("read" or "search")  |
| operation_mode    | String(50)   | Mode: "full", "overview", "symbol", "semantic", "tri_index" |
| parameters        | JSON/JSONB   | Operation parameters (e.g., {compress: true, limit: 5}) |
| file_path         | String(512)  | File path (nullable for search)      |
| tokens_original   | Integer      | Original token count (without optimization) |
| tokens_actual     | Integer      | Actual tokens sent to API            |
| tokens_prevented  | Integer      | Tokens prevented from API (savings)  |
| response_time_ms  | Float        | Operation response time in milliseconds |
| created_at        | DateTime     | Operation timestamp                  |
| tool_id           | String(50)   | Tool identifier                      |

**Indexes:**
- `idx_session_created`: (session_id, created_at) - For session queries
- `idx_tool_operation`: (tool_name, operation_mode) - For breakdown queries
- `idx_tool_operations_session_id`: (session_id) - For filtering by session
- `idx_tool_operations_tool_name`: (tool_name) - For filtering by tool
- `idx_tool_operations_operation_mode`: (operation_mode) - For filtering by mode
- `idx_tool_operations_created_at`: (created_at) - For time-based queries

## Configuration

### Environment Variables

| Variable              | Default                          | Description                     |
|-----------------------|----------------------------------|---------------------------------|
| OMNIMEMORY_DB_TYPE    | sqlite                           | Database type: "sqlite" or "postgresql" |
| DATABASE_URL          | (varies)                         | Full database connection string |
| SQLITE_DB_PATH        | ./omnimemory_metrics.db          | SQLite database file path       |

### SQLite (Development)

```bash
# Default configuration (no env vars needed)
python -m src.database
```

Database file: `./omnimemory_metrics.db`

### PostgreSQL (Production)

```bash
# Set environment variables
export OMNIMEMORY_DB_TYPE=postgresql
export DATABASE_URL="postgresql://omnimemory:omnimemory@localhost:5432/omnimemory"

# Test connection
python -m src.database
```

## Migrations

### Create a New Migration

```bash
alembic revision -m "description_of_changes"
```

### Apply Migrations

```bash
# Upgrade to latest
alembic upgrade head

# Upgrade one version
alembic upgrade +1

# Downgrade one version
alembic downgrade -1

# Downgrade to base
alembic downgrade base
```

### View Migration History

```bash
# Show current version
alembic current

# Show migration history
alembic history
```

## Migration from SQLite to PostgreSQL

When moving from development to production, use the provided migration script:

```bash
./scripts/migrate_sqlite_to_postgres.sh [sqlite_db_path] [postgres_connection_string]
```

Example:

```bash
./scripts/migrate_sqlite_to_postgres.sh \
    ./omnimemory_metrics.db \
    "postgresql://omnimemory:omnimemory@localhost:5432/omnimemory"
```

The script will:
1. Check database connections
2. Migrate all data from SQLite to PostgreSQL
3. Verify data integrity
4. Report success/failure

**Note:** The script is idempotent - it can be run multiple times safely.

## Usage in Code

### Using the Database Session

```python
from src.database import get_db_context, ToolSession, ToolOperation
from datetime import datetime
import uuid

# Create a session
with get_db_context() as db:
    session = ToolSession(
        session_id=uuid.uuid4(),
        tool_id="claude-code",
        tool_version="1.0.0",
    )
    db.add(session)
    db.commit()

    # Create an operation
    operation = ToolOperation(
        id=uuid.uuid4(),
        session_id=session.session_id,
        tool_name="read",
        operation_mode="overview",
        parameters={"compress": True},
        file_path="/path/to/file.py",
        tokens_original=5000,
        tokens_actual=500,
        tokens_prevented=4500,
        response_time_ms=120.5,
        tool_id="claude-code",
    )
    db.add(operation)
    db.commit()
```

### Using with FastAPI

```python
from fastapi import Depends
from src.database import get_db, ToolOperation
from sqlalchemy.orm import Session

@app.get("/operations")
def get_operations(db: Session = Depends(get_db)):
    operations = db.query(ToolOperation).all()
    return operations
```

## Testing

Run the test suite:

```bash
# Test database connection and models
python test_database.py
```

Expected output:
```
Testing database operations...

1. Initializing database...
   ✅ Database initialized

2. Database configuration:
   db_type: sqlite
   database_url: sqlite:///./omnimemory_metrics.db

3. Creating test session...
   ✅ Created session: <uuid>

4. Creating test operations...
   ✅ Created 3 operations

5. Querying operations...
   Found 3 operations:
   - read/full: 0 tokens prevented
   - read/overview: 4500 tokens prevented
   - search/semantic: 45000 tokens prevented

6. Total tokens prevented: 49500
   Cost saved: $0.7425

7. Testing session -> operations relationship...
   Session has 3 operations

✅ All tests passed!
```

## Troubleshooting

### Connection Issues

**SQLite:**
```bash
# Check if database file exists
ls -lh omnimemory_metrics.db

# Inspect schema
sqlite3 omnimemory_metrics.db ".schema"
```

**PostgreSQL:**
```bash
# Test connection
psql postgresql://user:pass@localhost/omnimemory -c "SELECT 1"

# Check database exists
psql -U postgres -c "\l" | grep omnimemory
```

### Migration Issues

**Error: "table already exists"**
```bash
# Reset database (WARNING: destroys all data)
rm omnimemory_metrics.db
alembic upgrade head
```

**Error: "can't locate revision"**
```bash
# Check alembic version table
sqlite3 omnimemory_metrics.db "SELECT * FROM alembic_version"

# Stamp current version
alembic stamp head
```

## Database Maintenance

### Backup (SQLite)

```bash
# Simple copy
cp omnimemory_metrics.db omnimemory_metrics.db.backup

# With sqlite3
sqlite3 omnimemory_metrics.db ".backup omnimemory_metrics.db.backup"
```

### Backup (PostgreSQL)

```bash
# Full database dump
pg_dump -U omnimemory omnimemory > backup.sql

# Restore
psql -U omnimemory omnimemory < backup.sql
```

### Cleanup Old Data

```python
from src.database import get_db_context, ToolOperation
from datetime import datetime, timedelta

# Delete operations older than 30 days
with get_db_context() as db:
    cutoff = datetime.utcnow() - timedelta(days=30)
    deleted = db.query(ToolOperation).filter(
        ToolOperation.created_at < cutoff
    ).delete()
    db.commit()
    print(f"Deleted {deleted} old operations")
```

## Performance Optimization

### PostgreSQL

```sql
-- Analyze tables for query optimization
ANALYZE tool_sessions;
ANALYZE tool_operations;

-- Create additional indexes if needed
CREATE INDEX idx_custom ON tool_operations(column_name);

-- Monitor query performance
EXPLAIN ANALYZE SELECT * FROM tool_operations WHERE session_id = 'uuid';
```

### SQLite

```sql
-- Analyze database
ANALYZE;

-- Vacuum to reclaim space
VACUUM;

-- Enable WAL mode for better concurrency
PRAGMA journal_mode=WAL;
```

## Cross-Database Compatibility

The schema uses custom types to ensure compatibility:

- **GUID**: PostgreSQL UUID or SQLite CHAR(36)
- **JSONType**: PostgreSQL JSONB or SQLite JSON

This allows the same models to work on both databases without code changes.

## References

- [SQLAlchemy 2.0 Documentation](https://docs.sqlalchemy.org/en/20/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [SQLite Documentation](https://www.sqlite.org/docs.html)
