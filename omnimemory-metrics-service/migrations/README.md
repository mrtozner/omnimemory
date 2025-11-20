# Database Migrations

This directory contains database migration scripts for the OmniMemory Metrics Service.

## Available Migrations

### 001_create_response_cache.py
Creates the response cache table for caching API responses.

### 002_add_process_id.py
Adds `process_id` column to `tool_sessions` table for session deduplication.

**Changes:**
- Adds `process_id INTEGER` column to `tool_sessions` table
- Creates index `idx_tool_sessions_pid` on `(process_id, ended_at)` for fast deduplication queries
- Existing sessions will have `process_id = NULL`

**Run:**
```bash
python3 migrations/002_add_process_id.py
```

**Rollback:**
```bash
python3 migrations/002_add_process_id.py --rollback
```

## Running Migrations

Migrations are standalone Python scripts that can be run directly:

```bash
cd omnimemory-metrics-service
python3 migrations/<migration_file>.py
```

All migrations are idempotent - safe to run multiple times. If the migration has already been applied, it will detect this and skip gracefully.

## Database Location

Default database path: `~/.omnimemory/dashboard.db`

## Verification

After running a migration, verify it succeeded:

```bash
# Check column exists
sqlite3 ~/.omnimemory/dashboard.db "PRAGMA table_info(tool_sessions);" | grep process_id

# Check index exists
sqlite3 ~/.omnimemory/dashboard.db "PRAGMA index_list(tool_sessions);" | grep pid

# Check data
sqlite3 ~/.omnimemory/dashboard.db "SELECT session_id, process_id FROM tool_sessions LIMIT 5;"
```

## Migration Log

| Migration | Date | Status | Description |
|-----------|------|--------|-------------|
| 001 | - | ✅ Applied | Response cache table |
| 002 | 2025-11-14 | ✅ Applied | Process ID for session deduplication |
