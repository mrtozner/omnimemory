# PostgreSQL Fact Store Setup Guide

This guide covers setting up and using the PostgreSQL-based fact store for OmniMemory's structural facts.

## Overview

The PostgreSQL fact store provides persistent storage for structural facts extracted from code files (imports, classes, functions, etc.) as part of the TriIndex architecture.

## Prerequisites

- Docker (with PostgreSQL container) OR local PostgreSQL installation
- Python 3.8+
- Required Python packages: `asyncpg`, `psycopg2-binary`

## Quick Start

### 1. Install Dependencies

```bash
cd omnimemory-storage
pip install asyncpg psycopg2-binary
```

### 2. Run Setup Script

The setup script automatically detects your PostgreSQL installation (Docker or local) and sets up the database:

```bash
./setup_postgres.sh
```

This will:
- Detect PostgreSQL (Docker container or local)
- Create the `omnimemory` database if needed
- Apply the schema from `schemas/fact_store_schema.sql`
- Create all tables, views, indexes, and triggers
- Verify the setup with basic operations

### 3. Run Tests

Verify everything works correctly:

```bash
python3 test_postgres_setup.py
```

Expected output: **23/23 tests passed** ✅

## Configuration

The scripts use environment variables with sensible defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_HOST` | `localhost` | Database host |
| `POSTGRES_PORT` | `5432` | Database port |
| `POSTGRES_USER` | `omnimemory` | Database user |
| `POSTGRES_PASSWORD` | `omnimemory_dev_pass` | Database password |
| `POSTGRES_DB` | `omnimemory` | Database name |

### Custom Configuration Example

```bash
export POSTGRES_PASSWORD=my_secure_password
./setup_postgres.sh
```

## Database Schema

The fact store includes:

### Tables

1. **facts** - Core structural facts
   - Stores predicates (imports, defines_class, defines_function)
   - Includes file path, hash, confidence, line numbers
   - Unique constraint on (file_path, predicate, object)

2. **file_facts** - Junction table for file-fact relationships
   - Links files to their facts
   - Enables efficient file-based queries

3. **fact_domains** - Domain classifications
   - Categorizes facts by domain (e.g., "authentication", "database")
   - Supports domain-specific searches

4. **fact_access_log** - Access tracking
   - Records fact usage patterns
   - Used for intelligent caching and relevance scoring

### Views

1. **fact_statistics** - Aggregated statistics per predicate type
2. **file_fact_summary** - Summary of facts per file
3. **hot_facts** - Most frequently accessed facts (7-day window)

### Indexes

- 23 indexes for optimal query performance
- Full-text search support on object names
- Composite indexes for common query patterns

## Using the FactStore Class

### Basic Usage

```python
import asyncio
from src.fact_store import FactStore

async def main():
    # Initialize
    store = FactStore(
        host="localhost",
        port=5432,
        database="omnimemory",
        user="omnimemory",
        password="omnimemory_dev_pass"
    )

    await store.connect()

    # Store facts
    facts = [
        {"predicate": "imports", "object": "asyncio", "line_number": 1},
        {"predicate": "defines_class", "object": "MyClass", "line_number": 10},
        {"predicate": "defines_function", "object": "my_function", "line_number": 20}
    ]

    count = await store.store_facts(
        file_path="/path/to/file.py",
        facts=facts,
        file_hash="abc123..."
    )
    print(f"Stored {count} facts")

    # Retrieve facts
    facts = await store.get_facts(file_path="/path/to/file.py")
    for fact in facts:
        print(f"{fact.predicate}: {fact.object} (line {fact.line_number})")

    # Search facts
    results = await store.search_facts(
        predicate="defines_class",
        object_pattern="My%"
    )
    print(f"Found {len(results)} classes")

    # Get statistics
    stats = await store.get_statistics()
    print(f"Total facts: {stats['total_facts']}")

    await store.close()

asyncio.run(main())
```

### Connection String Usage

```python
store = FactStore(
    connection_string="postgresql://omnimemory:password@localhost:5432/omnimemory"
)
```

### Update Facts When File Changes

```python
# Automatically removes old facts and stores new ones
new_facts = [...]
await store.update_facts(
    file_path="/path/to/file.py",
    new_facts=new_facts,
    new_file_hash="new_hash_456"
)
```

### Log Fact Access for Caching

```python
await store.log_access(
    fact_id="uuid-here",
    tool_id="semantic_search",
    query_context="authentication methods",
    relevance_score=0.95
)
```

## Performance

Based on test results:

- **Insert**: ~2,700 facts/second
- **Retrieve**: ~93,000 facts/second
- **Search**: Sub-millisecond for typical queries
- **Indexes**: 23 indexes ensure optimal query performance

## Maintenance

### Drop and Recreate Database

```bash
docker exec -i omnimemory-postgres psql -U omnimemory -d postgres -c "DROP DATABASE omnimemory;"
./setup_postgres.sh
```

### View Database Statistics

```bash
docker exec -it omnimemory-postgres psql -U omnimemory -d omnimemory
```

Then run:
```sql
SELECT * FROM fact_statistics;
SELECT * FROM file_fact_summary;
SELECT * FROM hot_facts LIMIT 10;
```

### Backup Database

```bash
docker exec omnimemory-postgres pg_dump -U omnimemory omnimemory > omnimemory_backup.sql
```

### Restore Database

```bash
cat omnimemory_backup.sql | docker exec -i omnimemory-postgres psql -U omnimemory -d omnimemory
```

## Troubleshooting

### Connection Issues

**Problem**: Cannot connect to PostgreSQL

**Solutions**:
1. Check if container is running: `docker ps | grep postgres`
2. Verify credentials match container environment
3. Try connecting directly: `docker exec -it omnimemory-postgres psql -U omnimemory -d omnimemory`

### Schema Errors

**Problem**: Schema application fails with "already exists" errors

**Solution**: Drop and recreate the database:
```bash
docker exec -i omnimemory-postgres psql -U omnimemory -d postgres -c "DROP DATABASE omnimemory;"
./setup_postgres.sh
```

### Missing Tables/Views

**Problem**: Some tables or views are missing

**Solution**: Re-run the schema:
```bash
docker exec -i omnimemory-postgres psql -U omnimemory -d omnimemory < schemas/fact_store_schema.sql
```

## Docker Container Setup

If you don't have the PostgreSQL container yet:

```bash
docker run -d \
  --name omnimemory-postgres \
  -e POSTGRES_USER=omnimemory \
  -e POSTGRES_PASSWORD=omnimemory_dev_pass \
  -e POSTGRES_DB=omnimemory \
  -p 5432:5432 \
  postgres:16-alpine
```

## Files

- `setup_postgres.sh` - Setup script for database initialization
- `test_postgres_setup.py` - Comprehensive test suite (23 tests)
- `schemas/fact_store_schema.sql` - Database schema definition
- `src/fact_store.py` - Python interface to PostgreSQL
- `POSTGRES_SETUP.md` - This file

## Next Steps

1. ✅ Database is set up and tested
2. 🔄 Integrate with code indexing pipeline
3. 🔄 Add fact extraction from Python/TypeScript files
4. 🔄 Connect to semantic search for hybrid queries
5. 🔄 Implement caching based on access patterns

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review test output: `python3 test_postgres_setup.py`
3. Inspect logs: `docker logs omnimemory-postgres`
