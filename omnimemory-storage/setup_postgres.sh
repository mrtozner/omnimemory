#!/bin/bash
#
# PostgreSQL Fact Store Setup Script
#
# Sets up the PostgreSQL database for OmniMemory's fact store.
# Handles both Docker container and local PostgreSQL installations.
#
# Usage:
#   ./setup_postgres.sh
#
# Environment Variables (optional):
#   POSTGRES_HOST      - Database host (default: localhost)
#   POSTGRES_PORT      - Database port (default: 5432)
#   POSTGRES_USER      - Database user (default: omnimemory)
#   POSTGRES_PASSWORD  - Database password (default: omnimemory_dev_pass)
#   POSTGRES_DB        - Database name (default: omnimemory)
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration with environment variable defaults
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-omnimemory}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-omnimemory_dev_pass}"
POSTGRES_DB="${POSTGRES_DB:-omnimemory}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHEMA_FILE="$SCRIPT_DIR/schemas/fact_store_schema.sql"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}PostgreSQL Fact Store Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if schema file exists
if [ ! -f "$SCHEMA_FILE" ]; then
    echo -e "${RED}✗ Schema file not found: $SCHEMA_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Schema file found${NC}"

# Check if PostgreSQL is accessible
echo ""
echo -e "${BLUE}Checking PostgreSQL connection...${NC}"
export PGPASSWORD="$POSTGRES_PASSWORD"

# Check if Docker container is running
DOCKER_CONTAINER=$(docker ps --filter "name=omnimemory-postgres" --format "{{.Names}}" 2>/dev/null || echo "")

if [ -n "$DOCKER_CONTAINER" ]; then
    echo -e "${GREEN}✓ Found PostgreSQL Docker container: $DOCKER_CONTAINER${NC}"
    CONNECTION_METHOD="docker"
    PSQL_CMD="docker exec -i $DOCKER_CONTAINER psql -U $POSTGRES_USER"
else
    echo -e "${YELLOW}⚠ Docker container not found, checking local PostgreSQL...${NC}"
    CONNECTION_METHOD="direct"

    # Check if psql is installed
    if ! command -v psql &> /dev/null; then
        echo -e "${RED}✗ psql command not found${NC}"
        echo "Please install PostgreSQL client tools or use Docker container"
        exit 1
    fi

    PSQL_CMD="psql -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER"
fi

# Test connection
if [ "$CONNECTION_METHOD" = "docker" ]; then
    if ! docker exec -i "$DOCKER_CONTAINER" psql -U "$POSTGRES_USER" -d postgres -c '\q' 2>/dev/null; then
        echo -e "${RED}✗ Cannot connect to PostgreSQL in Docker container${NC}"
        exit 1
    fi
else
    if ! psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d postgres -c '\q' 2>/dev/null; then
        echo -e "${RED}✗ Cannot connect to PostgreSQL at $POSTGRES_HOST:$POSTGRES_PORT${NC}"
        echo ""
        echo "Please ensure PostgreSQL is running and credentials are correct."
        echo ""
        echo "Current settings:"
        echo "  Host: $POSTGRES_HOST"
        echo "  Port: $POSTGRES_PORT"
        echo "  User: $POSTGRES_USER"
        echo "  Database: $POSTGRES_DB"
        echo ""
        echo "To start PostgreSQL with Docker:"
        echo "  docker run -d --name omnimemory-postgres \\"
        echo "    -e POSTGRES_USER=$POSTGRES_USER \\"
        echo "    -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \\"
        echo "    -e POSTGRES_DB=$POSTGRES_DB \\"
        echo "    -p $POSTGRES_PORT:5432 \\"
        echo "    postgres:16-alpine"
        exit 1
    fi
fi

echo -e "${GREEN}✓ PostgreSQL connection successful${NC}"

# Check if database exists
echo ""
echo -e "${BLUE}Checking database...${NC}"
DB_EXISTS=$($PSQL_CMD -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$POSTGRES_DB'" 2>/dev/null || echo "")

if [ -z "$DB_EXISTS" ]; then
    echo -e "${YELLOW}⚠ Database '$POSTGRES_DB' does not exist. Creating...${NC}"
    $PSQL_CMD -d postgres -c "CREATE DATABASE $POSTGRES_DB;"
    echo -e "${GREEN}✓ Database '$POSTGRES_DB' created${NC}"
else
    echo -e "${GREEN}✓ Database '$POSTGRES_DB' already exists${NC}"
fi

# Apply schema
echo ""
echo -e "${BLUE}Applying schema...${NC}"
if [ "$CONNECTION_METHOD" = "docker" ]; then
    if docker exec -i "$DOCKER_CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" < "$SCHEMA_FILE" 2>&1 | grep -i error; then
        echo -e "${RED}✗ Schema application failed${NC}"
        exit 1
    fi
else
    if $PSQL_CMD -d "$POSTGRES_DB" -f "$SCHEMA_FILE" 2>&1 | grep -i error; then
        echo -e "${RED}✗ Schema application failed${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✓ Schema applied successfully${NC}"

# Verify tables
echo ""
echo -e "${BLUE}Verifying tables...${NC}"
TABLES=$($PSQL_CMD -d "$POSTGRES_DB" -tAc "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename;")

EXPECTED_TABLES=("fact_access_log" "fact_domains" "facts" "file_facts")
MISSING_TABLES=()

for table in "${EXPECTED_TABLES[@]}"; do
    if echo "$TABLES" | grep -q "^$table$"; then
        echo -e "${GREEN}  ✓ $table${NC}"
    else
        echo -e "${RED}  ✗ $table${NC}"
        MISSING_TABLES+=("$table")
    fi
done

if [ ${#MISSING_TABLES[@]} -gt 0 ]; then
    echo -e "${RED}✗ Missing tables: ${MISSING_TABLES[*]}${NC}"
    exit 1
fi

# Verify views
echo ""
echo -e "${BLUE}Verifying views...${NC}"
VIEWS=$($PSQL_CMD -d "$POSTGRES_DB" -tAc "SELECT viewname FROM pg_views WHERE schemaname='public' ORDER BY viewname;")

EXPECTED_VIEWS=("fact_statistics" "file_fact_summary" "hot_facts")
MISSING_VIEWS=()

for view in "${EXPECTED_VIEWS[@]}"; do
    if echo "$VIEWS" | grep -q "^$view$"; then
        echo -e "${GREEN}  ✓ $view${NC}"
    else
        echo -e "${RED}  ✗ $view${NC}"
        MISSING_VIEWS+=("$view")
    fi
done

if [ ${#MISSING_VIEWS[@]} -gt 0 ]; then
    echo -e "${RED}✗ Missing views: ${MISSING_VIEWS[*]}${NC}"
    exit 1
fi

# Verify indexes
echo ""
echo -e "${BLUE}Verifying indexes...${NC}"
INDEX_COUNT=$($PSQL_CMD -d "$POSTGRES_DB" -tAc "SELECT COUNT(*) FROM pg_indexes WHERE schemaname='public';")
echo -e "${GREEN}  ✓ $INDEX_COUNT indexes created${NC}"

# Verify extensions
echo ""
echo -e "${BLUE}Verifying extensions...${NC}"
EXTENSIONS=$($PSQL_CMD -d "$POSTGRES_DB" -tAc "SELECT extname FROM pg_extension;")

if echo "$EXTENSIONS" | grep -q "uuid-ossp"; then
    echo -e "${GREEN}  ✓ uuid-ossp extension${NC}"
else
    echo -e "${RED}  ✗ uuid-ossp extension${NC}"
    exit 1
fi

# Test basic operations
echo ""
echo -e "${BLUE}Testing basic operations...${NC}"

# Test insert
$PSQL_CMD -d "$POSTGRES_DB" -c "
    INSERT INTO facts (predicate, object, file_path, file_hash, confidence, line_number)
    VALUES ('test_predicate', 'test_object', '/tmp/test.py', 'test_hash', 1.0, 1)
    ON CONFLICT (file_path, predicate, object) DO NOTHING;
" > /dev/null 2>&1
echo -e "${GREEN}  ✓ Insert operation${NC}"

# Test select
$PSQL_CMD -d "$POSTGRES_DB" -c "
    SELECT * FROM facts WHERE predicate = 'test_predicate' LIMIT 1;
" > /dev/null 2>&1
echo -e "${GREEN}  ✓ Select operation${NC}"

# Test delete (cleanup)
$PSQL_CMD -d "$POSTGRES_DB" -c "
    DELETE FROM facts WHERE predicate = 'test_predicate';
" > /dev/null 2>&1
echo -e "${GREEN}  ✓ Delete operation${NC}"

# Get statistics
echo ""
echo -e "${BLUE}Database statistics:${NC}"
$PSQL_CMD -d "$POSTGRES_DB" -c "
    SELECT
        (SELECT COUNT(*) FROM facts) as total_facts,
        (SELECT COUNT(DISTINCT file_path) FROM facts) as total_files,
        (SELECT COUNT(*) FROM fact_domains) as total_domains,
        (SELECT COUNT(*) FROM fact_access_log) as total_accesses;
"

# Connection info
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Connection details:"
echo "  Host:     $POSTGRES_HOST"
echo "  Port:     $POSTGRES_PORT"
echo "  User:     $POSTGRES_USER"
echo "  Database: $POSTGRES_DB"
echo ""
echo "Connection string:"
echo "  postgresql://$POSTGRES_USER:****@$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB"
echo ""
echo "Next steps:"
echo "  1. Run the test suite: python test_postgres_setup.py"
echo "  2. Check the Python interface: python -c 'from src.fact_store import FactStore'"
if [ "$CONNECTION_METHOD" = "docker" ]; then
    echo "  3. Review the schema: docker exec -it $DOCKER_CONTAINER psql -U $POSTGRES_USER -d $POSTGRES_DB"
else
    echo "  3. Review the schema: psql -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER -d $POSTGRES_DB"
fi
echo ""
