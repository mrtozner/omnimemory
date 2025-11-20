#!/bin/bash
# Test script for OmniMemory REST API endpoints
# Usage: ./test_endpoints.sh [api_key]

set -e

BASE_URL="http://localhost:8009"
API_KEY="$1"

echo "=========================================="
echo "OmniMemory REST API - Endpoint Tests"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test 1: Health Check (no auth required)
echo -e "${BLUE}Test 1: Health Check${NC}"
echo "GET $BASE_URL/health"
response=$(curl -s "$BASE_URL/health")
if echo "$response" | grep -q "healthy"; then
    echo -e "${GREEN}✓ Health check passed${NC}"
    echo "$response" | jq '.' || echo "$response"
else
    echo -e "${RED}✗ Health check failed${NC}"
    echo "$response"
fi
echo ""

# Test 2: Create User (get API key)
if [ -z "$API_KEY" ]; then
    echo -e "${BLUE}Test 2: Create User (Get API Key)${NC}"
    echo "POST $BASE_URL/api/v1/users"
    response=$(curl -s -X POST "$BASE_URL/api/v1/users" \
        -H "Content-Type: application/json" \
        -d '{
            "email": "test@example.com",
            "name": "Test User",
            "metadata": {"platform": "test"}
        }')

    if echo "$response" | grep -q "api_key"; then
        echo -e "${GREEN}✓ User created successfully${NC}"
        API_KEY=$(echo "$response" | jq -r '.api_key')
        echo "API Key: $API_KEY"
        echo "$response" | jq '.'
    else
        echo -e "${RED}✗ User creation failed${NC}"
        echo "$response"
        exit 1
    fi
    echo ""
else
    echo -e "${BLUE}Using provided API key: ${API_KEY:0:20}...${NC}"
    echo ""
fi

# Test 3: Store Memory
echo -e "${BLUE}Test 3: Store Memory${NC}"
echo "POST $BASE_URL/api/v1/memory/store"
response=$(curl -s -X POST "$BASE_URL/api/v1/memory/store" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "content": "User prefers TypeScript for type safety",
        "metadata": {"category": "preference"},
        "compress": true
    }')

if echo "$response" | grep -q "id"; then
    echo -e "${GREEN}✓ Memory stored successfully${NC}"
    echo "$response" | jq '.' || echo "$response"
else
    echo -e "${RED}✗ Store memory failed${NC}"
    echo "$response"
fi
echo ""

# Test 4: Search Memories
echo -e "${BLUE}Test 4: Search Memories${NC}"
echo "POST $BASE_URL/api/v1/memory/search"
response=$(curl -s -X POST "$BASE_URL/api/v1/memory/search" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "query": "programming preferences",
        "limit": 5,
        "min_relevance": 0.7
    }')

if echo "$response" | grep -q "results"; then
    echo -e "${GREEN}✓ Search completed successfully${NC}"
    echo "$response" | jq '.' || echo "$response"
else
    echo -e "${RED}✗ Search failed${NC}"
    echo "$response"
fi
echo ""

# Test 5: Compress Content
echo -e "${BLUE}Test 5: Compress Content${NC}"
echo "POST $BASE_URL/api/v1/memory/compress"
response=$(curl -s -X POST "$BASE_URL/api/v1/memory/compress" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "content": "This is a very long piece of text that needs to be compressed to save tokens. It contains many details about user preferences, coding style, and other important information that should be preserved in the compressed version.",
        "target_ratio": 8.0
    }')

if echo "$response" | grep -q "compressed"; then
    echo -e "${GREEN}✓ Content compressed successfully${NC}"
    echo "$response" | jq '.' || echo "$response"
else
    echo -e "${RED}✗ Compression failed${NC}"
    echo "$response"
fi
echo ""

# Test 6: Get Statistics
echo -e "${BLUE}Test 6: Get Statistics${NC}"
echo "GET $BASE_URL/api/v1/stats"
response=$(curl -s -X GET "$BASE_URL/api/v1/stats" \
    -H "Authorization: Bearer $API_KEY")

if echo "$response" | grep -q "total_memories"; then
    echo -e "${GREEN}✓ Statistics retrieved successfully${NC}"
    echo "$response" | jq '.' || echo "$response"
else
    echo -e "${RED}✗ Get statistics failed${NC}"
    echo "$response"
fi
echo ""

# Test 7: Invalid API Key (should fail)
echo -e "${BLUE}Test 7: Invalid API Key (Expected Failure)${NC}"
echo "POST $BASE_URL/api/v1/memory/search"
response=$(curl -s -X POST "$BASE_URL/api/v1/memory/search" \
    -H "Authorization: Bearer invalid_key" \
    -H "Content-Type: application/json" \
    -d '{"query": "test"}')

if echo "$response" | grep -q "401"; then
    echo -e "${GREEN}✓ Invalid key correctly rejected${NC}"
    echo "$response" | jq '.' || echo "$response"
else
    echo -e "${RED}✗ Invalid key test failed (should have been rejected)${NC}"
    echo "$response"
fi
echo ""

echo "=========================================="
echo "All tests completed!"
echo "=========================================="
echo ""
echo "API Key for future use:"
echo "$API_KEY"
