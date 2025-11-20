# How to Reproduce Error Handling Issues

This document provides specific curl commands to reproduce the error handling issues found in testing.

## Critical Issues

### 1. SQL Injection in Project ID → 500 Error

```bash
# Expected: 400/404/422 (Bad Request)
# Actual: 500 (Internal Server Error)

curl -X GET "http://localhost:8003/projects/%27%3B%20DROP%20TABLE%20projects%3B%20--/settings"
# Returns 500 instead of 400/404/422
```

### 2. Unicode Characters → 500 Error

```bash
# Expected: 200 (Should store unicode)
# Actual: 500 (Crash)

# Create a test project first
curl -X POST http://localhost:8003/sessions/start \
  -H "Content-Type: application/json" \
  -d '{"tool_id": "test", "workspace_path": "/tmp/test"}'

# Get project_id from response, then:
curl -X POST http://localhost:8003/projects/{project_id}/memories \
  -H "Content-Type: application/json" \
  -d '{"key": "unicode_test", "value": "🔥💾🚀 Unicode test 中文 العربية עברית"}'
# Returns 500 - fails to store unicode
```

### 3. Non-Existent Project Returns 200 OK

```bash
# Expected: 404 (Not Found)
# Actual: 200 (Success)

curl -X GET http://localhost:8003/projects/proj_nonexistent_12345/settings
# Returns 200 with null/empty data instead of 404
```

### 4. Non-Existent Session → 500 Error

```bash
# Expected: 404 (Not Found)
# Actual: 500 (Internal Server Error)

curl -X GET http://localhost:8003/sessions/sess_nonexistent_12345/context
# Crashes with 500 instead of returning 404
```

### 5. Null Byte Injection → 500 Error

```bash
# Expected: 400/422 (Bad Request)
# Actual: 500 (Internal Server Error)

curl -X GET "http://localhost:8003/sessions/sess_%00null/context"
# URL encoded null byte (%00) causes crash
```

### 6. Very Long Session ID → 500 Error

```bash
# Expected: 400/422 (Bad Request)
# Actual: 500 (Internal Server Error)

# Generate 1000+ character session ID
LONG_ID="sess_$(python3 -c 'print("x" * 1000)')"
curl -X GET "http://localhost:8003/sessions/${LONG_ID}/context"
# Crashes with 500
```

### 7. Whitespace-Only Project ID → 200 OK

```bash
# Expected: 400/404/422
# Actual: 200 (Success)

curl -X GET "http://localhost:8003/projects/%20%20%20/settings"
# URL encoded spaces (%20) - returns 200 instead of error
```

## Medium Severity Issues

### 8. Large Memory Value → 500 Error

```bash
# Expected: 200/413 (Success or Payload Too Large)
# Actual: 500 (Crash)

# Create project first, then:
curl -X POST http://localhost:8003/projects/{project_id}/memories \
  -H "Content-Type: application/json" \
  -d "{\"key\": \"large\", \"value\": \"$(python3 -c 'print("x" * 1000000)')\"}"
# 1MB value causes crash
```

### 9. Float Infinity Accepted

```bash
# Expected: 422 (Validation error)
# Actual: 200 (Accepted)

# Create session first, then:
curl -X POST http://localhost:8003/sessions/{session_id}/context \
  -H "Content-Type: application/json" \
  -d '{"file_path": "test.py", "file_importance": Infinity}'
# Note: This might fail as valid JSON, but Python json.dumps(float('inf')) creates it
```

### 10. NaN in JSON → 500 Error

```bash
# Expected: 400/422
# Actual: 500

curl -X POST http://localhost:8003/projects/{project_id}/settings \
  -H "Content-Type: application/json" \
  --data-raw '{"settings": {"key": NaN}}'
# NaN causes internal error
```

## Test Working Error Handling

### Good: Empty String ID → 404

```bash
# Works correctly!
curl -X GET http://localhost:8003/projects//settings
# Returns 404 as expected
```

### Good: Invalid Data Type → 422

```bash
# Works correctly!
curl -X POST http://localhost:8003/projects/{project_id}/settings \
  -H "Content-Type: application/json" \
  -d '{"settings": "not a dict"}'
# Returns 422 as expected
```

### Good: Missing Required Fields → 422

```bash
# Works correctly!
curl -X POST http://localhost:8003/projects/{project_id}/memories \
  -H "Content-Type: application/json" \
  -d '{"key": "test"}'
# Missing "value" field - returns 422 as expected
```

### Good: SQL Injection on Session → 404

```bash
# Session endpoints handle SQL injection correctly!
curl -X GET "http://localhost:8003/sessions/%27%3B%20DROP%20TABLE%20sessions%3B%20--/context"
# Returns 404, doesn't crash
```

## Automated Testing

Run the full error handling test suite:

```bash
cd omnimemory-metrics-service
python3 -m pytest tests/test_error_handling_week3.py -v
```

Run specific test categories:

```bash
# Test SQL injection handling
python3 -m pytest tests/test_error_handling_week3.py::TestSQLInjectionAttempts -v

# Test invalid identifiers
python3 -m pytest tests/test_error_handling_week3.py::TestInvalidIdentifiers -v

# Test unicode support
python3 -m pytest tests/test_error_handling_week3.py::TestEdgeCaseValues::test_unicode_and_special_characters -v

# Test non-existent resources
python3 -m pytest tests/test_error_handling_week3.py::TestNonExistentResources -v
```

## Summary of Response Codes

| Scenario | Expected | Actual | Status |
|----------|----------|--------|--------|
| SQL injection (project) | 400/404/422 | 500 | ❌ FAIL |
| Unicode characters | 200 | 500 | ❌ FAIL |
| Non-existent project | 404 | 200 | ❌ FAIL |
| Non-existent session | 404 | 500 | ❌ FAIL |
| Null byte injection | 400/422 | 500 | ❌ FAIL |
| Very long ID | 400/422 | 500 | ❌ FAIL |
| Whitespace project ID | 400/404/422 | 200 | ❌ FAIL |
| Large payload | 200/413 | 500 | ❌ FAIL |
| SQL injection (session) | 400/404/422 | 404 | ✅ PASS |
| Invalid data type | 422 | 422 | ✅ PASS |
| Missing required field | 422 | 422 | ✅ PASS |
| Wrong HTTP method | 405 | 405 | ✅ PASS |

## Monitoring Server Logs

Watch for 500 errors in real-time:

```bash
# If using uvicorn with --reload
tail -f /path/to/logs | grep "500\|ERROR\|Exception"
```

Check for unhandled exceptions:

```bash
python3 -m pytest tests/test_error_handling_week3.py -v --log-cli-level=DEBUG
```
