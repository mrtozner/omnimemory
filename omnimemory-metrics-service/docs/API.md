# OmniMemory Metrics API Documentation

Version: 1.0.0

## Overview

The OmniMemory Metrics API provides comprehensive session management, context tracking, and project memory capabilities for AI coding tools integrated with OmniMemory.

**Base URL**: `http://localhost:8003`

## Features

- **Session Management**: Create, query, and manage AI tool sessions
- **Session State Control**: Pin and archive sessions for organization
- **Context Tracking**: Track files, searches, and decisions within sessions
- **Project Memories**: Store and retrieve project-specific knowledge
- **Project Settings**: Configure project-level preferences

## Authentication

Currently, the API does not require authentication. All endpoints are accessible without credentials.

## Rate Limits

No rate limits are currently enforced.

## Error Handling

All endpoints follow consistent error response formats:

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common HTTP Status Codes

- **200 OK**: Request successful
- **404 Not Found**: Resource not found (session, project, memory)
- **500 Internal Server Error**: Server error occurred
- **503 Service Unavailable**: Metrics store not initialized

---

## API Endpoints

### Sessions

#### Query Sessions

Query sessions with optional filtering.

**Endpoint**: `GET /sessions`

**Query Parameters**:
- `project_id` (optional): Filter by project ID
- `workspace_path` (optional): Filter by workspace path
- `limit` (optional, default=10): Maximum sessions to return (1-100)
- `include_archived` (optional, default=false): Include archived sessions

**Response**:
```json
{
  "sessions": [
    {
      "session_id": "sess_abc123",
      "project_id": "proj_xyz789",
      "workspace_path": "/path/to/workspace",
      "pinned": false,
      "archived": false,
      "start_time": "2025-01-14T10:30:00"
    }
  ],
  "count": 1,
  "filters": {
    "project_id": "proj_xyz789",
    "workspace_path": null,
    "include_archived": false
  }
}
```

**cURL Example**:
```bash
# Query all sessions
curl http://localhost:8003/sessions

# Query sessions for specific project
curl http://localhost:8003/sessions?project_id=my-project

# Query with workspace path and higher limit
curl http://localhost:8003/sessions?workspace_path=/path/to/workspace&limit=20

# Include archived sessions
curl http://localhost:8003/sessions?include_archived=true
```

---

#### Pin Session

Pin a session to prevent auto-deletion.

**Endpoint**: `POST /sessions/{session_id}/pin`

**Path Parameters**:
- `session_id` (required): Session ID to pin

**Response**:
```json
{
  "session_id": "sess_abc123",
  "message": "Session pinned successfully",
  "pinned": true,
  "session": {
    "session_id": "sess_abc123",
    "pinned": true,
    "archived": false
  }
}
```

**cURL Example**:
```bash
curl -X POST http://localhost:8003/sessions/sess_abc123/pin
```

---

#### Unpin Session

Unpin a session to allow normal cleanup.

**Endpoint**: `POST /sessions/{session_id}/unpin`

**Path Parameters**:
- `session_id` (required): Session ID to unpin

**Response**:
```json
{
  "session_id": "sess_abc123",
  "message": "Session unpinned successfully",
  "pinned": false,
  "session": {
    "session_id": "sess_abc123",
    "pinned": false,
    "archived": false
  }
}
```

**cURL Example**:
```bash
curl -X POST http://localhost:8003/sessions/sess_abc123/unpin
```

---

#### Archive Session

Archive a session to hide from active lists.

**Endpoint**: `POST /sessions/{session_id}/archive`

**Path Parameters**:
- `session_id` (required): Session ID to archive

**Response**:
```json
{
  "session_id": "sess_abc123",
  "message": "Session archived successfully",
  "archived": true,
  "session": {
    "session_id": "sess_abc123",
    "pinned": false,
    "archived": true
  }
}
```

**cURL Example**:
```bash
curl -X POST http://localhost:8003/sessions/sess_abc123/archive
```

---

#### Unarchive Session

Unarchive a session to show in active lists.

**Endpoint**: `POST /sessions/{session_id}/unarchive`

**Path Parameters**:
- `session_id` (required): Session ID to unarchive

**Response**:
```json
{
  "session_id": "sess_abc123",
  "message": "Session unarchived successfully",
  "archived": false,
  "session": {
    "session_id": "sess_abc123",
    "pinned": false,
    "archived": false
  }
}
```

**cURL Example**:
```bash
curl -X POST http://localhost:8003/sessions/sess_abc123/unarchive
```

---

### Context

#### Get Session Context

Retrieve the full context for a session.

**Endpoint**: `GET /sessions/{session_id}/context`

**Path Parameters**:
- `session_id` (required): Session ID

**Response**:
```json
{
  "session_id": "sess_abc123",
  "context": {
    "files": [
      {
        "path": "src/main.py",
        "importance": 0.8,
        "access_count": 5
      }
    ],
    "searches": ["authentication implementation"],
    "decisions": ["Use JWT for auth tokens"],
    "memory_references": []
  },
  "compressed": true,
  "size_bytes": 1024
}
```

**cURL Example**:
```bash
curl http://localhost:8003/sessions/sess_abc123/context
```

---

#### Append to Session Context

Add items to session context incrementally.

**Endpoint**: `POST /sessions/{session_id}/context`

**Path Parameters**:
- `session_id` (required): Session ID

**Request Body**:
```json
{
  "file_path": "src/main.py",
  "file_importance": 0.8,
  "search_query": "authentication implementation"
}
```

**Request Fields** (all optional, but at least one should be provided):
- `file_path`: File path for file access tracking
- `file_importance`: File importance score (0.0 to 1.0)
- `search_query`: Search query to track
- `decision`: Decision text to save
- `memory_id`: Memory ID reference
- `memory_key`: Memory key reference

**Response**:
```json
{
  "session_id": "sess_abc123",
  "message": "Context updated successfully",
  "context": {
    "files": [
      {
        "path": "src/main.py",
        "importance": 0.8,
        "access_count": 1
      }
    ],
    "searches": ["authentication implementation"],
    "decisions": [],
    "memory_references": []
  }
}
```

**cURL Examples**:
```bash
# Track file access
curl -X POST http://localhost:8003/sessions/sess_abc123/context \
  -H "Content-Type: application/json" \
  -d '{"file_path": "src/main.py", "file_importance": 0.8}'

# Track search query
curl -X POST http://localhost:8003/sessions/sess_abc123/context \
  -H "Content-Type: application/json" \
  -d '{"search_query": "authentication implementation"}'

# Save decision
curl -X POST http://localhost:8003/sessions/sess_abc123/context \
  -H "Content-Type: application/json" \
  -d '{"decision": "Use JWT for authentication tokens"}'

# Track multiple items at once
curl -X POST http://localhost:8003/sessions/sess_abc123/context \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "src/auth.py",
    "file_importance": 0.9,
    "search_query": "JWT implementation",
    "decision": "Use PyJWT library"
  }'
```

---

### Memories

#### Create Project Memory

Create a project-specific memory entry.

**Endpoint**: `POST /projects/{project_id}/memories`

**Path Parameters**:
- `project_id` (required): Project ID

**Request Body**:
```json
{
  "key": "architecture",
  "value": "Microservices with event-driven architecture using FastAPI and PostgreSQL",
  "metadata": {
    "tags": ["design", "backend"],
    "author": "team-lead"
  },
  "ttl_seconds": 2592000
}
```

**Request Fields**:
- `key` (required): Memory key (e.g., 'architecture')
- `value` (required): Memory content
- `metadata` (optional): Optional metadata dictionary
- `ttl_seconds` (optional): Time to live in seconds

**Response**:
```json
{
  "memory_id": "mem_abc123",
  "project_id": "proj_xyz789",
  "key": "architecture",
  "value": "Microservices with event-driven architecture using FastAPI and PostgreSQL",
  "metadata": {
    "tags": ["design", "backend"],
    "author": "team-lead"
  },
  "created_at": "2025-01-14T10:30:00",
  "expires_at": "2025-02-13T10:30:00"
}
```

**cURL Example**:
```bash
# Create memory without TTL
curl -X POST http://localhost:8003/projects/proj_xyz789/memories \
  -H "Content-Type: application/json" \
  -d '{
    "key": "architecture",
    "value": "Microservices with event-driven architecture",
    "metadata": {"tags": ["design"]}
  }'

# Create memory with 30-day TTL
curl -X POST http://localhost:8003/projects/proj_xyz789/memories \
  -H "Content-Type: application/json" \
  -d '{
    "key": "temp_note",
    "value": "Temporary architectural decision",
    "ttl_seconds": 2592000
  }'
```

---

#### Get Project Memories

Retrieve project memories with optional filtering.

**Endpoint**: `GET /projects/{project_id}/memories`

**Path Parameters**:
- `project_id` (required): Project ID

**Query Parameters**:
- `key` (optional): Memory key to filter by (returns single memory)
- `limit` (optional, default=20): Maximum memories to return (1-100)

**Response** (when getting all memories):
```json
{
  "project_id": "proj_xyz789",
  "memories": [
    {
      "memory_id": "mem_abc123",
      "key": "architecture",
      "value": "Microservices with event-driven architecture",
      "created_at": "2025-01-14T10:30:00"
    }
  ],
  "count": 1
}
```

**Response** (when getting specific memory by key):
```json
{
  "project_id": "proj_xyz789",
  "memory": {
    "memory_id": "mem_abc123",
    "key": "architecture",
    "value": "Microservices with event-driven architecture",
    "metadata": {"tags": ["design"]},
    "created_at": "2025-01-14T10:30:00",
    "expires_at": null
  }
}
```

**cURL Examples**:
```bash
# Get all memories for project
curl http://localhost:8003/projects/proj_xyz789/memories

# Get specific memory by key
curl http://localhost:8003/projects/proj_xyz789/memories?key=architecture

# Get memories with limit
curl http://localhost:8003/projects/proj_xyz789/memories?limit=50
```

---

### Settings

#### Get Project Settings

Retrieve project-specific settings.

**Endpoint**: `GET /projects/{project_id}/settings`

**Path Parameters**:
- `project_id` (required): Project ID

**Response**:
```json
{
  "project_id": "proj_xyz789",
  "settings": {
    "auto_compress": true,
    "embeddings_enabled": true,
    "context_window_size": 5000,
    "max_context_items": 100
  },
  "updated_at": "2025-01-14T10:30:00"
}
```

**cURL Example**:
```bash
curl http://localhost:8003/projects/proj_xyz789/settings
```

---

#### Update Project Settings

Update project-specific settings.

**Endpoint**: `PUT /projects/{project_id}/settings`

**Path Parameters**:
- `project_id` (required): Project ID

**Request Body**:
```json
{
  "settings": {
    "auto_compress": true,
    "embeddings_enabled": true,
    "context_window_size": 5000,
    "max_context_items": 100
  }
}
```

**Note**: This operation **merges** settings with existing values. Only provided fields are updated.

**Response**:
```json
{
  "project_id": "proj_xyz789",
  "settings": {
    "auto_compress": true,
    "embeddings_enabled": true,
    "context_window_size": 5000,
    "max_context_items": 100
  },
  "message": "Settings updated successfully",
  "updated_at": "2025-01-14T10:30:00"
}
```

**cURL Examples**:
```bash
# Update single setting
curl -X PUT http://localhost:8003/projects/proj_xyz789/settings \
  -H "Content-Type: application/json" \
  -d '{
    "settings": {
      "auto_compress": false
    }
  }'

# Update multiple settings
curl -X PUT http://localhost:8003/projects/proj_xyz789/settings \
  -H "Content-Type: application/json" \
  -d '{
    "settings": {
      "auto_compress": true,
      "embeddings_enabled": true,
      "context_window_size": 8000
    }
  }'
```

---

## Interactive Documentation

The API provides interactive documentation through:

- **Swagger UI**: http://localhost:8003/docs
- **ReDoc**: http://localhost:8003/redoc
- **OpenAPI Spec**: http://localhost:8003/openapi.json

---

## Example Workflows

### Workflow 1: Session with Context Tracking

```bash
# 1. Start a session (use existing session ID from your tool)
SESSION_ID="sess_abc123"

# 2. Track file access
curl -X POST http://localhost:8003/sessions/$SESSION_ID/context \
  -H "Content-Type: application/json" \
  -d '{"file_path": "src/main.py", "file_importance": 0.9}'

# 3. Track search
curl -X POST http://localhost:8003/sessions/$SESSION_ID/context \
  -H "Content-Type: application/json" \
  -d '{"search_query": "implement authentication"}'

# 4. Save decision
curl -X POST http://localhost:8003/sessions/$SESSION_ID/context \
  -H "Content-Type: application/json" \
  -d '{"decision": "Use JWT with refresh tokens"}'

# 5. Get full context
curl http://localhost:8003/sessions/$SESSION_ID/context

# 6. Pin session for later reference
curl -X POST http://localhost:8003/sessions/$SESSION_ID/pin
```

### Workflow 2: Project Memory Management

```bash
# 1. Create project memory for architecture decision
curl -X POST http://localhost:8003/projects/my-project/memories \
  -H "Content-Type: application/json" \
  -d '{
    "key": "auth_architecture",
    "value": "JWT-based authentication with Redis for token storage",
    "metadata": {"decision_date": "2025-01-14"}
  }'

# 2. Create another memory for API patterns
curl -X POST http://localhost:8003/projects/my-project/memories \
  -H "Content-Type: application/json" \
  -d '{
    "key": "api_patterns",
    "value": "RESTful endpoints with Pydantic validation"
  }'

# 3. Get all project memories
curl http://localhost:8003/projects/my-project/memories

# 4. Get specific memory
curl http://localhost:8003/projects/my-project/memories?key=auth_architecture
```

### Workflow 3: Session Organization

```bash
# 1. Query active sessions
curl http://localhost:8003/sessions?limit=10

# 2. Archive old sessions
curl -X POST http://localhost:8003/sessions/old_session_1/archive
curl -X POST http://localhost:8003/sessions/old_session_2/archive

# 3. Pin important sessions
curl -X POST http://localhost:8003/sessions/important_session/pin

# 4. Query only active (non-archived) sessions
curl http://localhost:8003/sessions?include_archived=false

# 5. Query including archived sessions
curl http://localhost:8003/sessions?include_archived=true
```

---

## Tips and Best Practices

### Context Tracking
- Set `file_importance` based on how central the file is to the current work (0.0 = reference only, 1.0 = core file)
- Track search queries to understand what information was sought
- Save decisions to create a decision log for the session

### Project Memories
- Use descriptive keys (e.g., `auth_architecture`, `api_design_v2`)
- Include metadata for better organization (`tags`, `author`, `decision_date`)
- Set `ttl_seconds` for temporary memories (e.g., experiment results, temporary notes)

### Session Management
- Pin important sessions for debugging or reference
- Archive completed sessions to keep active list clean
- Use `workspace_path` filtering to query sessions for specific projects

### Settings
- Configure `auto_compress` based on context size preferences
- Adjust `context_window_size` based on your AI tool's requirements
- Use `max_context_items` to limit context tracking overhead

---

## Support

For issues or questions:
- Check interactive documentation at http://localhost:8003/docs
- Review this API documentation
- Check service logs for error details
