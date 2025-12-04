# OmniMemory Universal MCP - Implementation Quick Start

This guide provides step-by-step instructions for implementing the Universal MCP Integration Architecture.

## Prerequisites

Before starting implementation:

- [x] Read `MCP_INTEGRATION_ARCHITECTURE.md` (comprehensive spec)
- [x] Understand existing `mcp_server/omnimemory_mcp.py` (457KB production server)
- [x] Review `omnimemory_universal_mcp_sample.py` (reference implementation)
- [x] Check `MCP_TOOLS_OPENAPI.yaml` (API specification)

## Implementation Roadmap

### Week 1-2: Core MCP Server Refactoring

#### Day 1-2: Extract Tool-Specific Code

**Goal**: Make existing server tool-agnostic

**Tasks**:

1. **Identify Claude-specific assumptions**
   ```bash
   cd /Users/mertozoner/Documents/GitHub/omnimemory/mcp_server
   grep -r "claude" omnimemory_mcp.py > claude_deps.txt
   ```

2. **Create tool adapter system**
   ```bash
   mkdir -p tool_adapters
   touch tool_adapters/__init__.py
   touch tool_adapters/base_adapter.py
   touch tool_adapters/claude_adapter.py
   touch tool_adapters/cursor_adapter.py
   touch tool_adapters/copilot_adapter.py
   ```

3. **Implement base adapter**
   ```python
   # tool_adapters/base_adapter.py

   class ToolAdapter:
       """Base adapter for different AI tools."""

       def __init__(self, tool_id: str):
           self.tool_id = tool_id

       def detect_workspace_path(self) -> str:
           """Detect workspace path for this tool."""
           raise NotImplementedError

       def get_instance_id(self) -> str:
           """Get stable instance ID."""
           raise NotImplementedError
   ```

4. **Test with existing server**
   ```bash
   # Backup current server
   cp omnimemory_mcp.py omnimemory_mcp_backup.py

   # Test adapter detection
   python -c "from tool_adapters.claude_adapter import ClaudeAdapter; a = ClaudeAdapter(); print(a.detect_workspace_path())"
   ```

**Deliverable**: Working tool adapter system with 3 adapters

---

#### Day 3: Standardize Tool Names

**Goal**: Rename tools to `omn_*` convention with backwards compatibility

**Tasks**:

1. **Create tool name mapping**
   ```python
   # tool_registry.py

   TOOL_NAME_MAP = {
       # New name -> Old name (for backwards compat)
       "omn_read": "read",
       "omn_search": "search",
       "omn_compress": "omn1_compress",
       # ... etc
   }
   ```

2. **Add alias decorator**
   ```python
   def tool_with_alias(new_name: str, old_name: str):
       """Register tool with both new and old names."""
       def decorator(func):
           # Register with new name
           mcp.tool(name=new_name)(func)
           # Register alias
           mcp.tool(name=old_name)(func)
           return func
       return decorator
   ```

3. **Update documentation**
   ```bash
   # Update README.md with new tool names
   # Add migration guide
   ```

**Deliverable**: All tools renamed with backwards compatibility

---

#### Day 4-6: Implement MCP Resources

**Goal**: Add 5 core resource types

**Tasks**:

1. **Create resource framework**
   ```bash
   mkdir -p resources
   touch resources/__init__.py
   touch resources/user_preferences.py
   touch resources/session_context.py
   touch resources/project_knowledge.py
   touch resources/workflow_patterns.py
   touch resources/conversation_history.py
   ```

2. **Implement user preferences resource**
   ```python
   # resources/user_preferences.py

   @mcp.resource("omnimemory://user/{user_id}/preferences")
   async def get_user_preferences(uri: str) -> Dict:
       user_id = extract_user_id_from_uri(uri)
       prefs = await db.fetchrow(
           "SELECT preferences FROM user_preferences WHERE user_id = $1",
           user_id
       )
       return {
           "uri": uri,
           "mimeType": "application/json",
           "content": json.loads(prefs["preferences"]) if prefs else {}
       }
   ```

3. **Add resource caching**
   ```python
   # Use Redis for resource caching
   RESOURCE_CACHE_TTL = 300  # 5 minutes

   async def get_resource_cached(uri: str) -> Dict:
       cached = await redis.get(f"resource:{uri}")
       if cached:
           return json.loads(cached)

       resource = await get_resource(uri)
       await redis.setex(f"resource:{uri}", RESOURCE_CACHE_TTL, json.dumps(resource))
       return resource
   ```

4. **Test resources**
   ```bash
   pytest tests/test_resources.py -v
   ```

**Deliverable**: 5 working resources with caching

---

#### Day 7-8: Implement MCP Prompts

**Goal**: Add 8 prompt templates

**Tasks**:

1. **Create prompt templates**
   ```bash
   mkdir -p prompts/templates
   touch prompts/__init__.py
   touch prompts/templates/explain_code.py
   touch prompts/templates/find_similar.py
   # ... etc
   ```

2. **Implement prompt with variable injection**
   ```python
   # prompts/templates/explain_code.py

   @mcp.prompt()
   async def explain_code(
       file_path: Optional[str] = None,
       code_snippet: Optional[str] = None,
       detail_level: str = "detailed"
   ) -> str:
       # Load project context
       project_context = await get_project_context()

       template = f"""You are analyzing code from: {file_path}

       Code:
       ```
       {code_snippet}
       ```

       Please provide a {detail_level} explanation...

       Context: {project_context}
       """

       return template
   ```

**Deliverable**: 8 working prompts

---

#### Day 9-10: Testing & Documentation

**Goal**: Comprehensive test suite and updated docs

**Tasks**:

1. **Write unit tests**
   ```bash
   pytest tests/test_tools.py --cov=mcp_server --cov-report=html
   # Target: 80%+ coverage
   ```

2. **Write integration tests**
   ```bash
   # Test with mock MCP client
   pytest tests/test_integration.py -v
   ```

3. **Update documentation**
   ```bash
   # Update README.md
   # Create migration guide from v1 to v2
   # Document all new tools/resources/prompts
   ```

**Deliverable**: Test suite with 80%+ coverage + docs

---

### Week 3: Memory Operations

#### Day 11-12: Database Schema

**Tasks**:

1. **Create migration script**
   ```sql
   -- migrations/002_add_memory_tables.sql

   CREATE TABLE memories (
       id SERIAL PRIMARY KEY,
       memory_id UUID UNIQUE NOT NULL,
       key VARCHAR(255) NOT NULL,
       content TEXT NOT NULL,
       compressed_content TEXT,
       metadata JSONB DEFAULT '{}',
       scope VARCHAR(20) DEFAULT 'private',
       user_id VARCHAR(255),
       project_id VARCHAR(255),
       created_at TIMESTAMP NOT NULL DEFAULT NOW(),
       updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
       expires_at TIMESTAMP,
       access_count INTEGER DEFAULT 0,

       INDEX idx_memory_id (memory_id),
       INDEX idx_key (key),
       INDEX idx_scope_user (scope, user_id),
       INDEX idx_project_id (project_id)
   );

   CREATE TABLE memory_tags (
       id SERIAL PRIMARY KEY,
       memory_id UUID NOT NULL,
       tag VARCHAR(100) NOT NULL,

       FOREIGN KEY (memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
       INDEX idx_tag (tag)
   );
   ```

2. **Run migration**
   ```bash
   psql -U omnimemory -d omnimemory -f migrations/002_add_memory_tables.sql
   ```

**Deliverable**: Database schema v2.0

---

#### Day 13-15: Implement Memory Tools

**Tasks**:

1. **Implement `omn_store_memory`**
   ```python
   @mcp.tool()
   async def omn_store_memory(content: str, key: str, metadata: Optional[Dict] = None, compress: bool = True) -> str:
       # Generate ID
       memory_id = str(uuid.uuid4())

       # Compress if needed
       if compress and len(content) > 1000:
           result = await compression_service.compress(content)
           compressed_content = result["compressed"]
       else:
           compressed_content = content

       # Store in database
       await db.execute("INSERT INTO memories ...", ...)

       # Generate embedding
       embedding = await embeddings_service.generate(content)

       # Store in Qdrant
       await qdrant.upsert(collection="memories", points=[...])

       # Index tags
       if metadata and metadata.get("tags"):
           for tag in metadata["tags"]:
               await db.execute("INSERT INTO memory_tags ...", ...)

       return json.dumps({"memory_id": memory_id, ...})
   ```

2. **Implement `omn_retrieve_memory`**
   ```python
   @mcp.tool()
   async def omn_retrieve_memory(query: Optional[str] = None, key: Optional[str] = None, filters: Optional[Dict] = None, limit: int = 10) -> str:
       if key:
           # Exact lookup
           result = await db.fetchrow("SELECT * FROM memories WHERE key = $1", key)
           memories = [dict(result)] if result else []

       elif query:
           # Semantic search
           query_emb = await embeddings_service.generate(query)

           # Search Qdrant
           results = await qdrant.search(
               collection="memories",
               query_vector=query_emb,
               limit=limit
           )

           # Fetch from database
           memory_ids = [r.id for r in results]
           memories = await db.fetch("SELECT * FROM memories WHERE memory_id = ANY($1)", memory_ids)

       else:
           # List all (with filters)
           memories = await db.fetch("SELECT * FROM memories LIMIT $1", limit)

       return json.dumps({"memories": memories, "total_found": len(memories)})
   ```

3. **Test memory operations**
   ```bash
   pytest tests/test_memory_operations.py -v
   ```

**Deliverable**: 8 memory tools

---

### Week 4: Session Portability

#### Day 16-17: Session Export

**Tasks**:

1. **Implement Memory Passport export**
   ```python
   # session_passport.py

   class SessionPassport:
       def __init__(self, secret_key: str):
           self.secret_key = secret_key

       async def export_session(self, session: Session, generate_qr: bool = False) -> Dict:
           # Compress context
           compressed = await compress_context(session.context)

           # Build passport
           passport = {
               "version": "2.0",
               "session_id": session.session_id,
               "exported_at": datetime.now().isoformat(),
               "context": session.context.model_dump(),
               "compressed_context": base64.b64encode(compressed.encode()).decode(),
               # ... etc
           }

           # Sign with HMAC
           passport_json = json.dumps(passport, sort_keys=True)
           signature = hmac.new(self.secret_key.encode(), passport_json.encode(), hashlib.sha256).hexdigest()
           passport["signature"] = signature

           # Generate QR code
           if generate_qr:
               import qrcode
               # ... generate QR

           return passport
   ```

2. **Test export**
   ```bash
   pytest tests/test_session_export.py -v
   ```

**Deliverable**: Session export with signature

---

#### Day 18-19: Session Import

**Tasks**:

1. **Implement passport import**
   ```python
   async def import_session(self, passport_data: Dict, new_tool_id: str) -> Session:
       # Validate signature
       if not self.validate_signature(passport_data):
           raise ValueError("Invalid signature")

       # Decompress context
       compressed = base64.b64decode(passport_data["compressed_context"]).decode()
       context = await decompress_context(compressed)

       # Create session
       session = Session(
           session_id=passport_data["session_id"],
           tool_id=new_tool_id,
           context=SessionContext(**context)
       )

       # Log migration
       await db.execute(
           "INSERT INTO session_migrations (session_id, from_tool, to_tool) VALUES ($1, $2, $3)",
           session.session_id,
           passport_data["exported_by_tool"],
           new_tool_id
       )

       return session
   ```

**Deliverable**: Session import with validation

---

#### Day 20-22: Cross-Tool Testing

**Tasks**:

1. **Test Claude → Cursor migration**
   ```python
   @pytest.mark.asyncio
   async def test_claude_to_cursor():
       # Create session in Claude
       claude_session = await create_session("claude-code", "/tmp/project")
       await track_file_access("src/auth.py", 0.9)

       # Export passport
       passport = await export_session(claude_session.session_id)

       # Import in Cursor
       cursor_session = await restore_session(passport=json.dumps(passport), tool_id="cursor")

       # Verify context preserved
       assert cursor_session.context.files_accessed[0]["path"] == "src/auth.py"
   ```

2. **Benchmark migration speed**
   ```python
   import time

   start = time.time()
   # ... migration
   duration = time.time() - start

   assert duration < 2.0  # Must complete in <2s
   ```

**Deliverable**: Cross-tool test suite

---

### Week 5: Authentication & Security

#### Day 23-24: API Key System

**Tasks**:

1. **Generate API keys**
   ```bash
   python -c "import secrets; print(f'omn_{secrets.token_urlsafe(32)}')"
   ```

2. **Implement validation**
   ```python
   async def validate_api_key(api_key: str) -> Optional[Dict]:
       if not api_key.startswith("omn_"):
           return None

       parts = api_key.split("_")
       key_id = parts[1]
       raw_key = parts[2]

       record = await db.fetchrow("SELECT * FROM api_keys WHERE key_id = $1", key_id)
       if not record:
           return None

       key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
       if key_hash != record["key_hash"]:
           return None

       return {"user_id": record["user_id"], "scopes": json.loads(record["scopes"])}
   ```

**Deliverable**: API key system

---

#### Day 25-26: Scope Enforcement

**Tasks**:

1. **Add scope checks to all tools**
   ```python
   async def check_scope(user_id: str, resource: Dict, required_scope: str) -> bool:
       if resource["scope"] == "private":
           return resource["user_id"] == user_id
       elif resource["scope"] == "shared":
           # Check team membership
           return await is_team_member(user_id, resource["project_id"])
       elif resource["scope"] == "public":
           return True
       return False
   ```

**Deliverable**: Scope enforcement

---

### Week 6: Integration Guides

#### Day 27: Claude Desktop Guide

Create `docs/integrations/CLAUDE_DESKTOP.md`

#### Day 28: Cursor Guide

Create `docs/integrations/CURSOR.md`

#### Day 29: VS Code Guide

Create `docs/integrations/VSCODE.md`

#### Day 30: API Documentation

Generate from OpenAPI spec:
```bash
npx @redocly/cli build-docs MCP_TOOLS_OPENAPI.yaml
```

---

## Quick Commands

### Development

```bash
# Start all services
./launch_omnimemory.sh

# Run tests
pytest -v --cov=mcp_server

# Lint code
ruff check mcp_server/

# Type check
mypy mcp_server/
```

### Testing with Claude

```bash
# Configure Claude
code ~/.config/claude/config.json

# Start Claude
open -a "Claude"

# Test in Claude
> /mcp list
> Use omn_semantic_search to find authentication code
```

### Monitoring

```bash
# Check service health
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health

# View metrics
open http://localhost:8004  # Dashboard
```

---

## Success Criteria

After implementation:

- [ ] 3+ AI tools integrated (Claude, Cursor, one other)
- [ ] 85%+ API cost reduction maintained
- [ ] <2s session restoration time
- [ ] >60% L1 cache hit rate
- [ ] >90% search accuracy
- [ ] 80%+ test coverage
- [ ] All security tests pass
- [ ] Documentation complete

---

## Troubleshooting

### Issue: MCP server not detected

**Solution**:
1. Check `~/.config/claude/config.json` syntax
2. Verify absolute paths
3. Restart Claude completely
4. Check logs: `tail -f ~/.config/claude/logs/mcp.log`

### Issue: Database migration fails

**Solution**:
1. Check PostgreSQL is running: `pg_isready`
2. Verify credentials in `.env`
3. Run migrations manually
4. Check for conflicting migrations

### Issue: Session migration slow (>2s)

**Solution**:
1. Check compression service latency
2. Optimize context size before export
3. Use local compression fallback
4. Add progress indicators

---

## Resources

- **Architecture**: `MCP_INTEGRATION_ARCHITECTURE.md`
- **Sample Code**: `omnimemory_universal_mcp_sample.py`
- **API Spec**: `MCP_TOOLS_OPENAPI.yaml`
- **MCP Protocol**: https://modelcontextprotocol.io/
- **Support**: GitHub Issues

---

**Last Updated**: December 4, 2025
**Version**: 1.0
