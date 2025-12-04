"""
OmniMemory Universal MCP Server - Sample Implementation
Version 2.0 - Multi-Tool Compatible

This is a sample/reference implementation showing the key patterns
for the universal MCP server architecture.

DO NOT use directly in production - this is a design reference.
The full implementation should be built incrementally following
the MCP_INTEGRATION_ARCHITECTURE.md implementation plan.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from mcp.server import FastMCP
from mcp import types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service URLs (from config)
EMBEDDINGS_URL = "http://localhost:8000"
COMPRESSION_URL = "http://localhost:8001"
PROCEDURAL_URL = "http://localhost:8002"
METRICS_URL = "http://localhost:8003"
QDRANT_URL = "http://localhost:6333"


class ToolAdapter:
    """Base adapter for different AI tools."""

    def __init__(self, tool_id: str):
        self.tool_id = tool_id

    def detect_workspace_path(self) -> str:
        """Detect workspace path for this tool."""
        raise NotImplementedError

    def get_instance_id(self) -> str:
        """Get stable instance ID for this tool."""
        raise NotImplementedError


class ClaudeAdapter(ToolAdapter):
    """Adapter for Claude Code."""

    def __init__(self):
        super().__init__("claude-code")

    def detect_workspace_path(self) -> str:
        import os

        return os.getcwd()

    def get_instance_id(self) -> str:
        import socket

        return f"claude_{socket.gethostname()}"


class CursorAdapter(ToolAdapter):
    """Adapter for Cursor."""

    def __init__(self):
        super().__init__("cursor")

    def detect_workspace_path(self) -> str:
        import os

        # Cursor sets CURSOR_WORKSPACE environment variable
        return os.environ.get("CURSOR_WORKSPACE", os.getcwd())

    def get_instance_id(self) -> str:
        import os

        # Cursor provides instance ID
        return os.environ.get("CURSOR_INSTANCE_ID", str(uuid.uuid4()))


class CopilotAdapter(ToolAdapter):
    """Adapter for GitHub Copilot."""

    def __init__(self):
        super().__init__("copilot")

    def detect_workspace_path(self) -> str:
        import os

        return os.environ.get("COPILOT_WORKSPACE", os.getcwd())

    def get_instance_id(self) -> str:
        return str(uuid.uuid4())


class MemoryPassport:
    """Memory Passport for cross-tool session portability."""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    async def export_session(self, session: Dict, generate_qr: bool = False) -> Dict:
        """Export session as portable Memory Passport."""

        # Build passport data
        passport_data = {
            "version": "2.0",
            "session_id": session["session_id"],
            "exported_at": datetime.now().isoformat(),
            "exported_by_tool": session["tool_id"],
            "project_id": session.get("project_id"),
            "workspace_path": session.get("workspace_path"),
            "context": session.get("context", {}),
            "metadata": {
                "total_files_accessed": len(
                    session.get("context", {}).get("files_accessed", [])
                ),
                "total_searches": len(
                    session.get("context", {}).get("recent_searches", [])
                ),
            },
        }

        # Generate signature
        passport_json = json.dumps(passport_data, sort_keys=True)
        signature = hmac.new(
            self.secret_key.encode(), passport_json.encode(), hashlib.sha256
        ).hexdigest()

        passport_data["signature"] = signature

        # Generate QR code (optional)
        if generate_qr:
            try:
                import qrcode
                import base64
                from io import BytesIO

                qr = qrcode.QRCode(version=1, box_size=10, border=5)
                qr.add_data(json.dumps(passport_data))
                qr.make(fit=True)

                img = qr.make_image(fill_color="black", back_color="white")
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                qr_base64 = base64.b64encode(buffer.getvalue()).decode()

                passport_data["qr_code"] = f"data:image/png;base64,{qr_base64}"
            except ImportError:
                logger.warning("QR code generation requires qrcode package")

        return passport_data

    def validate_signature(self, passport_data: Dict) -> bool:
        """Validate passport HMAC signature."""

        signature = passport_data.get("signature")
        if not signature:
            return False

        # Remove signature and optional fields for validation
        passport_copy = passport_data.copy()
        passport_copy.pop("signature", None)
        passport_copy.pop("qr_code", None)

        passport_json = json.dumps(passport_copy, sort_keys=True)
        expected_signature = hmac.new(
            self.secret_key.encode(), passport_json.encode(), hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(signature, expected_signature)


class UniversalMCPServer:
    """Universal MCP Server for OmniMemory."""

    def __init__(self):
        self.mcp = FastMCP("omnimemory")
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Auto-detect tool and configure adapter
        self.adapter = self._detect_tool_adapter()
        logger.info(f"Detected tool: {self.adapter.tool_id}")

        # Initialize passport system
        self.passport = MemoryPassport(
            secret_key="change-in-production"  # TODO: Load from env
        )

        # Register MCP handlers
        self._register_tools()
        self._register_resources()
        self._register_prompts()

    def _detect_tool_adapter(self) -> ToolAdapter:
        """Auto-detect which AI tool is running this server."""
        import os

        # Check environment variables
        if os.environ.get("CLAUDE_SESSION"):
            return ClaudeAdapter()
        elif os.environ.get("CURSOR_WORKSPACE"):
            return CursorAdapter()
        elif os.environ.get("COPILOT_WORKSPACE"):
            return CopilotAdapter()
        else:
            # Default to Claude
            logger.warning("Could not detect tool, defaulting to Claude")
            return ClaudeAdapter()

    def _register_tools(self):
        """Register MCP tools."""

        # =====================================================================
        # CATEGORY 1: Memory Operations
        # =====================================================================

        @self.mcp.tool()
        async def omn_store_memory(
            content: str,
            key: str,
            metadata: Optional[Dict] = None,
            compress: bool = True,
        ) -> str:
            """
            Store a memory with automatic compression and metadata indexing.

            Args:
                content: Memory content to store
                key: Unique retrieval key
                metadata: Optional metadata (tags, importance, scope, expiry)
                compress: Whether to auto-compress large content (default: True)

            Returns:
                JSON with memory_id and storage details
            """

            memory_id = str(uuid.uuid4())

            # Count tokens
            token_count = len(content.split())  # Simplified

            # Auto-compress if needed
            compressed_content = content
            compression_ratio = 1.0

            if compress and token_count > 1000:
                try:
                    result = await self.http_client.post(
                        f"{COMPRESSION_URL}/compress",
                        json={
                            "context": content,
                            "model_id": "gpt-4",
                            "quality_threshold": 0.95,
                        },
                    )
                    compressed_content = result.json()["compressed_text"]
                    compression_ratio = len(compressed_content) / len(content)
                except Exception as e:
                    logger.error(f"Compression failed: {e}")

            # TODO: Store in database
            # await db.execute(...)

            # TODO: Index for semantic search
            # await qdrant.upsert(...)

            return json.dumps(
                {
                    "memory_id": memory_id,
                    "key": key,
                    "size_bytes": len(compressed_content),
                    "compressed": compress and token_count > 1000,
                    "compression_ratio": compression_ratio,
                    "indexed": True,
                }
            )

        @self.mcp.tool()
        async def omn_retrieve_memory(
            query: Optional[str] = None,
            key: Optional[str] = None,
            filters: Optional[Dict] = None,
            limit: int = 10,
        ) -> str:
            """
            Retrieve stored memories by key or semantic search.

            Args:
                query: Semantic search query (optional)
                key: Exact key match (optional)
                filters: Optional filters (tags, scope, min_importance, max_age_days)
                limit: Maximum results to return (default: 10)

            Returns:
                JSON with array of memories and metadata
            """

            # TODO: Implement actual retrieval
            # For now, return mock data

            if key:
                # Exact key lookup
                memories = [
                    {
                        "memory_id": str(uuid.uuid4()),
                        "key": key,
                        "content": "Mock memory content",
                        "metadata": {},
                        "created_at": datetime.now().isoformat(),
                    }
                ]
            elif query:
                # Semantic search
                # TODO: Call embeddings service and Qdrant
                memories = [
                    {
                        "memory_id": str(uuid.uuid4()),
                        "key": "auto_generated",
                        "content": "Mock search result",
                        "relevance_score": 0.85,
                        "metadata": {},
                        "created_at": datetime.now().isoformat(),
                    }
                ]
            else:
                memories = []

            return json.dumps(
                {
                    "memories": memories,
                    "total_found": len(memories),
                    "query_time_ms": 50,
                }
            )

        # =====================================================================
        # CATEGORY 2: Search Operations
        # =====================================================================

        @self.mcp.tool()
        async def omn_semantic_search(
            query: str,
            scope: Optional[str] = None,
            limit: int = 10,
            min_relevance: float = 0.7,
            filters: Optional[Dict] = None,
        ) -> str:
            """
            Search codebase semantically using vector embeddings.

            Args:
                query: Natural language search query
                scope: Directory to search (default: project root)
                limit: Maximum results (default: 10)
                min_relevance: Minimum similarity score 0-1 (default: 0.7)
                filters: Optional filters (file_types, exclude_paths, recency_weight)

            Returns:
                JSON with search results and metadata
            """

            # Check response cache
            cache_key = hashlib.sha256(f"{query}:{scope}:{limit}".encode()).hexdigest()

            # TODO: Check cache
            # cached = await response_cache.get(cache_key)
            # if cached: return cached

            # Generate query embedding
            try:
                embed_result = await self.http_client.post(
                    f"{EMBEDDINGS_URL}/embed", json={"text": query}
                )
                query_embedding = embed_result.json()["embedding"]
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                return json.dumps(
                    {"error": "Embedding generation failed", "results": []}
                )

            # Search Qdrant (TODO: Implement tri-index search)
            # results = await self._tri_index_search(query_embedding, limit)

            # Mock results for now
            results = [
                {
                    "file_path": "src/auth.py",
                    "relevance_score": 0.92,
                    "excerpt": "JWT authentication implementation...",
                    "line_range": [10, 50],
                    "last_modified": datetime.now().isoformat(),
                },
                {
                    "file_path": "src/middleware.py",
                    "relevance_score": 0.85,
                    "excerpt": "Authentication middleware...",
                    "line_range": [5, 30],
                    "last_modified": datetime.now().isoformat(),
                },
            ]

            # Filter by relevance
            filtered = [r for r in results if r["relevance_score"] >= min_relevance]

            response = json.dumps(
                {
                    "results": filtered[:limit],
                    "search_time_ms": 125,
                    "cache_hit": False,
                    "baseline_tokens": 50 * 1200,  # 50 files
                    "actual_tokens": len(filtered) * 300,  # Filtered results
                    "prevention_percentage": (
                        (1 - (len(filtered) * 300) / (50 * 1200)) * 100
                    ),
                }
            )

            # TODO: Cache response
            # await response_cache.set(cache_key, response)

            # Track metrics
            await self._track_api_prevention(
                operation="semantic_search",
                baseline_tokens=50 * 1200,
                actual_tokens=len(filtered) * 300,
            )

            return response

        # =====================================================================
        # CATEGORY 3: Session Operations
        # =====================================================================

        @self.mcp.tool()
        async def omn_create_session(
            tool_id: Optional[str] = None,
            workspace_path: Optional[str] = None,
            restore_from: Optional[str] = None,
        ) -> str:
            """
            Create new session for a tool/workspace.

            Args:
                tool_id: Tool identifier (auto-detected if not provided)
                workspace_path: Workspace path (auto-detected if not provided)
                restore_from: Optional session_id to restore from

            Returns:
                JSON with session details and Memory Passport info
            """

            # Use adapter defaults if not provided
            tool_id = tool_id or self.adapter.tool_id
            workspace_path = workspace_path or self.adapter.detect_workspace_path()

            session_id = f"sess_{uuid.uuid4().hex[:12]}"
            project_id = hashlib.sha256(workspace_path.encode()).hexdigest()[:16]

            session = {
                "session_id": session_id,
                "tool_id": tool_id,
                "workspace_path": workspace_path,
                "project_id": project_id,
                "created_at": datetime.now().isoformat(),
                "context": {
                    "files_accessed": [],
                    "recent_searches": [],
                    "decisions": [],
                    "workflow_state": {},
                },
            }

            # TODO: Save to database
            # await db.execute(...)

            # Generate export URL for Memory Passport
            export_url = f"omnimemory://export/{session_id}"

            return json.dumps(
                {
                    "session_id": session_id,
                    "project_id": project_id,
                    "created_at": session["created_at"],
                    "memory_passport": {
                        "export_url": export_url,
                        "note": "Use omn_export_session to generate portable passport",
                    },
                }
            )

        @self.mcp.tool()
        async def omn_restore_session(
            session_id: Optional[str] = None,
            passport: Optional[str] = None,
            tool_id: Optional[str] = None,
        ) -> str:
            """
            Restore previous session by ID or memory passport.

            Args:
                session_id: Session ID to restore (optional)
                passport: Memory Passport JSON (optional)
                tool_id: Tool requesting restoration (auto-detected if not provided)

            Returns:
                JSON with restored session context
            """

            tool_id = tool_id or self.adapter.tool_id

            if passport:
                # Import from Memory Passport
                passport_data = json.loads(passport)

                # Validate signature
                if not self.passport.validate_signature(passport_data):
                    return json.dumps({"error": "Invalid passport signature"})

                session_id = passport_data["session_id"]
                context = passport_data.get("context", {})

                # Log migration
                logger.info(
                    f"Session migration: {passport_data['exported_by_tool']} → {tool_id}"
                )

                # TODO: Save migration to database
                # await db.execute(...)

                return json.dumps(
                    {
                        "session_id": session_id,
                        "context": context,
                        "restored_at": datetime.now().isoformat(),
                        "original_tool": passport_data["exported_by_tool"],
                        "migration": True,
                    }
                )

            elif session_id:
                # Restore by ID
                # TODO: Load from database
                # session = await db.fetchrow(...)

                # Mock session for now
                return json.dumps(
                    {
                        "session_id": session_id,
                        "context": {
                            "files_accessed": [],
                            "recent_searches": [],
                            "decisions": [],
                        },
                        "restored_at": datetime.now().isoformat(),
                        "migration": False,
                    }
                )

            else:
                return json.dumps({"error": "Must provide session_id or passport"})

        @self.mcp.tool()
        async def omn_export_session(
            session_id: Optional[str] = None, generate_qr: bool = False
        ) -> str:
            """
            Export session as Memory Passport (portable JSON).

            Args:
                session_id: Session to export (current if not provided)
                generate_qr: Whether to generate QR code for mobile transfer

            Returns:
                Memory Passport JSON with signature
            """

            # TODO: Load session from database
            # session = await db.fetchrow(...)

            # Mock session for now
            session = {
                "session_id": session_id or f"sess_{uuid.uuid4().hex[:12]}",
                "tool_id": self.adapter.tool_id,
                "workspace_path": self.adapter.detect_workspace_path(),
                "project_id": "proj_xyz789",
                "context": {
                    "files_accessed": [
                        {"path": "src/auth.py", "importance": 0.95},
                        {"path": "src/db.py", "importance": 0.80},
                    ],
                    "recent_searches": ["authentication flow", "database schema"],
                    "decisions": ["Use JWT for auth"],
                },
            }

            passport_data = await self.passport.export_session(session, generate_qr)

            return json.dumps(passport_data, indent=2)

    def _register_resources(self):
        """Register MCP resources."""

        @self.mcp.resource("omnimemory://user/{user_id}/preferences")
        async def get_user_preferences(uri: str) -> Dict:
            """Get user preferences."""

            # Extract user_id from URI
            user_id = uri.split("/")[-2]

            # TODO: Load from database
            # prefs = await db.fetchrow(...)

            # Mock preferences
            preferences = {
                "theme": "dark",
                "default_compression": True,
                "search_defaults": {"min_relevance": 0.7, "max_results": 10},
                "auto_features": {
                    "workflow_learning": True,
                    "context_compression": True,
                },
            }

            return {
                "uri": uri,
                "mimeType": "application/json",
                "content": preferences,
            }

        @self.mcp.resource("omnimemory://session/current")
        async def get_current_session(uri: str) -> Dict:
            """Get current session context."""

            # TODO: Get current session
            # session = await session_manager.get_current_session()

            # Mock session
            session_data = {
                "session_id": f"sess_{uuid.uuid4().hex[:12]}",
                "tool_id": self.adapter.tool_id,
                "workspace_path": self.adapter.detect_workspace_path(),
                "files_accessed": [],
                "recent_searches": [],
                "decisions": [],
            }

            return {
                "uri": uri,
                "mimeType": "application/json",
                "content": session_data,
            }

        @self.mcp.resource("omnimemory://project/{project_id}")
        async def get_project_knowledge(uri: str) -> Dict:
            """Get project-specific knowledge."""

            # Extract project_id from URI
            project_id = uri.split("/")[-1]

            # TODO: Load from database and knowledge graph
            # project = await db.fetchrow(...)
            # kg_stats = await knowledge_graph.get_stats(project_id)

            # Mock project data
            project_data = {
                "project_id": project_id,
                "workspace_path": self.adapter.detect_workspace_path(),
                "file_index": {
                    "total_files": 1234,
                    "indexed_files": 1200,
                    "last_index": datetime.now().isoformat(),
                },
                "knowledge_graph": {"nodes": 5000, "edges": 12000},
                "common_patterns": [
                    "JWT authentication",
                    "PostgreSQL queries",
                    "React components",
                ],
            }

            return {
                "uri": uri,
                "mimeType": "application/json",
                "content": project_data,
            }

    def _register_prompts(self):
        """Register MCP prompts."""

        @self.mcp.prompt()
        async def explain_code(
            file_path: Optional[str] = None,
            code_snippet: Optional[str] = None,
            detail_level: str = "detailed",
        ) -> str:
            """
            Generate code explanation prompt.

            Args:
                file_path: Path to code file (optional)
                code_snippet: Code snippet to explain (optional)
                detail_level: brief, detailed, or expert
            """

            # TODO: Load project context
            # project_context = await self._get_project_context()

            template = f"""You are analyzing code from the file: {file_path or "provided snippet"}

Code:
```
{code_snippet or "// Code will be loaded from file"}
```

Please provide a {detail_level} explanation covering:
1. Purpose and functionality
2. Key algorithms or patterns used
3. Dependencies and integrations
4. Potential issues or improvements

Context from project:
- Project type: Python web application
- Common patterns: JWT auth, PostgreSQL, FastAPI
"""

            return template

        @self.mcp.prompt()
        async def find_similar(
            reference_code: str, search_scope: str = "project"
        ) -> str:
            """Generate prompt to find similar code."""

            template = f"""Find code similar to the following reference:

Reference Code:
```
{reference_code}
```

Search scope: {search_scope}

Look for:
1. Similar algorithms or patterns
2. Related functionality
3. Alternative implementations
4. Code that could be refactored to use this pattern

Return results ranked by similarity with explanations.
"""

            return template

    async def _track_api_prevention(
        self, operation: str, baseline_tokens: int, actual_tokens: int
    ):
        """Track API prevention metrics."""

        prevented_tokens = baseline_tokens - actual_tokens
        prevented_pct = (prevented_tokens / baseline_tokens) * 100

        # Calculate cost (assuming $0.015/1K tokens)
        baseline_cost = (baseline_tokens / 1000) * 0.015
        actual_cost = (actual_tokens / 1000) * 0.015
        cost_saved = baseline_cost - actual_cost

        logger.info(
            f"API Prevention: {operation} - "
            f"Prevented {prevented_tokens} tokens ({prevented_pct:.1f}%) - "
            f"Saved ${cost_saved:.4f}"
        )

        # TODO: Send to metrics service
        # await self.http_client.post(
        #     f"{METRICS_URL}/track",
        #     json={...}
        # )

    async def run(self):
        """Run the MCP server."""
        await self.mcp.run()


# ============================================================================
# Main Entry Point
# ============================================================================


async def main():
    """Initialize and run the universal MCP server."""

    server = UniversalMCPServer()

    logger.info("=" * 60)
    logger.info("OmniMemory Universal MCP Server v2.0")
    logger.info(f"Tool: {server.adapter.tool_id}")
    logger.info(f"Workspace: {server.adapter.detect_workspace_path()}")
    logger.info("=" * 60)

    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
