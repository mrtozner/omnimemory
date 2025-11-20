"""
Unit tests for ContextInjector class.

Tests dual-mode context formatting:
- IDE Tools: Formatted context (MCP/LSP)
- Agents: Raw JSON context
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from context_injector import ContextInjector
from tool_registry import ToolInfo, ToolRegistry


class TestContextInjectorBasics:
    """Test basic ContextInjector functionality."""

    def test_init(self):
        """Test ContextInjector initialization."""
        registry = ToolRegistry()
        injector = ContextInjector(registry)

        assert injector.registry == registry
        assert injector.merger is None

    def test_init_with_merger(self):
        """Test ContextInjector initialization with context merger."""
        registry = ToolRegistry()
        merger = Mock()
        injector = ContextInjector(registry, merger)

        assert injector.registry == registry
        assert injector.merger == merger


class TestIDEToolFormatting:
    """Test formatting for IDE tools (Cursor, VSCode, Claude Code, Continue)."""

    @pytest.mark.asyncio
    async def test_format_for_cursor(self):
        """Test MCP format for Cursor IDE."""
        registry = ToolRegistry()
        injector = ContextInjector(registry)

        # Register Cursor tool
        tool = await registry.register_tool("cursor-123", "cursor")

        # Sample context
        session_context = {
            "files_accessed": [
                {"path": "src/main.py", "importance": 0.9},
                {"path": "src/utils.py", "importance": 0.7},
                {"path": "tests/test_main.py", "importance": 0.5},
            ],
            "recent_searches": [
                {"query": "authentication", "timestamp": "2025-11-15T10:00:00Z"},
                {"query": "user model", "timestamp": "2025-11-15T09:45:00Z"},
            ],
            "decisions": [
                {"decision": "Use JWT for authentication"},
                {"decision": "PostgreSQL for database"},
            ],
        }

        # Format context
        result = await injector.prepare_context_for_tool(
            "cursor-123", "project-123", session_context
        )

        # Verify structure
        assert "system_prompt_addition" in result
        assert "context_metadata" in result

        # Verify prompt contains file info
        prompt = result["system_prompt_addition"]
        assert "src/main.py" in prompt
        assert "authentication" in prompt
        assert "Use JWT for authentication" in prompt

        # Verify metadata
        metadata = result["context_metadata"]
        assert len(metadata["files_accessed"]) <= 10
        assert len(metadata["recent_searches"]) <= 5
        assert len(metadata["decisions"]) <= 3

    @pytest.mark.asyncio
    async def test_format_for_claude_code(self):
        """Test Claude Code gets same format as Cursor."""
        registry = ToolRegistry()
        injector = ContextInjector(registry)

        # Register Claude Code tool
        tool = await registry.register_tool("claude-code-456", "claude-code")

        session_context = {
            "files_accessed": [{"path": "src/app.py", "importance": 0.8}],
            "recent_searches": [{"query": "login"}],
            "decisions": [{"decision": "Use FastAPI"}],
        }

        # Format context
        result = await injector.prepare_context_for_tool(
            "claude-code-456", "project-123", session_context
        )

        # Should have same structure as Cursor
        assert "system_prompt_addition" in result
        assert "context_metadata" in result
        assert "src/app.py" in result["system_prompt_addition"]

    @pytest.mark.asyncio
    async def test_format_for_vscode(self):
        """Test VSCode includes LSP symbols."""
        registry = ToolRegistry()
        injector = ContextInjector(registry)

        # Register VSCode tool
        tool = await registry.register_tool("vscode-789", "vscode")

        session_context = {
            "files_accessed": [
                {"path": "src/models/user.py", "importance": 0.9},
                {"path": "src/api/routes.py", "importance": 0.7},
            ],
            "recent_searches": [{"query": "user validation"}],
            "decisions": [],
        }

        # Format context
        result = await injector.prepare_context_for_tool(
            "vscode-789", "project-123", session_context
        )

        # Verify VSCode-specific structure
        assert "mcp_context" in result
        assert "lsp_context" in result
        assert "ui_hints" in result

        # MCP context
        assert "system_prompt_addition" in result["mcp_context"]
        assert "src/models/user.py" in result["mcp_context"]["system_prompt_addition"]

        # LSP context
        lsp = result["lsp_context"]
        assert "symbols_to_watch" in lsp
        assert len(lsp["symbols_to_watch"]) > 0
        assert lsp["symbols_to_watch"][0]["file"] == "src/models/user.py"
        assert lsp["symbols_to_watch"][0]["type"] == "file"
        assert "importance" in lsp["symbols_to_watch"][0]

        # UI hints
        ui = result["ui_hints"]
        assert ui["show_context_panel"] is True
        assert "src/models/user.py" in ui["highlight_files"]
        assert ui["sidebar_state"] == "context"

    @pytest.mark.asyncio
    async def test_format_for_continue(self):
        """Test Continue.dev gets simplified format."""
        registry = ToolRegistry()
        injector = ContextInjector(registry)

        # Register Continue tool
        tool = await registry.register_tool("continue-101", "continue")

        session_context = {
            "files_accessed": [
                {"path": "src/feature.ts", "importance": 0.8},
                {"path": "src/types.ts", "importance": 0.6},
            ],
            "recent_searches": [{"query": "typescript types"}],
        }

        # Format context
        result = await injector.prepare_context_for_tool(
            "continue-101", "project-123", session_context
        )

        # Verify Continue-specific structure
        assert "context" in result
        context = result["context"]

        assert "files" in context
        assert "searches" in context
        assert "conversation_starters" in context

        # Verify content
        assert "src/feature.ts" in context["files"]
        assert "typescript types" in context["searches"]
        assert len(context["conversation_starters"]) > 0
        assert "Continue with:" in context["conversation_starters"][0]


class TestAgentFormatting:
    """Test formatting for autonomous agents (n8n, custom agents)."""

    @pytest.mark.asyncio
    async def test_format_for_n8n_agent(self):
        """Test agents get raw JSON context."""
        registry = ToolRegistry()
        injector = ContextInjector(registry)

        # Register n8n agent
        tool = await registry.register_tool(
            "n8n-agent-999",
            "n8n-agent",
            {"webhook_url": "https://n8n.example.com/webhook/abc123"},
        )

        session_context = {
            "files_accessed": [
                {"path": "src/data.json", "importance": 0.9},
                {"path": "src/processor.py", "importance": 0.7},
            ],
            "recent_searches": [{"query": "data processing"}],
            "decisions": [{"decision": "Use pandas for data processing"}],
            "saved_memories": [{"memory": "User prefers verbose logging"}],
            "file_importance_scores": {"src/data.json": 0.9},
        }

        # Format context
        result = await injector.prepare_context_for_tool(
            "n8n-agent-999", "project-123", session_context
        )

        # Verify raw format
        assert result["context_type"] == "raw"

        # Verify project context (raw, unformatted)
        project_ctx = result["project_context"]
        assert "files_accessed" in project_ctx
        assert "recent_searches" in project_ctx
        assert "decisions" in project_ctx
        assert "saved_memories" in project_ctx
        assert "file_importance_scores" in project_ctx

        # Verify all data is preserved (not truncated)
        assert len(project_ctx["files_accessed"]) == 2
        assert len(project_ctx["recent_searches"]) == 1
        assert len(project_ctx["decisions"]) == 1
        assert len(project_ctx["saved_memories"]) == 1

        # Verify metadata
        metadata = result["metadata"]
        assert metadata["agent_id"] == "n8n-agent-999"
        assert metadata["agent_type"] == "n8n-agent"
        assert "context_retrieved_at" in metadata
        assert "max_context_tokens" in metadata

    @pytest.mark.asyncio
    async def test_format_for_custom_agent(self):
        """Test custom agents get raw context."""
        registry = ToolRegistry()
        injector = ContextInjector(registry)

        # Register custom agent
        tool = await registry.register_tool(
            "custom-agent-555",
            "custom-agent",
            {
                "webhook_url": "https://agent.example.com/context",
                "api_key": "secret123",
            },
        )

        session_context = {
            "files_accessed": [{"path": "src/agent.py", "importance": 1.0}],
            "recent_searches": [],
            "decisions": [],
        }

        # Format context
        result = await injector.prepare_context_for_tool(
            "custom-agent-555", "project-123", session_context
        )

        # Verify it's raw format
        assert result["context_type"] == "raw"
        assert result["metadata"]["agent_id"] == "custom-agent-555"
        assert result["metadata"]["agent_type"] == "custom-agent"

    @pytest.mark.asyncio
    async def test_agent_vs_ide_formatting_difference(self):
        """Test agents get raw, IDEs get formatted."""
        registry = ToolRegistry()
        injector = ContextInjector(registry)

        # Register both types
        cursor = await registry.register_tool("cursor-123", "cursor")
        agent = await registry.register_tool("n8n-agent-456", "n8n-agent")

        session_context = {
            "files_accessed": [{"path": "src/test.py", "importance": 0.8}],
            "recent_searches": [{"query": "testing"}],
            "decisions": [{"decision": "Use pytest"}],
        }

        # Format for Cursor (IDE)
        cursor_result = await injector.prepare_context_for_tool(
            "cursor-123", "project-123", session_context
        )

        # Format for Agent
        agent_result = await injector.prepare_context_for_tool(
            "n8n-agent-456", "project-123", session_context
        )

        # Cursor gets formatted prompt
        assert "system_prompt_addition" in cursor_result
        assert "**Recently Accessed Files:**" in cursor_result["system_prompt_addition"]

        # Agent gets raw JSON
        assert agent_result["context_type"] == "raw"
        assert "system_prompt_addition" not in agent_result
        assert "project_context" in agent_result


class TestHelperMethods:
    """Test helper formatting methods."""

    def test_format_files_for_display(self):
        """Test file formatting with importance bars."""
        files = [
            {"path": "src/main.py", "importance": 1.0},
            {"path": "src/utils.py", "importance": 0.6},
            {"path": "src/config.py", "importance": 0.2},
        ]

        result = ContextInjector._format_files_for_display(files, limit=10)

        # Should contain paths
        assert "src/main.py" in result
        assert "src/utils.py" in result
        assert "src/config.py" in result

        # Should contain importance bars
        assert "█████" in result  # 1.0 importance = 5 filled blocks
        assert "███" in result  # 0.6 importance = 3 filled blocks
        assert "█" in result  # 0.2 importance = 1 filled block

    def test_format_files_for_display_empty(self):
        """Test formatting empty file list."""
        result = ContextInjector._format_files_for_display([])
        assert result == "  (none)"

    def test_format_files_for_display_limit(self):
        """Test file list respects limit."""
        files = [{"path": f"file{i}.py", "importance": 0.5} for i in range(20)]
        result = ContextInjector._format_files_for_display(files, limit=5)

        # Should only have 5 files
        lines = [line for line in result.split("\n") if line.strip()]
        assert len(lines) == 5

    def test_format_searches_for_display(self):
        """Test search history formatting."""
        searches = [
            {"query": "authentication"},
            {"query": "database models"},
            {"query": "API endpoints"},
        ]

        result = ContextInjector._format_searches_for_display(searches, limit=5)

        # Should contain queries
        assert '"authentication"' in result
        assert '"database models"' in result
        assert '"API endpoints"' in result

        # Should have bullet points
        assert "  - " in result

    def test_format_searches_for_display_empty(self):
        """Test formatting empty search list."""
        result = ContextInjector._format_searches_for_display([])
        assert result == "  (none)"

    def test_format_decisions(self):
        """Test decision formatting."""
        decisions = [
            {"decision": "Use FastAPI framework"},
            {"decision": "PostgreSQL for database"},
            {"decision": "JWT for authentication"},
        ]

        result = ContextInjector._format_decisions(decisions, limit=3)

        # Should contain decision text
        assert "Use FastAPI framework" in result
        assert "PostgreSQL for database" in result
        assert "JWT for authentication" in result

        # Should have bullet points
        assert "  • " in result

    def test_format_decisions_empty(self):
        """Test formatting empty decisions list."""
        result = ContextInjector._format_decisions([])
        assert result == "  (none)"


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_unknown_tool_id(self):
        """Test handling of unknown tool ID."""
        registry = ToolRegistry()
        injector = ContextInjector(registry)

        session_context = {"files_accessed": []}

        # Try to format for non-existent tool
        result = await injector.prepare_context_for_tool(
            "unknown-999", "project-123", session_context
        )

        # Should return empty dict
        assert result == {}

    @pytest.mark.asyncio
    async def test_empty_context(self):
        """Test handling of empty context."""
        registry = ToolRegistry()
        injector = ContextInjector(registry)

        # Register tool
        tool = await registry.register_tool("cursor-123", "cursor")

        # Empty context
        session_context = {}

        # Format context
        result = await injector.prepare_context_for_tool(
            "cursor-123", "project-123", session_context
        )

        # Should still have structure
        assert "system_prompt_addition" in result
        assert "context_metadata" in result

        # Metadata should have empty lists
        metadata = result["context_metadata"]
        assert metadata["files_accessed"] == []
        assert metadata["recent_searches"] == []
        assert metadata["decisions"] == []

    @pytest.mark.asyncio
    async def test_malformed_context_data(self):
        """Test handling of malformed context data."""
        registry = ToolRegistry()
        injector = ContextInjector(registry)

        tool = await registry.register_tool("cursor-123", "cursor")

        # Malformed data (missing expected fields)
        session_context = {
            "files_accessed": [
                {"importance": 0.8},  # Missing 'path'
                {"path": "valid.py"},  # Missing 'importance'
            ],
            "recent_searches": [
                {},  # Missing 'query'
            ],
        }

        # Should not crash
        result = await injector.prepare_context_for_tool(
            "cursor-123", "project-123", session_context
        )

        # Should have structure
        assert "system_prompt_addition" in result

    @pytest.mark.asyncio
    async def test_unknown_tool_type(self):
        """Test handling of unknown tool type."""
        registry = ToolRegistry()
        injector = ContextInjector(registry)

        # Manually create tool with unknown type
        tool = ToolInfo(
            tool_id="unknown-tool-123",
            tool_type="unknown-ide",
            connected_at=datetime.now(),
            last_activity=datetime.now(),
            capabilities={"supports_mcp": True, "supports_rest": False},
        )
        registry.tools["unknown-tool-123"] = tool

        session_context = {"files_accessed": []}

        # Format context
        result = await injector.prepare_context_for_tool(
            "unknown-tool-123", "project-123", session_context
        )

        # Should return raw context as fallback
        assert result == session_context

    @pytest.mark.asyncio
    async def test_get_other_tools_in_project(self):
        """Test getting other tools in project."""
        registry = ToolRegistry()
        injector = ContextInjector(registry)

        # Register multiple tools
        cursor = await registry.register_tool("cursor-123", "cursor")
        vscode = await registry.register_tool("vscode-456", "vscode")

        # Link them to same project
        await registry.link_tool_to_session("cursor-123", "session-1", "project-123")
        await registry.link_tool_to_session("vscode-456", "session-1", "project-123")

        # Get other tools
        other_tools = injector._get_other_tools_in_project(cursor)

        # Should return VSCode (not Cursor itself)
        assert len(other_tools) == 1
        assert other_tools[0]["tool_id"] == "vscode-456"
        assert other_tools[0]["tool_type"] == "vscode"
        assert "last_activity" in other_tools[0]

    @pytest.mark.asyncio
    async def test_get_other_tools_no_project(self):
        """Test getting other tools when tool has no project."""
        registry = ToolRegistry()
        injector = ContextInjector(registry)

        # Register tool but don't link to project
        tool = await registry.register_tool("cursor-123", "cursor")

        # Get other tools (should be empty)
        other_tools = injector._get_other_tools_in_project(tool)
        assert other_tools == []


class TestIntegration:
    """Integration tests with ToolRegistry."""

    @pytest.mark.asyncio
    async def test_full_workflow_ide_tool(self):
        """Test full workflow for IDE tool."""
        registry = ToolRegistry()
        injector = ContextInjector(registry)

        # 1. Register tool
        tool = await registry.register_tool("cursor-123", "cursor")

        # 2. Link to project
        await registry.link_tool_to_session("cursor-123", "session-1", "project-123")

        # 3. Prepare context
        session_context = {
            "files_accessed": [
                {"path": "src/app.py", "importance": 0.9},
                {"path": "src/db.py", "importance": 0.7},
            ],
            "recent_searches": [
                {"query": "database connection"},
            ],
            "decisions": [
                {"decision": "Use SQLAlchemy ORM"},
            ],
        }

        result = await injector.prepare_context_for_tool(
            "cursor-123", "project-123", session_context
        )

        # Verify complete formatted output
        assert "system_prompt_addition" in result
        assert "src/app.py" in result["system_prompt_addition"]
        assert "database connection" in result["system_prompt_addition"]
        assert "Use SQLAlchemy ORM" in result["system_prompt_addition"]

    @pytest.mark.asyncio
    async def test_full_workflow_agent(self):
        """Test full workflow for autonomous agent."""
        registry = ToolRegistry()
        injector = ContextInjector(registry)

        # 1. Register agent
        tool = await registry.register_tool(
            "n8n-agent-456", "n8n-agent", {"webhook_url": "https://example.com/webhook"}
        )

        # 2. Link to project
        await registry.link_tool_to_session("n8n-agent-456", "session-1", "project-456")

        # 3. Prepare context
        session_context = {
            "files_accessed": [{"path": "data.json", "importance": 1.0}],
            "recent_searches": [{"query": "JSON parsing"}],
            "decisions": [],
            "saved_memories": [{"memory": "Process data daily at 2am"}],
            "file_importance_scores": {"data.json": 1.0},
        }

        result = await injector.prepare_context_for_tool(
            "n8n-agent-456", "project-456", session_context
        )

        # Verify raw format
        assert result["context_type"] == "raw"
        assert result["project_context"]["files_accessed"][0]["path"] == "data.json"
        assert (
            result["project_context"]["saved_memories"][0]["memory"]
            == "Process data daily at 2am"
        )
        assert result["metadata"]["agent_id"] == "n8n-agent-456"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
