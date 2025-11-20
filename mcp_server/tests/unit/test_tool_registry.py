"""
Unit tests for ToolRegistry class.

Tests registration, session management, and coordination of both IDE tools
and autonomous agents.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tool_registry import ToolInfo, ToolRegistry


class TestToolInfo:
    """Test ToolInfo model."""

    def test_tool_info_creation_ide(self):
        """Test creating ToolInfo for IDE tool."""
        tool = ToolInfo(
            tool_id="cursor-123",
            tool_type="cursor",
            connected_at=datetime.now(),
            last_activity=datetime.now(),
            capabilities={
                "supports_mcp": True,
                "supports_rest": False,
                "max_context_tokens": 20000,
            },
        )

        assert tool.tool_id == "cursor-123"
        assert tool.tool_type == "cursor"
        assert tool.capabilities["supports_mcp"] is True
        assert tool.capabilities["supports_rest"] is False
        assert tool.webhook_url is None

    def test_tool_info_creation_agent(self):
        """Test creating ToolInfo for autonomous agent."""
        tool = ToolInfo(
            tool_id="n8n-agent-456",
            tool_type="n8n-agent",
            connected_at=datetime.now(),
            last_activity=datetime.now(),
            capabilities={
                "supports_mcp": False,
                "supports_rest": True,
                "requires_webhook": True,
            },
            webhook_url="https://n8n.example.com/webhook/abc123",
            api_key="secret-key",
        )

        assert tool.tool_id == "n8n-agent-456"
        assert tool.tool_type == "n8n-agent"
        assert tool.capabilities["supports_rest"] is True
        assert tool.webhook_url == "https://n8n.example.com/webhook/abc123"
        assert tool.api_key == "secret-key"


class TestToolRegistry:
    """Test ToolRegistry class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ToolRegistry()

    @pytest.mark.asyncio
    async def test_register_ide_tool(self):
        """Test registering an IDE tool."""
        tool = await self.registry.register_tool("cursor-123", "cursor")

        assert tool.tool_id == "cursor-123"
        assert tool.tool_type == "cursor"
        assert tool.capabilities["supports_mcp"] is True
        assert tool.capabilities["supports_rest"] is False
        assert "cursor-123" in self.registry.tools
        assert "cursor-123" in self.registry.tool_locks

    @pytest.mark.asyncio
    async def test_register_agent_tool(self):
        """Test registering an autonomous agent."""
        tool = await self.registry.register_tool(
            "n8n-agent-456",
            "n8n-agent",
            config={"webhook_url": "https://n8n.example.com/webhook/abc123"},
        )

        assert tool.tool_id == "n8n-agent-456"
        assert tool.tool_type == "n8n-agent"
        assert tool.capabilities["supports_rest"] is True
        assert tool.capabilities["supports_mcp"] is False
        assert tool.webhook_url == "https://n8n.example.com/webhook/abc123"
        assert "n8n-agent-456" in self.registry.tools

    @pytest.mark.asyncio
    async def test_register_tool_reconnection(self):
        """Test tool reconnection updates last_activity."""
        # First connection
        tool1 = await self.registry.register_tool("cursor-123", "cursor")
        first_activity = tool1.last_activity

        # Wait a bit
        await asyncio.sleep(0.01)

        # Reconnection
        tool2 = await self.registry.register_tool("cursor-123", "cursor")

        assert tool2.last_activity > first_activity
        assert tool1.tool_id == tool2.tool_id

    @pytest.mark.asyncio
    async def test_link_tool_to_session(self):
        """Test linking a tool to a session."""
        # Register tool
        await self.registry.register_tool("cursor-123", "cursor")

        # Link to session
        await self.registry.link_tool_to_session(
            "cursor-123", "session-abc", "project-xyz"
        )

        # Verify mappings
        assert self.registry.tool_sessions["cursor-123"] == "session-abc"
        assert "cursor-123" in self.registry.project_tools["project-xyz"]
        assert self.registry.tools["cursor-123"].current_session_id == "session-abc"
        assert self.registry.tools["cursor-123"].current_project_id == "project-xyz"

    @pytest.mark.asyncio
    async def test_link_unregistered_tool_raises_error(self):
        """Test linking unregistered tool raises KeyError."""
        with pytest.raises(KeyError):
            await self.registry.link_tool_to_session(
                "unknown-tool", "session-abc", "project-xyz"
            )

    @pytest.mark.asyncio
    async def test_get_tools_for_project(self):
        """Test getting all tools for a project."""
        # Register multiple tools
        await self.registry.register_tool("cursor-123", "cursor")
        await self.registry.register_tool("vscode-456", "vscode")
        await self.registry.register_tool("n8n-agent-789", "n8n-agent")

        # Link to same project
        await self.registry.link_tool_to_session(
            "cursor-123", "session-1", "project-xyz"
        )
        await self.registry.link_tool_to_session(
            "vscode-456", "session-2", "project-xyz"
        )
        await self.registry.link_tool_to_session(
            "n8n-agent-789", "session-3", "project-xyz"
        )

        # Get tools for project
        tools = self.registry.get_tools_for_project("project-xyz")

        assert len(tools) == 3
        tool_ids = [t.tool_id for t in tools]
        assert "cursor-123" in tool_ids
        assert "vscode-456" in tool_ids
        assert "n8n-agent-789" in tool_ids

    @pytest.mark.asyncio
    async def test_get_tools_for_session(self):
        """Test getting all tools for a session."""
        # Register tools
        await self.registry.register_tool("cursor-123", "cursor")
        await self.registry.register_tool("vscode-456", "vscode")

        # Link to same session
        await self.registry.link_tool_to_session(
            "cursor-123", "session-abc", "project-xyz"
        )
        await self.registry.link_tool_to_session(
            "vscode-456", "session-abc", "project-xyz"
        )

        # Get tools for session
        tools = self.registry.get_tools_for_session("session-abc")

        assert len(tools) == 2
        tool_ids = [t.tool_id for t in tools]
        assert "cursor-123" in tool_ids
        assert "vscode-456" in tool_ids

    @pytest.mark.asyncio
    async def test_multi_tool_scenario(self):
        """Test scenario with 2 IDEs and 1 agent on same project."""
        # Register tools
        await self.registry.register_tool("cursor-123", "cursor")
        await self.registry.register_tool("vscode-456", "vscode")
        await self.registry.register_tool(
            "custom-agent-789",
            "custom-agent",
            config={"webhook_url": "https://agent.example.com/webhook"},
        )

        # Link all to same project
        await self.registry.link_tool_to_session(
            "cursor-123", "session-1", "project-xyz"
        )
        await self.registry.link_tool_to_session(
            "vscode-456", "session-2", "project-xyz"
        )
        await self.registry.link_tool_to_session(
            "custom-agent-789", "session-3", "project-xyz"
        )

        # Verify all tools are tracked
        tools = self.registry.get_tools_for_project("project-xyz")
        assert len(tools) == 3

        # Verify IDE tools have MCP support
        cursor = next(t for t in tools if t.tool_id == "cursor-123")
        assert cursor.capabilities["supports_mcp"] is True

        # Verify agent has REST support
        agent = next(t for t in tools if t.tool_id == "custom-agent-789")
        assert agent.capabilities["supports_rest"] is True
        assert agent.webhook_url == "https://agent.example.com/webhook"

    @pytest.mark.asyncio
    async def test_unregister_tool(self):
        """Test unregistering a tool."""
        # Register and link tool
        await self.registry.register_tool("cursor-123", "cursor")
        await self.registry.link_tool_to_session(
            "cursor-123", "session-abc", "project-xyz"
        )

        # Unregister
        await self.registry.unregister_tool("cursor-123")

        # Verify cleanup
        assert "cursor-123" not in self.registry.tools
        assert "cursor-123" not in self.registry.tool_sessions
        assert "cursor-123" not in self.registry.tool_locks
        assert "cursor-123" not in self.registry.project_tools.get("project-xyz", [])

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_tool(self):
        """Test unregistering a nonexistent tool doesn't raise error."""
        # Should not raise
        await self.registry.unregister_tool("unknown-tool")

    @pytest.mark.asyncio
    async def test_update_activity(self):
        """Test updating tool activity timestamp."""
        tool = await self.registry.register_tool("cursor-123", "cursor")
        first_activity = tool.last_activity

        await asyncio.sleep(0.01)

        await self.registry.update_activity("cursor-123")

        assert self.registry.tools["cursor-123"].last_activity > first_activity

    def test_get_tool(self):
        """Test getting tool by ID."""
        # Tool doesn't exist
        assert self.registry.get_tool("unknown") is None

    @pytest.mark.asyncio
    async def test_get_tool_exists(self):
        """Test getting existing tool."""
        await self.registry.register_tool("cursor-123", "cursor")
        tool = self.registry.get_tool("cursor-123")

        assert tool is not None
        assert tool.tool_id == "cursor-123"

    @pytest.mark.asyncio
    async def test_get_context_for_vscode(self):
        """Test getting context formatted for VSCode."""
        await self.registry.register_tool("vscode-123", "vscode")

        session_context = {
            "files_accessed": ["/path/to/file.py"],
            "recent_searches": ["search query"],
        }

        context = await self.registry.get_context_for_tool(
            "vscode-123", session_context
        )

        assert context["format"] == "vscode-lsp"

    @pytest.mark.asyncio
    async def test_get_context_for_cursor(self):
        """Test getting context formatted for Cursor."""
        await self.registry.register_tool("cursor-123", "cursor")

        session_context = {
            "files_accessed": ["/path/to/file.py"],
            "recent_searches": ["search query"],
        }

        context = await self.registry.get_context_for_tool(
            "cursor-123", session_context
        )

        assert context["format"] == "cursor-mcp"

    @pytest.mark.asyncio
    async def test_get_context_for_agent(self):
        """Test getting context formatted for autonomous agent."""
        await self.registry.register_tool(
            "n8n-agent-123",
            "n8n-agent",
            config={"webhook_url": "https://n8n.example.com/webhook"},
        )

        session_context = {
            "files_accessed": [f"/file{i}.py" for i in range(20)],  # 20 files
            "recent_searches": [f"query {i}" for i in range(10)],  # 10 searches
            "decisions": [{"decision": f"dec {i}"} for i in range(10)],  # 10 decisions
        }

        context = await self.registry.get_context_for_tool(
            "n8n-agent-123", session_context
        )

        assert context["format"] == "agent-rest"
        assert len(context["files_accessed"]) == 10  # Limited to 10
        assert len(context["recent_searches"]) == 5  # Limited to 5
        assert len(context["decisions"]) == 5  # Limited to 5

    @pytest.mark.asyncio
    async def test_get_context_for_unknown_tool(self):
        """Test getting context for unknown tool returns raw context."""
        session_context = {"test": "data"}

        context = await self.registry.get_context_for_tool(
            "unknown-tool", session_context
        )

        assert context == session_context

    def test_get_capabilities_ide_tools(self):
        """Test capabilities for IDE tools."""
        # Cursor
        cursor_caps = ToolRegistry._get_capabilities("cursor")
        assert cursor_caps["supports_mcp"] is True
        assert cursor_caps["supports_rest"] is False
        assert cursor_caps["can_execute_code"] is True

        # VSCode
        vscode_caps = ToolRegistry._get_capabilities("vscode")
        assert vscode_caps["supports_lsp"] is True
        assert vscode_caps["supports_mcp"] is True

        # Claude Code
        claude_caps = ToolRegistry._get_capabilities("claude-code")
        assert claude_caps["supports_mcp"] is True
        assert claude_caps["can_execute_code"] is True

        # Continue
        continue_caps = ToolRegistry._get_capabilities("continue")
        assert continue_caps["supports_mcp"] is True
        assert continue_caps["can_execute_code"] is False

    def test_get_capabilities_agent_tools(self):
        """Test capabilities for autonomous agent tools."""
        # n8n agent
        n8n_caps = ToolRegistry._get_capabilities("n8n-agent")
        assert n8n_caps["supports_rest"] is True
        assert n8n_caps["supports_mcp"] is False
        assert n8n_caps["requires_webhook"] is True
        assert n8n_caps["can_execute_code"] is False

        # Custom agent
        custom_caps = ToolRegistry._get_capabilities("custom-agent")
        assert custom_caps["supports_rest"] is True
        assert custom_caps["can_execute_code"] is True

        # LangChain agent
        langchain_caps = ToolRegistry._get_capabilities("langchain-agent")
        assert langchain_caps["supports_rest"] is True
        assert langchain_caps["requires_webhook"] is True

        # AutoGen agent
        autogen_caps = ToolRegistry._get_capabilities("autogen-agent")
        assert autogen_caps["supports_rest"] is True
        assert autogen_caps["can_execute_code"] is True

    def test_get_capabilities_unknown_type(self):
        """Test capabilities for unknown tool type defaults to cursor."""
        unknown_caps = ToolRegistry._get_capabilities("unknown-type")
        cursor_caps = ToolRegistry._get_capabilities("cursor")

        assert unknown_caps == cursor_caps

    @pytest.mark.asyncio
    async def test_broadcast_to_project_tools(self):
        """Test broadcasting event to project tools."""
        # Register multiple tools
        await self.registry.register_tool("cursor-123", "cursor")
        await self.registry.register_tool("vscode-456", "vscode")

        # Link to same project
        await self.registry.link_tool_to_session(
            "cursor-123", "session-1", "project-xyz"
        )
        await self.registry.link_tool_to_session(
            "vscode-456", "session-2", "project-xyz"
        )

        # Mock _notify_tool
        with patch.object(
            self.registry, "_notify_tool", new_callable=AsyncMock
        ) as mock_notify:
            await self.registry._broadcast_to_project_tools(
                "project-xyz",
                event="test_event",
                data={"key": "value"},
                exclude_tool_id="cursor-123",
            )

            # Should notify vscode-456 only (cursor-123 excluded)
            mock_notify.assert_called_once_with(
                "vscode-456", "test_event", {"key": "value"}
            )

    @pytest.mark.asyncio
    async def test_notify_tool_mcp(self):
        """Test notifying MCP-based IDE tool."""
        await self.registry.register_tool("cursor-123", "cursor")

        # Should log debug message for MCP notification
        with patch("tool_registry.logger") as mock_logger:
            await self.registry._notify_tool(
                "cursor-123", "test_event", {"data": "test"}
            )

            # Check debug log was called
            mock_logger.debug.assert_called()

    @pytest.mark.asyncio
    async def test_notify_tool_rest(self):
        """Test notifying REST-based agent tool."""
        await self.registry.register_tool(
            "n8n-agent-123",
            "n8n-agent",
            config={
                "webhook_url": "https://n8n.example.com/webhook",
                "api_key": "secret-key",
            },
        )

        # Mock httpx client
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            await self.registry._notify_tool(
                "n8n-agent-123", "test_event", {"data": "test"}
            )

            # Verify POST was called with correct parameters
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args

            assert call_args[0][0] == "https://n8n.example.com/webhook"
            assert call_args[1]["json"]["event"] == "test_event"
            assert call_args[1]["json"]["data"] == {"data": "test"}
            assert call_args[1]["headers"]["Authorization"] == "Bearer secret-key"

    @pytest.mark.asyncio
    async def test_notify_tool_rest_no_webhook(self):
        """Test notifying agent without webhook URL logs warning."""
        await self.registry.register_tool("custom-agent-123", "custom-agent")

        with patch("tool_registry.logger") as mock_logger:
            await self.registry._notify_tool(
                "custom-agent-123", "test_event", {"data": "test"}
            )

            # Should log warning
            mock_logger.warning.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
