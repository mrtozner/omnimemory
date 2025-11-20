"""
Tool Registry for Phase 2 Multi-Tool Context Bridge.

Manages registration, tracking, and coordination of connected tools:
- IDE Tools (via MCP): Cursor, VSCode, Claude Code, Continue
- Autonomous Agents (via REST API): n8n, custom agents, LangChain, AutoGen

This registry enables multi-tool collaboration on the same project,
broadcasting events and coordinating shared context.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)


class ToolInfo(BaseModel):
    """Information about a connected tool (IDE or autonomous agent)."""

    tool_id: str  # "cursor-123", "vscode-456", "n8n-agent-789"
    tool_type: str  # "cursor", "claude-code", "vscode", "continue", "n8n-agent", "custom-agent"
    connected_at: datetime
    last_activity: datetime

    # Capabilities (what this tool can do)
    capabilities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool capabilities including supports_lsp, supports_mcp, supports_rest, etc.",
    )

    # Tool-specific settings
    config: Dict[str, Any] = Field(default_factory=dict)

    # Session tracking
    current_session_id: Optional[str] = None
    current_project_id: Optional[str] = None

    # Agent-specific fields
    webhook_url: Optional[str] = None  # For REST API callbacks to agents
    api_key: Optional[str] = None  # For authenticated agent communication


class ToolRegistry:
    """
    Maintains registry of connected tools (IDE tools + autonomous agents) and coordinates them.

    This registry tracks all tools connected to OmniMemory, whether they connect via:
    - MCP protocol (IDE tools like Cursor, VSCode, Claude Code, Continue)
    - REST API (autonomous agents like n8n, custom agents, LangChain, AutoGen)

    It coordinates multi-tool sessions, broadcasts events, and manages tool capabilities.
    """

    def __init__(self):
        """Initialize the tool registry."""
        self.tools: Dict[str, ToolInfo] = {}  # tool_id -> ToolInfo
        self.tool_sessions: Dict[str, str] = {}  # tool_id -> session_id
        self.project_tools: Dict[str, List[str]] = {}  # project_id -> [tool_ids]
        self.tool_locks: Dict[str, asyncio.Lock] = {}  # Prevent race conditions

        logger.info("ToolRegistry initialized")

    async def register_tool(
        self, tool_id: str, tool_type: str, config: Optional[dict] = None
    ) -> ToolInfo:
        """
        Register a tool (IDE or agent) when it connects.

        Args:
            tool_id: Unique identifier for this tool instance
            tool_type: Type of tool (cursor, vscode, n8n-agent, etc.)
            config: Optional tool-specific configuration

        Returns:
            ToolInfo object representing the registered tool

        Example:
            >>> registry = ToolRegistry()
            >>> tool = await registry.register_tool("cursor-123", "cursor")
            >>> print(tool.capabilities["supports_mcp"])
            True

            >>> agent = await registry.register_tool("n8n-agent-456", "n8n-agent", {
            ...     "webhook_url": "https://n8n.example.com/webhook/abc123"
            ... })
            >>> print(agent.capabilities["supports_rest"])
            True
        """
        if tool_id in self.tools:
            # Tool already registered (reconnection after crash/disconnect)
            logger.info(f"Tool {tool_id} reconnected, updating last_activity")
            self.tools[tool_id].last_activity = datetime.now()
            return self.tools[tool_id]

        # Create new tool registration
        tool = ToolInfo(
            tool_id=tool_id,
            tool_type=tool_type,
            connected_at=datetime.now(),
            last_activity=datetime.now(),
            capabilities=self._get_capabilities(tool_type),
            config=config or {},
        )

        # Extract agent-specific configuration
        if config:
            if "webhook_url" in config:
                tool.webhook_url = config["webhook_url"]
            if "api_key" in config:
                tool.api_key = config["api_key"]

        self.tools[tool_id] = tool
        self.tool_locks[tool_id] = asyncio.Lock()

        logger.info(
            f"Registered {tool_type} tool: {tool_id} (supports_mcp: {tool.capabilities.get('supports_mcp')}, supports_rest: {tool.capabilities.get('supports_rest')})"
        )

        return tool

    async def link_tool_to_session(
        self, tool_id: str, session_id: str, project_id: str
    ) -> None:
        """
        Link tool to project session.

        This is called when a tool opens a project. It updates mappings,
        broadcasts to other tools using the same project, and coordinates
        multi-tool collaboration.

        Args:
            tool_id: ID of the tool joining the session
            session_id: Session ID for this project
            project_id: Project ID being worked on

        Raises:
            KeyError: If tool_id is not registered

        Example:
            >>> await registry.link_tool_to_session("cursor-123", "session-abc", "project-xyz")
            # Other tools using project-xyz will receive "tool_joined" event
        """
        if tool_id not in self.tools:
            raise KeyError(f"Tool {tool_id} not registered")

        async with self.tool_locks[tool_id]:
            # Update mappings
            self.tool_sessions[tool_id] = session_id

            if project_id not in self.project_tools:
                self.project_tools[project_id] = []

            if tool_id not in self.project_tools[project_id]:
                self.project_tools[project_id].append(tool_id)

            # Update tool info
            self.tools[tool_id].current_session_id = session_id
            self.tools[tool_id].current_project_id = project_id
            self.tools[tool_id].last_activity = datetime.now()

            logger.info(
                f"Linked {tool_id} to session {session_id}, project {project_id}"
            )

        # Broadcast to all tools using this project (except the one that just joined)
        await self._broadcast_to_project_tools(
            project_id,
            event="tool_joined",
            data={
                "tool_id": tool_id,
                "tool_type": self.tools[tool_id].tool_type,
                "session_id": session_id,
                "capabilities": self.tools[tool_id].capabilities,
            },
            exclude_tool_id=tool_id,
        )

    def get_tools_for_project(self, project_id: str) -> List[ToolInfo]:
        """
        Get all tools (IDEs + agents) currently using this project.

        Args:
            project_id: Project ID to query

        Returns:
            List of ToolInfo objects for tools using this project

        Example:
            >>> tools = registry.get_tools_for_project("project-xyz")
            >>> for tool in tools:
            ...     print(f"{tool.tool_type}: {tool.tool_id}")
            cursor: cursor-123
            n8n-agent: n8n-agent-456
        """
        tool_ids = self.project_tools.get(project_id, [])
        return [self.tools[tid] for tid in tool_ids if tid in self.tools]

    def get_tools_for_session(self, session_id: str) -> List[ToolInfo]:
        """
        Get all tools using this session.

        Args:
            session_id: Session ID to query

        Returns:
            List of ToolInfo objects for tools in this session

        Example:
            >>> tools = registry.get_tools_for_session("session-abc")
            >>> print(len(tools))
            2
        """
        tool_ids = [tid for tid, sid in self.tool_sessions.items() if sid == session_id]
        return [self.tools[tid] for tid in tool_ids if tid in self.tools]

    async def get_context_for_tool(self, tool_id: str, session_context: dict) -> dict:
        """
        Get context suitable for specific tool.

        Different tools have different capabilities and context requirements.
        This method formats the session context appropriately for each tool type.

        Args:
            tool_id: ID of the tool requesting context
            session_context: Raw session context data

        Returns:
            Formatted context suitable for this tool

        Example:
            >>> context = await registry.get_context_for_tool("vscode-123", session_context)
            # VSCode receives LSP-formatted symbol information

            >>> context = await registry.get_context_for_tool("n8n-agent-456", session_context)
            # Agent receives simplified REST-friendly format
        """
        if tool_id not in self.tools:
            logger.warning(f"Tool {tool_id} not found, returning raw context")
            return session_context

        tool = self.tools[tool_id]

        # Format context based on tool type and capabilities
        if tool.tool_type == "vscode":
            # VSCode can use LSP, so include symbol information
            return self._format_for_vscode(session_context)
        elif tool.tool_type == "cursor":
            # Cursor prefers file-level context via MCP
            return self._format_for_cursor(session_context)
        elif tool.tool_type == "claude-code":
            # Claude Code supports full context
            return self._format_for_claude_code(session_context)
        elif tool.tool_type == "continue":
            # Continue uses simplified context
            return self._format_for_continue(session_context)
        elif tool.capabilities.get("supports_rest"):
            # Autonomous agents use REST API format
            return self._format_for_agent(session_context, tool)
        else:
            # Unknown tool type, return raw context
            return session_context

    async def unregister_tool(self, tool_id: str) -> None:
        """
        Unregister a tool when it closes.

        Cleans up all mappings and broadcasts to other tools in the same project.

        Args:
            tool_id: ID of the tool to unregister

        Example:
            >>> await registry.unregister_tool("cursor-123")
            # Other tools in the project receive "tool_left" event
        """
        if tool_id not in self.tools:
            logger.warning(f"Tool {tool_id} not found, cannot unregister")
            return

        async with self.tool_locks[tool_id]:
            project_id = self.tools[tool_id].current_project_id
            session_id = self.tools[tool_id].current_session_id

            # Clean up mappings
            if project_id and tool_id in self.project_tools.get(project_id, []):
                self.project_tools[project_id].remove(tool_id)

            if tool_id in self.tool_sessions:
                del self.tool_sessions[tool_id]

            del self.tools[tool_id]
            del self.tool_locks[tool_id]

            logger.info(f"Unregistered tool: {tool_id}")

        # Broadcast to other tools in the project
        if project_id:
            await self._broadcast_to_project_tools(
                project_id,
                event="tool_left",
                data={"tool_id": tool_id, "session_id": session_id},
                exclude_tool_id=tool_id,
            )

    async def update_activity(self, tool_id: str) -> None:
        """
        Update last activity timestamp for a tool.

        Args:
            tool_id: ID of the tool to update
        """
        if tool_id in self.tools:
            self.tools[tool_id].last_activity = datetime.now()

    def get_tool(self, tool_id: str) -> Optional[ToolInfo]:
        """
        Get tool information by ID.

        Args:
            tool_id: ID of the tool to retrieve

        Returns:
            ToolInfo object or None if not found
        """
        return self.tools.get(tool_id)

    async def _broadcast_to_project_tools(
        self,
        project_id: str,
        event: str,
        data: dict,
        exclude_tool_id: Optional[str] = None,
    ) -> None:
        """
        Broadcast event to all tools using this project.

        Handles both MCP-based IDE tools and REST-based autonomous agents.

        Args:
            project_id: Project ID to broadcast to
            event: Event name (e.g., "tool_joined", "tool_left", "context_updated")
            data: Event data
            exclude_tool_id: Optional tool ID to exclude from broadcast
        """
        tools = self.get_tools_for_project(project_id)

        for tool in tools:
            if exclude_tool_id and tool.tool_id == exclude_tool_id:
                continue

            try:
                # Send notification (implementation depends on tool type)
                await self._notify_tool(tool.tool_id, event, data)
            except Exception as e:
                logger.error(
                    f"Failed to notify tool {tool.tool_id} of event {event}: {e}"
                )

    async def _notify_tool(self, tool_id: str, event: str, data: dict) -> None:
        """
        Send notification to a specific tool.

        For IDE tools (MCP): Use MCP notification protocol
        For agents (REST): Send webhook callback

        Args:
            tool_id: ID of the tool to notify
            event: Event name
            data: Event data
        """
        if tool_id not in self.tools:
            return

        tool = self.tools[tool_id]

        if tool.capabilities.get("supports_mcp"):
            # MCP-based notification (for IDE tools)
            # This would integrate with the MCP server's notification system
            logger.debug(f"Sending MCP notification to {tool_id}: {event}")
            # TODO: Integrate with MCP server notification system

        elif tool.capabilities.get("supports_rest") and tool.webhook_url:
            # REST-based notification (for autonomous agents)
            logger.debug(f"Sending webhook notification to {tool_id}: {event}")
            try:
                import httpx

                async with httpx.AsyncClient() as client:
                    headers = {}
                    if tool.api_key:
                        headers["Authorization"] = f"Bearer {tool.api_key}"

                    await client.post(
                        tool.webhook_url,
                        json={"event": event, "data": data, "tool_id": tool_id},
                        headers=headers,
                        timeout=5.0,
                    )
                    logger.debug(f"Webhook sent successfully to {tool_id}")
            except Exception as e:
                logger.error(f"Failed to send webhook to {tool_id}: {e}")
        else:
            logger.warning(f"Tool {tool_id} has no notification mechanism configured")

    @staticmethod
    def _get_capabilities(tool_type: str) -> dict:
        """
        Get capabilities based on tool type.

        Defines what each tool type can do and how it should be treated.

        Args:
            tool_type: Type of tool (cursor, vscode, n8n-agent, etc.)

        Returns:
            Dictionary of capabilities

        Example:
            >>> caps = ToolRegistry._get_capabilities("cursor")
            >>> print(caps["supports_mcp"])
            True

            >>> caps = ToolRegistry._get_capabilities("n8n-agent")
            >>> print(caps["supports_rest"])
            True
        """
        capabilities_map = {
            # IDE Tools (MCP protocol)
            "cursor": {
                "supports_lsp": False,
                "supports_mcp": True,
                "supports_rest": False,
                "max_context_tokens": 20000,
                "compression_enabled": True,
                "can_execute_code": True,
                "can_edit_files": True,
            },
            "claude-code": {
                "supports_lsp": False,
                "supports_mcp": True,
                "supports_rest": False,
                "max_context_tokens": 20000,
                "compression_enabled": True,
                "can_execute_code": True,
                "can_edit_files": True,
            },
            "vscode": {
                "supports_lsp": True,
                "supports_mcp": True,
                "supports_rest": False,
                "max_context_tokens": 15000,
                "compression_enabled": True,
                "can_execute_code": False,
                "can_edit_files": True,
            },
            "continue": {
                "supports_lsp": False,
                "supports_mcp": True,
                "supports_rest": False,
                "max_context_tokens": 20000,
                "compression_enabled": True,
                "can_execute_code": False,
                "can_edit_files": False,
            },
            # Autonomous Agents (REST API)
            "n8n-agent": {
                "supports_lsp": False,
                "supports_mcp": False,
                "supports_rest": True,
                "max_context_tokens": 10000,
                "compression_enabled": True,
                "can_execute_code": False,
                "can_edit_files": False,
                "requires_webhook": True,
            },
            "custom-agent": {
                "supports_lsp": False,
                "supports_mcp": False,
                "supports_rest": True,
                "max_context_tokens": 15000,
                "compression_enabled": True,
                "can_execute_code": True,  # Custom agents may execute code
                "can_edit_files": False,
                "requires_webhook": True,
            },
            "langchain-agent": {
                "supports_lsp": False,
                "supports_mcp": False,
                "supports_rest": True,
                "max_context_tokens": 12000,
                "compression_enabled": True,
                "can_execute_code": True,
                "can_edit_files": False,
                "requires_webhook": True,
            },
            "autogen-agent": {
                "supports_lsp": False,
                "supports_mcp": False,
                "supports_rest": True,
                "max_context_tokens": 12000,
                "compression_enabled": True,
                "can_execute_code": True,
                "can_edit_files": False,
                "requires_webhook": True,
            },
        }

        # Return capabilities for known tool types, or default to cursor-like capabilities
        return capabilities_map.get(tool_type, capabilities_map["cursor"])

    # Context formatting methods for different tool types

    def _format_for_vscode(self, session_context: dict) -> dict:
        """Format context for VSCode (LSP-enabled)."""
        # VSCode can use LSP symbol information
        formatted = session_context.copy()
        formatted["format"] = "vscode-lsp"
        # Add LSP-specific formatting here
        return formatted

    def _format_for_cursor(self, session_context: dict) -> dict:
        """Format context for Cursor (file-level MCP)."""
        formatted = session_context.copy()
        formatted["format"] = "cursor-mcp"
        # Cursor prefers file-level context
        return formatted

    def _format_for_claude_code(self, session_context: dict) -> dict:
        """Format context for Claude Code (full context)."""
        formatted = session_context.copy()
        formatted["format"] = "claude-code-full"
        # Claude Code supports comprehensive context
        return formatted

    def _format_for_continue(self, session_context: dict) -> dict:
        """Format context for Continue (simplified)."""
        formatted = session_context.copy()
        formatted["format"] = "continue-simple"
        # Continue uses simplified context
        return formatted

    def _format_for_agent(self, session_context: dict, tool: ToolInfo) -> dict:
        """
        Format context for autonomous agents (REST API).

        Agents get a simplified, REST-friendly format with:
        - Essential context only (to fit token limits)
        - JSON-serializable data
        - No tool-specific formatting
        """
        # Simplify context for agents
        formatted = {
            "format": "agent-rest",
            "tool_id": tool.tool_id,
            "tool_type": tool.tool_type,
            "max_tokens": tool.capabilities.get("max_context_tokens", 10000),
            "session_id": tool.current_session_id,
            "project_id": tool.current_project_id,
            # Include only essential context fields
            "files_accessed": session_context.get("files_accessed", [])[
                :10
            ],  # Limit to 10 most recent
            "recent_searches": session_context.get("recent_searches", [])[
                :5
            ],  # Limit to 5 most recent
            "saved_memories": session_context.get("saved_memories", []),
            "decisions": session_context.get("decisions", [])[
                :5
            ],  # Limit to 5 most recent
        }

        return formatted
