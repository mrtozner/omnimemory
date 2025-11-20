"""
Context Injector for Phase 2 Multi-Tool Context Bridge.

Formats context for tool-specific protocols:
- IDE Tools (Cursor, VSCode, Claude Code): Formatted context via MCP/LSP
- Agents (n8n, custom): Raw JSON context via REST API
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ContextInjector:
    """Injects context into tool-specific formats.

    Two modes:
    1. IDE Tools: Format for MCP/LSP injection (system prompt, metadata)
    2. Agents: Return raw JSON (agent handles its own formatting)
    """

    def __init__(self, tool_registry, context_merger=None):
        """Initialize ContextInjector.

        Args:
            tool_registry: ToolRegistry instance for accessing tool info
            context_merger: Optional ContextMerger for merging contexts (future use)
        """
        self.registry = tool_registry
        self.merger = context_merger
        logger.info("ContextInjector initialized")

    async def prepare_context_for_tool(
        self, tool_id: str, project_id: str, session_context: dict
    ) -> dict:
        """Prepare context suitable for a specific tool.

        Args:
            tool_id: Tool identifier
            project_id: Project identifier
            session_context: Raw session context

        Returns:
            For IDE tools: Formatted context (MCP/LSP)
            For agents: Raw context (JSON)
        """
        tool = self.registry.tools.get(tool_id)
        if not tool:
            logger.warning(f"Tool {tool_id} not found, returning empty context")
            return {}

        # Check if tool is agent (supports REST API)
        if tool.capabilities.get("supports_rest"):
            # Agent: return raw context
            logger.debug(f"Formatting raw context for agent {tool_id}")
            return self._format_for_agent(session_context, tool)

        # IDE tool: format based on type
        logger.debug(f"Formatting IDE context for {tool.tool_type}: {tool_id}")

        if tool.tool_type == "cursor":
            return self._format_for_cursor(session_context, tool)
        elif tool.tool_type == "claude-code":
            return self._format_for_claude_code(session_context, tool)
        elif tool.tool_type == "vscode":
            return self._format_for_vscode(session_context, tool)
        elif tool.tool_type == "continue":
            return self._format_for_continue(session_context, tool)
        else:
            logger.warning(f"Unknown tool type {tool.tool_type}, returning raw context")
            return session_context

    @staticmethod
    def _format_for_cursor(context: dict, tool) -> dict:
        """Format context for Cursor IDE.

        Cursor uses MCP protocol, prefers file-level context.
        Can handle large context windows (20K tokens).

        Args:
            context: Raw session context
            tool: ToolInfo object

        Returns:
            Formatted context for Cursor with system prompt addition
        """
        files_list = ContextInjector._format_files_for_display(
            context.get("files_accessed", []), limit=10
        )

        searches_list = ContextInjector._format_searches_for_display(
            context.get("recent_searches", []), limit=5
        )

        decisions_list = ContextInjector._format_decisions(
            context.get("decisions", []), limit=3
        )

        system_prompt = f"""
Based on your previous sessions in this project:

**Recently Accessed Files:**
{files_list}

**Recent Searches:**
{searches_list}

**Key Decisions Made:**
{decisions_list}

Continue from where you left off. Use this context to maintain consistency.
"""

        return {
            "system_prompt_addition": system_prompt,
            "context_metadata": {
                "files_accessed": context.get("files_accessed", [])[:10],
                "recent_searches": context.get("recent_searches", [])[:5],
                "decisions": context.get("decisions", [])[:3],
                "tools_active": ContextInjector._get_other_tools_in_project_static(
                    tool, context
                ),
            },
        }

    @staticmethod
    def _format_for_claude_code(context: dict, tool) -> dict:
        """Format context for Claude Code.

        Claude Code is primary AI tool, gets full context.
        Uses same MCP format as Cursor.

        Args:
            context: Raw session context
            tool: ToolInfo object

        Returns:
            Formatted context for Claude Code
        """
        # Same as Cursor (both use MCP)
        return ContextInjector._format_for_cursor(context, tool)

    @staticmethod
    def _format_for_vscode(context: dict, tool) -> dict:
        """Format context for VSCode.

        VSCode can use both LSP (for symbols) and MCP (for context).
        Should include LSP symbol information if available.

        Args:
            context: Raw session context
            tool: ToolInfo object

        Returns:
            Formatted context with MCP, LSP, and UI hints
        """
        # Get file paths for context display
        file_paths = [f.get("path", "") for f in context.get("files_accessed", [])[:5]]
        search_queries = [
            s.get("query", "") for s in context.get("recent_searches", [])[:3]
        ]

        system_prompt = f"""
Previous session context (from Cursor/Claude Code):

Files: {', '.join(file_paths) if file_paths else '(none)'}
Recent searches: {', '.join(search_queries) if search_queries else '(none)'}

Continue in the same direction.
"""

        return {
            # MCP context (same as Cursor/Claude)
            "mcp_context": {
                "system_prompt_addition": system_prompt,
                "context_metadata": {
                    "files_accessed": context.get("files_accessed", [])[:10]
                },
            },
            # LSP symbol context (VSCode-specific)
            "lsp_context": {
                "symbols_to_watch": [
                    {
                        "file": f.get("path", ""),
                        "type": "file",
                        "importance": f.get("importance", 0.5),
                    }
                    for f in context.get("files_accessed", [])[:5]
                ]
            },
            # UI hints for VSCode
            "ui_hints": {
                "show_context_panel": True,
                "highlight_files": file_paths,
                "sidebar_state": "context",
            },
        }

    @staticmethod
    def _format_for_continue(context: dict, tool) -> dict:
        """Format context for Continue.dev IDE plugin.

        Continue uses simplified context with conversation starters.

        Args:
            context: Raw session context
            tool: ToolInfo object

        Returns:
            Formatted context for Continue
        """
        file_paths = [f.get("path", "") for f in context.get("files_accessed", [])[:5]]
        search_queries = [
            s.get("query", "") for s in context.get("recent_searches", [])[:5]
        ]

        # Create conversation starter from recent files
        conversation_starter = "Continue with: "
        if file_paths[:3]:
            conversation_starter += ", ".join(file_paths[:3])
        else:
            conversation_starter = "Start a new conversation"

        return {
            "context": {
                "files": file_paths,
                "searches": search_queries,
                "conversation_starters": [conversation_starter],
            }
        }

    @staticmethod
    def _format_for_agent(context: dict, tool) -> dict:
        """Format context for autonomous agents.

        Agents get RAW context (no formatting).
        They handle their own processing.

        Args:
            context: Raw session context
            tool: ToolInfo object

        Returns:
            {
                "context_type": "raw",
                "project_context": {
                    "files_accessed": [...],
                    "recent_searches": [...],
                    "decisions": [...],
                    "file_importance_scores": {...}
                },
                "metadata": {
                    "agent_id": "n8n-agent-123",
                    "agent_type": "n8n-agent",
                    "context_retrieved_at": "2025-11-15T10:00:00Z"
                }
            }
        """
        return {
            "context_type": "raw",
            "project_context": {
                "files_accessed": context.get("files_accessed", []),
                "recent_searches": context.get("recent_searches", []),
                "decisions": context.get("decisions", []),
                "saved_memories": context.get("saved_memories", []),
                "file_importance_scores": context.get("file_importance_scores", {}),
            },
            "metadata": {
                "agent_id": tool.tool_id,
                "agent_type": tool.tool_type,
                "context_retrieved_at": datetime.now().isoformat(),
                "max_context_tokens": tool.capabilities.get(
                    "max_context_tokens", 10000
                ),
            },
        }

    # Helper methods from spec

    @staticmethod
    def _format_files_for_display(files: List[dict], limit: int = 10) -> str:
        """Format file list for display in prompt.

        Args:
            files: List of file records with path and importance
            limit: Maximum number of files to display

        Returns:
            Formatted string with importance bars
        """
        items = []
        for file_rec in files[:limit]:
            path = file_rec.get("path", "unknown")
            importance = file_rec.get("importance", 0.5)

            # Create importance bar (█ = filled, ░ = empty)
            filled = int(importance * 5)
            importance_bar = "█" * filled + "░" * (5 - filled)
            items.append(f"  {path} [{importance_bar}]")

        return "\n".join(items) if items else "  (none)"

    @staticmethod
    def _format_searches_for_display(searches: List[dict], limit: int = 5) -> str:
        """Format search history for display.

        Args:
            searches: List of search records with query
            limit: Maximum number of searches to display

        Returns:
            Formatted string with search queries
        """
        items = []
        for search in searches[:limit]:
            query = search.get("query", "")
            items.append(f'  - "{query}"')

        return "\n".join(items) if items else "  (none)"

    @staticmethod
    def _format_decisions(decisions: List[dict], limit: int = 3) -> str:
        """Format decisions for display.

        Args:
            decisions: List of decision records
            limit: Maximum number of decisions to display

        Returns:
            Formatted string with decision points
        """
        items = []
        for decision in decisions[:limit]:
            text = decision.get("decision", "")
            items.append(f"  • {text}")

        return "\n".join(items) if items else "  (none)"

    def _get_other_tools_in_project(self, tool) -> List[dict]:
        """Get list of other tools currently using this project.

        Args:
            tool: ToolInfo object for current tool

        Returns:
            List of dict with tool_id, tool_type, last_activity
        """
        if not tool.current_project_id:
            return []

        other_tools = self.registry.get_tools_for_project(tool.current_project_id)

        return [
            {
                "tool_id": t.tool_id,
                "tool_type": t.tool_type,
                "last_activity": t.last_activity.isoformat(),
            }
            for t in other_tools
            if t.tool_id != tool.tool_id
        ]

    @staticmethod
    def _get_other_tools_in_project_static(tool, context: dict) -> List[dict]:
        """Static helper to get other tools from context.

        This is used in static methods where we don't have access to registry.

        Args:
            tool: ToolInfo object
            context: Session context that may contain tools_active

        Returns:
            List of other tools or empty list
        """
        # Get from context if available
        return context.get("tools_active", [])
