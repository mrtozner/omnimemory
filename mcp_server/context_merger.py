"""
Context Merger for Multi-Tool Context Bridge.

Intelligently merges context from multiple IDE tools (Cursor, VSCode, Claude Code, Continue).

NOTE: This is ONLY for IDE tools with auto_merge_context=True.
Autonomous agents use pull-based context retrieval via REST API, NOT this merger.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ContextMerger:
    """Intelligently merges context from multiple IDE tools.

    NOTE: This is ONLY for IDE tools (Cursor, VSCode, Claude Code, Continue).
    Autonomous agents use pull-based context retrieval (not auto-merged).

    Design Decision:
    - IDE Tools: Auto-merge context (this class handles it)
    - Agents: Pull model - query raw context directly via REST API
    """

    def __init__(self, session_manager, project_manager):
        """Initialize ContextMerger with managers.

        Args:
            session_manager: SessionManager instance for session operations
            project_manager: ProjectManager instance for project operations
        """
        self.sessions = session_manager
        self.projects = project_manager

    async def merge_contexts(
        self, requesting_tool_id: str, incoming_context: Dict[str, Any], project_id: str
    ) -> Dict[str, Any]:
        """Merge new context with existing multi-tool context.

        Args:
            requesting_tool_id: IDE tool asking for merged context (e.g., "vscode-456")
            incoming_context: What this IDE tool is contributing
            project_id: Which project

        Returns:
            Merged context suitable for the requesting IDE tool

        Note: This method is ONLY called for IDE tools with auto_merge_context=True.
        Agents do NOT use this - they pull raw context via REST API.
        """
        logger.info(
            f"Merging context for tool={requesting_tool_id}, project={project_id}"
        )

        # Get existing session context
        session = self.sessions.get_session_for_project(project_id)
        existing = session.context.model_dump() if session else {}

        # Merge using strategies
        merged = {
            # File access history - keep most recent from all IDE tools
            "files_accessed": self._merge_file_access(
                existing.get("files_accessed", []),
                incoming_context.get("files_accessed", []),
            ),
            # Search history - combine without duplicates
            "recent_searches": self._merge_searches(
                existing.get("recent_searches", []),
                incoming_context.get("recent_searches", []),
                max_items=20,
            ),
            # Saved memories - combine
            "saved_memories": self._merge_memories(
                existing.get("saved_memories", []),
                incoming_context.get("saved_memories", []),
            ),
            # Decisions - keep all, prefer most recent
            "decisions": self._merge_decisions(
                existing.get("decisions", []),
                incoming_context.get("decisions", []),
                max_items=10,
            ),
            # Tool-specific data - don't override other tools' data
            "tool_specific": self._merge_tool_specific(
                existing.get("tool_specific", {}),
                incoming_context.get("tool_specific", {}),
                requesting_tool_id,
            ),
            # File importance scores - merge intelligently
            "file_importance_scores": self._merge_importance_scores(
                existing.get("file_importance_scores", {}),
                incoming_context.get("file_importance_scores", {}),
            ),
        }

        logger.debug(
            f"Merged context: {len(merged['files_accessed'])} files, "
            f"{len(merged['recent_searches'])} searches, "
            f"{len(merged['decisions'])} decisions"
        )

        return merged

    @staticmethod
    def _merge_file_access(
        existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge file access histories, keeping most recent.

        File access history should reflect what files have been accessed
        across ALL IDE tools, most recent first.

        Args:
            existing: Existing file access records
            incoming: New file access records

        Returns:
            Merged file access list (max 20 items, most recent first)
        """
        # Create dict keyed by file path (most recent wins)
        all_files: Dict[str, Dict[str, Any]] = {}

        for file_record in existing + incoming:
            path = file_record.get("path", "")
            if not path:
                continue

            timestamp = file_record.get("accessed_at", "")

            if path not in all_files or timestamp > all_files[path].get(
                "accessed_at", ""
            ):
                all_files[path] = file_record

        # Sort by access time (most recent first)
        sorted_files = sorted(
            all_files.values(), key=lambda x: x.get("accessed_at", ""), reverse=True
        )

        # Return top 20
        return sorted_files[:20]

    @staticmethod
    def _merge_searches(
        existing: List[Dict[str, Any]],
        incoming: List[Dict[str, Any]],
        max_items: int = 20,
    ) -> List[Dict[str, Any]]:
        """Merge search histories without duplicates.

        Removes duplicate searches but preserves recency.

        Args:
            existing: Existing search records
            incoming: New search records
            max_items: Maximum number of searches to keep

        Returns:
            Merged search list (deduplicated, most recent first)
        """
        seen = set()
        merged = []

        # Process in reverse chronological order
        for search in sorted(
            existing + incoming, key=lambda x: x.get("timestamp", ""), reverse=True
        ):
            query = search.get("query", "")
            if not query:
                continue

            if query not in seen:
                seen.add(query)
                merged.append(search)

                if len(merged) >= max_items:
                    break

        return merged

    @staticmethod
    def _merge_decisions(
        existing: List[Dict[str, Any]],
        incoming: List[Dict[str, Any]],
        max_items: int = 10,
    ) -> List[Dict[str, Any]]:
        """Merge decision history.

        Decisions: "Use JWT instead of OAuth", "MongoDB instead of PostgreSQL"

        Args:
            existing: Existing decision records
            incoming: New decision records
            max_items: Maximum number of decisions to keep

        Returns:
            Merged decision list (deduplicated, most recent first)
        """
        # Combine and deduplicate by decision text
        seen_decisions = set()
        merged = []

        for decision in sorted(
            existing + incoming, key=lambda x: x.get("timestamp", ""), reverse=True
        ):
            text = decision.get("decision", "")
            if not text:
                continue

            if text not in seen_decisions:
                seen_decisions.add(text)
                merged.append(decision)

                if len(merged) >= max_items:
                    break

        return merged

    @staticmethod
    def _merge_tool_specific(
        existing: Dict[str, Dict[str, Any]],
        incoming: Dict[str, Dict[str, Any]],
        requesting_tool_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Merge tool-specific settings.

        Different IDE tools have different settings:
        - Cursor: compression_ratio preference, search_mode
        - VSCode: LSP symbol settings, color theme

        Don't override other tools' settings, just add new tool's settings.

        Args:
            existing: Existing tool-specific settings
            incoming: New tool-specific settings
            requesting_tool_id: ID of requesting tool (e.g., "cursor-123")

        Returns:
            Merged tool-specific settings
        """
        merged = existing.copy()

        # Extract tool type from tool_id (e.g., "cursor" from "cursor-123")
        tool_type = (
            requesting_tool_id.split("-")[0]
            if "-" in requesting_tool_id
            else requesting_tool_id
        )

        # Add this tool's settings
        if tool_type not in merged:
            merged[tool_type] = {}

        merged[tool_type].update(incoming.get(tool_type, {}))

        return merged

    @staticmethod
    def _merge_importance_scores(
        existing: Dict[str, float], incoming: Dict[str, float]
    ) -> Dict[str, float]:
        """Merge file importance scores.

        Importance is based on access frequency and recency.
        When merging, take the MAX score (if either tool thinks it's important, it is).

        Args:
            existing: Existing importance scores
            incoming: New importance scores

        Returns:
            Merged importance scores (max value wins)
        """
        merged = existing.copy()

        for file_path, score in incoming.items():
            if file_path not in merged:
                merged[file_path] = score
            else:
                # Take the maximum of both scores
                merged[file_path] = max(merged[file_path], score)

        return merged

    @staticmethod
    def _merge_memories(
        existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge saved memories from multiple IDE tools.

        Combines memories and deduplicates by memory ID.

        Args:
            existing: Existing memory records
            incoming: New memory records

        Returns:
            Merged memory list (deduplicated by ID)
        """
        # Deduplicate by memory ID
        seen_ids = set()
        merged = []

        for memory in existing + incoming:
            memory_id = memory.get("id", "")
            if not memory_id:
                # If no ID, use key as identifier
                memory_id = memory.get("key", "")

            if memory_id and memory_id not in seen_ids:
                seen_ids.add(memory_id)
                merged.append(memory)

        return merged
