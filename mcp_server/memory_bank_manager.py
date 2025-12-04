"""
Memory Bank Manager - Auto-generates structured project context files

Creates /memory-bank/ directory with:
- prd.md (Product requirements extracted from conversations)
- design.md (Architecture, DB schema, APIs from sessions)
- tasks.md (Development tasks and progress)
- context.md (Session-specific updates)
- patterns.md (Coding conventions and patterns learned)

Follows GitHub Copilot Memory Bank pattern for instant AI tool context.
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MemoryBankMetadata(BaseModel):
    """Metadata for memory bank files"""

    last_updated: datetime
    source_sessions: List[str] = Field(default_factory=list)
    total_conversations: int = 0
    total_decisions: int = 0
    total_patterns: int = 0
    version: str = "1.0.0"


class MemoryBankManager:
    """
    Manages Memory Bank - structured project context files auto-generated
    from session history, conversation memory, and procedural memory.

    Provides instant context for AI tools following the Copilot Memory Bank pattern.
    """

    def __init__(
        self,
        workspace_path: str,
        session_manager=None,
        conversation_memory=None,
        procedural_memory=None,
        db_path: Optional[str] = None,
    ):
        """
        Initialize Memory Bank Manager

        Args:
            workspace_path: Absolute path to project workspace
            session_manager: SessionManager instance (optional)
            conversation_memory: ConversationMemory instance (optional)
            procedural_memory: ProceduralMemoryEngine instance (optional)
            db_path: Path to SQLite database for metadata (optional)
        """
        self.workspace_path = Path(workspace_path)
        self.memory_bank_dir = self.workspace_path / "memory-bank"
        self.session_manager = session_manager
        self.conversation_memory = conversation_memory
        self.procedural_memory = procedural_memory

        # Database for tracking metadata
        if db_path:
            self.db_path = db_path
        else:
            self.db_path = str(Path.home() / ".omnimemory" / "memory_bank.db")

        # Ensure directories exist
        self.memory_bank_dir.mkdir(parents=True, exist_ok=True)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize metadata database
        self._ensure_database()

        logger.info(f"MemoryBankManager initialized for {workspace_path}")

    def _ensure_database(self):
        """Ensure metadata database and tables exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create metadata table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_bank_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_path TEXT UNIQUE NOT NULL,
                    last_updated TEXT NOT NULL,
                    source_sessions TEXT,
                    total_conversations INTEGER DEFAULT 0,
                    total_decisions INTEGER DEFAULT 0,
                    total_patterns INTEGER DEFAULT 0,
                    version TEXT DEFAULT '1.0.0',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create file tracking table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_bank_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_hash TEXT,
                    last_updated TEXT NOT NULL,
                    content_size INTEGER DEFAULT 0,
                    UNIQUE(workspace_path, file_name)
                )
            """
            )

            conn.commit()
            conn.close()

            logger.info("Memory Bank database schema ensured")

        except Exception as e:
            logger.error(f"Failed to ensure database: {e}", exc_info=True)
            raise

    async def generate_prd(self) -> str:
        """
        Generate Product Requirements Document (prd.md)
        Extracts product requirements from conversation history

        Returns:
            Path to generated prd.md file
        """
        try:
            logger.info("Generating PRD from conversation history")

            content = []
            content.append("# Product Requirements Document")
            content.append("")
            content.append(
                f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            content.append(f"**Project:** {self.workspace_path.name}")
            content.append("")

            # Extract from conversation memory if available
            if self.conversation_memory:
                try:
                    # Get all sessions
                    conn = self.conversation_memory.conn
                    cursor = conn.cursor()

                    # Find requirement-related conversations
                    cursor.execute(
                        """
                        SELECT DISTINCT session_id, timestamp, content, intent_primary
                        FROM conversation_turns
                        WHERE intent_primary IN ('feature_request', 'requirement', 'planning')
                           OR content LIKE '%requirement%'
                           OR content LIKE '%feature:%'
                           OR content LIKE '%need to%'
                        ORDER BY timestamp DESC
                        LIMIT 50
                    """
                    )

                    requirements = cursor.fetchall()

                    if requirements:
                        content.append("## Requirements")
                        content.append("")

                        # Group by session
                        session_requirements = defaultdict(list)
                        for row in requirements:
                            session_id = row[0]
                            timestamp = row[1]
                            text = row[2]
                            session_requirements[session_id].append((timestamp, text))

                        # Extract requirement statements
                        req_number = 1
                        for session_id, reqs in session_requirements.items():
                            for timestamp, text in reqs:
                                # Extract requirement sentences
                                if any(
                                    keyword in text.lower()
                                    for keyword in [
                                        "need to",
                                        "should",
                                        "must",
                                        "require",
                                        "feature:",
                                    ]
                                ):
                                    # Clean up text
                                    req_text = text.strip()
                                    if len(req_text) > 200:
                                        req_text = req_text[:200] + "..."

                                    content.append(f"### REQ-{req_number:03d}")
                                    content.append(
                                        f"- **Source:** Session {session_id[:8]}"
                                    )
                                    content.append(f"- **Date:** {timestamp}")
                                    content.append(f"- **Description:** {req_text}")
                                    content.append("")
                                    req_number += 1

                except Exception as e:
                    logger.warning(
                        f"Could not extract requirements from conversations: {e}"
                    )
                    content.append("## Requirements")
                    content.append("")
                    content.append("*No requirements extracted yet*")
                    content.append("")
            else:
                content.append("## Requirements")
                content.append("")
                content.append("*Conversation memory not available*")
                content.append("")

            # Extract from session decisions
            if self.session_manager and self.session_manager.current_session:
                decisions = self.session_manager.current_session.context.decisions
                if decisions:
                    content.append("## Decisions")
                    content.append("")
                    for i, decision in enumerate(decisions[-10:], 1):
                        content.append(
                            f"{i}. {decision['decision']} ({decision['timestamp']})"
                        )
                    content.append("")

            # Write to file
            prd_path = self.memory_bank_dir / "prd.md"
            prd_path.write_text("\n".join(content))

            logger.info(f"Generated PRD: {prd_path}")
            return str(prd_path)

        except Exception as e:
            logger.error(f"Failed to generate PRD: {e}", exc_info=True)
            raise

    async def generate_design(self) -> str:
        """
        Generate Design Document (design.md)
        Extracts architecture decisions, DB schema, APIs from sessions

        Returns:
            Path to generated design.md file
        """
        try:
            logger.info("Generating design document from session history")

            content = []
            content.append("# Design Document")
            content.append("")
            content.append(
                f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            content.append(f"**Project:** {self.workspace_path.name}")
            content.append("")

            # Extract architecture decisions from conversations
            if self.conversation_memory:
                try:
                    conn = self.conversation_memory.conn
                    cursor = conn.cursor()

                    # Find architecture-related conversations
                    cursor.execute(
                        """
                        SELECT DISTINCT timestamp, content
                        FROM conversation_turns
                        WHERE content LIKE '%architecture%'
                           OR content LIKE '%design pattern%'
                           OR content LIKE '%database schema%'
                           OR content LIKE '%API%'
                           OR content LIKE '%class%'
                           OR content LIKE '%interface%'
                        ORDER BY timestamp DESC
                        LIMIT 30
                    """
                    )

                    design_items = cursor.fetchall()

                    if design_items:
                        content.append("## Architecture Decisions")
                        content.append("")

                        for i, (timestamp, text) in enumerate(design_items, 1):
                            # Extract relevant sentences
                            sentences = text.split(".")
                            relevant = [
                                s.strip()
                                for s in sentences
                                if any(
                                    kw in s.lower()
                                    for kw in [
                                        "architecture",
                                        "design",
                                        "schema",
                                        "api",
                                        "pattern",
                                    ]
                                )
                            ]

                            if relevant:
                                content.append(f"### Decision {i}")
                                content.append(f"**Date:** {timestamp}")
                                content.append("")
                                for sentence in relevant[:3]:  # Limit to 3 sentences
                                    if sentence:
                                        content.append(f"- {sentence}")
                                content.append("")

                except Exception as e:
                    logger.warning(f"Could not extract design decisions: {e}")

            # Extract from session decisions
            if self.session_manager and self.session_manager.current_session:
                decisions = self.session_manager.current_session.context.decisions
                if decisions:
                    content.append("## Session Decisions")
                    content.append("")
                    for decision in decisions[-10:]:
                        content.append(
                            f"- {decision['decision']} ({decision['timestamp']})"
                        )
                    content.append("")

            # File structure section
            content.append("## File Structure")
            content.append("")
            if self.session_manager and self.session_manager.current_session:
                top_files = sorted(
                    self.session_manager.current_session.context.file_importance_scores.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:20]

                if top_files:
                    content.append("### Key Files")
                    content.append("")
                    for file_path, importance in top_files:
                        content.append(
                            f"- `{file_path}` (importance: {importance:.2f})"
                        )
                    content.append("")

            # Write to file
            design_path = self.memory_bank_dir / "design.md"
            design_path.write_text("\n".join(content))

            logger.info(f"Generated design document: {design_path}")
            return str(design_path)

        except Exception as e:
            logger.error(f"Failed to generate design document: {e}", exc_info=True)
            raise

    async def generate_tasks(self) -> str:
        """
        Generate Tasks Document (tasks.md)
        Extracts TODOs and task progress from conversations

        Returns:
            Path to generated tasks.md file
        """
        try:
            logger.info("Generating tasks document")

            content = []
            content.append("# Development Tasks")
            content.append("")
            content.append(
                f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            content.append(f"**Project:** {self.workspace_path.name}")
            content.append("")

            # Extract TODOs from conversations
            if self.conversation_memory:
                try:
                    conn = self.conversation_memory.conn
                    cursor = conn.cursor()

                    # Find task-related conversations
                    cursor.execute(
                        """
                        SELECT DISTINCT timestamp, content
                        FROM conversation_turns
                        WHERE content LIKE '%TODO%'
                           OR content LIKE '%task:%'
                           OR content LIKE '%implement%'
                           OR content LIKE '%fix%'
                           OR content LIKE '% [ ]%'
                           OR content LIKE '% [x]%'
                        ORDER BY timestamp DESC
                        LIMIT 50
                    """
                    )

                    task_items = cursor.fetchall()

                    if task_items:
                        content.append("## Pending Tasks")
                        content.append("")

                        task_number = 1
                        for timestamp, text in task_items:
                            # Extract task lines
                            lines = text.split("\n")
                            for line in lines:
                                line = line.strip()
                                # Check for task patterns
                                if any(
                                    pattern in line
                                    for pattern in [
                                        "TODO",
                                        "- [ ]",
                                        "task:",
                                        "implement",
                                    ]
                                ):
                                    # Clean up
                                    task_text = (
                                        line.replace("TODO:", "")
                                        .replace("- [ ]", "")
                                        .strip()
                                    )
                                    if len(task_text) > 200:
                                        task_text = task_text[:200] + "..."

                                    if (
                                        task_text and len(task_text) > 10
                                    ):  # Meaningful tasks
                                        content.append(
                                            f"- [ ] **Task {task_number}:** {task_text}"
                                        )
                                        content.append(f"  - *Added: {timestamp}*")
                                        content.append("")
                                        task_number += 1

                        content.append("## Completed Tasks")
                        content.append("")

                        # Find completed tasks
                        cursor.execute(
                            """
                            SELECT DISTINCT timestamp, content
                            FROM conversation_turns
                            WHERE content LIKE '% [x]%'
                               OR content LIKE '%completed%'
                               OR content LIKE '%done%'
                            ORDER BY timestamp DESC
                            LIMIT 20
                        """
                        )

                        completed_items = cursor.fetchall()
                        for timestamp, text in completed_items:
                            lines = text.split("\n")
                            for line in lines:
                                if "[x]" in line or "completed:" in line.lower():
                                    task_text = (
                                        line.replace("- [x]", "")
                                        .replace("completed:", "")
                                        .strip()
                                    )
                                    if len(task_text) > 200:
                                        task_text = task_text[:200] + "..."

                                    if task_text and len(task_text) > 10:
                                        content.append(f"- [x] {task_text}")
                                        content.append(f"  - *Completed: {timestamp}*")
                                        content.append("")

                except Exception as e:
                    logger.warning(f"Could not extract tasks from conversations: {e}")
                    content.append("*No tasks extracted yet*")
                    content.append("")
            else:
                content.append("*Conversation memory not available*")
                content.append("")

            # Write to file
            tasks_path = self.memory_bank_dir / "tasks.md"
            tasks_path.write_text("\n".join(content))

            logger.info(f"Generated tasks document: {tasks_path}")
            return str(tasks_path)

        except Exception as e:
            logger.error(f"Failed to generate tasks document: {e}", exc_info=True)
            raise

    async def generate_context(self) -> str:
        """
        Generate Current Context Document (context.md)
        Session-specific updates and current working context

        Returns:
            Path to generated context.md file
        """
        try:
            logger.info("Generating context document from current session")

            content = []
            content.append("# Current Session Context")
            content.append("")
            content.append(
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            content.append(f"**Project:** {self.workspace_path.name}")
            content.append("")

            if self.session_manager and self.session_manager.current_session:
                session = self.session_manager.current_session

                content.append("## Session Info")
                content.append("")
                content.append(f"- **Session ID:** {session.session_id}")
                content.append(f"- **Created:** {session.created_at}")
                content.append(f"- **Last Activity:** {session.last_activity}")
                content.append("")

                # Recent files
                content.append("## Recently Accessed Files")
                content.append("")
                recent_files = session.context.files_accessed[-10:]
                if recent_files:
                    for file_info in reversed(recent_files):
                        content.append(
                            f"- `{file_info['path']}` (accessed: {file_info['accessed_at']})"
                        )
                    content.append("")
                else:
                    content.append("*No files accessed yet*")
                    content.append("")

                # Recent searches
                content.append("## Recent Searches")
                content.append("")
                recent_searches = session.context.recent_searches[-10:]
                if recent_searches:
                    for search in reversed(recent_searches):
                        content.append(f"- `{search['query']}` ({search['timestamp']})")
                    content.append("")
                else:
                    content.append("*No searches yet*")
                    content.append("")

                # Saved memories
                content.append("## Saved Memories")
                content.append("")
                if session.context.saved_memories:
                    for memory in session.context.saved_memories[-5:]:
                        content.append(
                            f"- **{memory['key']}** (ID: {memory['id']}, saved: {memory['timestamp']})"
                        )
                    content.append("")
                else:
                    content.append("*No memories saved yet*")
                    content.append("")

                # Session metrics
                if session.metrics:
                    content.append("## Session Metrics")
                    content.append("")
                    for key, value in session.metrics.items():
                        content.append(f"- **{key}:** {value}")
                    content.append("")

            else:
                content.append("*No active session*")
                content.append("")

            # Write to file
            context_path = self.memory_bank_dir / "context.md"
            context_path.write_text("\n".join(content))

            logger.info(f"Generated context document: {context_path}")
            return str(context_path)

        except Exception as e:
            logger.error(f"Failed to generate context document: {e}", exc_info=True)
            raise

    async def generate_patterns(self) -> str:
        """
        Generate Coding Patterns Document (patterns.md)
        Coding conventions and patterns learned from workflow analysis

        Returns:
            Path to generated patterns.md file
        """
        try:
            logger.info("Generating patterns document from procedural memory")

            content = []
            content.append("# Coding Patterns & Conventions")
            content.append("")
            content.append(
                f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            content.append(f"**Project:** {self.workspace_path.name}")
            content.append("")

            # Extract from procedural memory if available
            if self.procedural_memory:
                try:
                    # Get top patterns by success rate
                    patterns = sorted(
                        self.procedural_memory.patterns.values(),
                        key=lambda p: p.confidence,
                        reverse=True,
                    )[:20]

                    if patterns:
                        content.append("## Learned Workflow Patterns")
                        content.append("")

                        for i, pattern in enumerate(patterns, 1):
                            content.append(f"### Pattern {i}")
                            content.append(
                                f"**Confidence:** {pattern.confidence:.2%} "
                                f"({pattern.success_count} successes, {pattern.failure_count} failures)"
                            )
                            content.append("")
                            content.append("**Command Sequence:**")
                            content.append("```")
                            for cmd in pattern.command_sequence:
                                content.append(f"  {cmd}")
                            content.append("```")
                            content.append("")

                    # Get common command transitions from graph
                    if (
                        hasattr(self.procedural_memory, "workflow_graph")
                        and self.procedural_memory.workflow_graph
                    ):
                        content.append("## Common Workflow Transitions")
                        content.append("")

                        # Get edges with highest weights
                        edges = []
                        for u, v, data in self.procedural_memory.workflow_graph.edges(
                            data=True
                        ):
                            weight = data.get("weight", 0)
                            success = data.get("success", 0)
                            success_rate = success / weight if weight > 0 else 0
                            edges.append((u, v, weight, success_rate))

                        # Sort by weight * success_rate
                        edges.sort(key=lambda x: x[2] * x[3], reverse=True)

                        for u, v, weight, success_rate in edges[:15]:
                            content.append(
                                f"- `{u}` → `{v}` "
                                f"(used {weight}x, success rate: {success_rate:.1%})"
                            )

                        content.append("")

                except Exception as e:
                    logger.warning(
                        f"Could not extract patterns from procedural memory: {e}"
                    )
                    content.append("*No patterns learned yet*")
                    content.append("")
            else:
                content.append("*Procedural memory not available*")
                content.append("")

            # Add file-based patterns
            content.append("## File Organization Patterns")
            content.append("")

            if self.session_manager and self.session_manager.current_session:
                # Analyze file paths to detect organization patterns
                files = self.session_manager.current_session.context.files_accessed
                if files:
                    # Extract directory patterns
                    directories = defaultdict(int)
                    extensions = defaultdict(int)

                    for file_info in files:
                        path = Path(file_info["path"])
                        if path.parent:
                            directories[str(path.parent)] += 1
                        if path.suffix:
                            extensions[path.suffix] += 1

                    # Top directories
                    if directories:
                        content.append("### Most Active Directories")
                        for directory, count in sorted(
                            directories.items(), key=lambda x: x[1], reverse=True
                        )[:10]:
                            content.append(f"- `{directory}` ({count} files)")
                        content.append("")

                    # File type distribution
                    if extensions:
                        content.append("### File Types")
                        for ext, count in sorted(
                            extensions.items(), key=lambda x: x[1], reverse=True
                        ):
                            content.append(f"- `{ext}` ({count} files)")
                        content.append("")

            # Write to file
            patterns_path = self.memory_bank_dir / "patterns.md"
            patterns_path.write_text("\n".join(content))

            logger.info(f"Generated patterns document: {patterns_path}")
            return str(patterns_path)

        except Exception as e:
            logger.error(f"Failed to generate patterns document: {e}", exc_info=True)
            raise

    async def sync_to_disk(self) -> Dict[str, str]:
        """
        Synchronize all Memory Bank files to disk

        Returns:
            Dictionary mapping file names to paths
        """
        try:
            logger.info("Syncing Memory Bank to disk")

            results = {}

            # Generate all files
            results["prd"] = await self.generate_prd()
            results["design"] = await self.generate_design()
            results["tasks"] = await self.generate_tasks()
            results["context"] = await self.generate_context()
            results["patterns"] = await self.generate_patterns()

            # Generate metadata file
            metadata = MemoryBankMetadata(
                last_updated=datetime.now(),
                source_sessions=[
                    self.session_manager.current_session.session_id
                    if self.session_manager and self.session_manager.current_session
                    else "unknown"
                ],
                total_conversations=await self._count_conversations(),
                total_decisions=await self._count_decisions(),
                total_patterns=await self._count_patterns(),
            )

            meta_path = self.memory_bank_dir / ".meta.json"
            meta_path.write_text(metadata.model_dump_json(indent=2))
            results["metadata"] = str(meta_path)

            # Update database
            self._update_metadata(metadata)

            logger.info(
                f"Synced {len(results)} Memory Bank files to {self.memory_bank_dir}"
            )

            return results

        except Exception as e:
            logger.error(f"Failed to sync Memory Bank: {e}", exc_info=True)
            raise

    async def sync_from_sessions(self) -> None:
        """
        Aggregate data from all session sources
        """
        logger.info("Syncing from all session sources")

        # This would aggregate from multiple sessions if needed
        # For now, we use the current session and available services
        await self.sync_to_disk()

    async def export_copilot_instructions(self) -> str:
        """
        Export Memory Bank as GitHub Copilot instructions format

        Returns:
            Path to generated .github/copilot-instructions.md
        """
        try:
            logger.info("Exporting to GitHub Copilot instructions format")

            github_dir = self.workspace_path / ".github"
            github_dir.mkdir(exist_ok=True)

            content = []
            content.append("# GitHub Copilot Instructions")
            content.append("")
            content.append(f"Project: {self.workspace_path.name}")
            content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            content.append("")

            # Read and include key memory bank files
            for file_name in ["prd.md", "design.md", "patterns.md"]:
                file_path = self.memory_bank_dir / file_name
                if file_path.exists():
                    content.append(f"## {file_name.replace('.md', '').upper()}")
                    content.append("")
                    content.append(file_path.read_text())
                    content.append("")

            copilot_path = github_dir / "copilot-instructions.md"
            copilot_path.write_text("\n".join(content))

            logger.info(f"Exported Copilot instructions to {copilot_path}")
            return str(copilot_path)

        except Exception as e:
            logger.error(f"Failed to export Copilot instructions: {e}", exc_info=True)
            raise

    async def _count_conversations(self) -> int:
        """Count total conversation turns"""
        if not self.conversation_memory:
            return 0

        try:
            cursor = self.conversation_memory.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM conversation_turns")
            result = cursor.fetchone()
            return result[0] if result else 0
        except Exception as e:
            logger.warning(f"Could not count conversations: {e}")
            return 0

    async def _count_decisions(self) -> int:
        """Count total decisions"""
        count = 0

        if self.session_manager and self.session_manager.current_session:
            count += len(self.session_manager.current_session.context.decisions)

        return count

    async def _count_patterns(self) -> int:
        """Count learned patterns"""
        if not self.procedural_memory:
            return 0

        return len(self.procedural_memory.patterns)

    def _update_metadata(self, metadata: MemoryBankMetadata):
        """Update metadata in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO memory_bank_metadata
                (workspace_path, last_updated, source_sessions, total_conversations,
                 total_decisions, total_patterns, version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    str(self.workspace_path),
                    metadata.last_updated.isoformat(),
                    json.dumps(metadata.source_sessions),
                    metadata.total_conversations,
                    metadata.total_decisions,
                    metadata.total_patterns,
                    metadata.version,
                ),
            )

            conn.commit()
            conn.close()

            logger.debug(f"Updated metadata for {self.workspace_path}")

        except Exception as e:
            logger.error(f"Failed to update metadata: {e}", exc_info=True)

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up MemoryBankManager")
        # No resources to cleanup currently
        pass
