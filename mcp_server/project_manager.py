"""
Project Manager for per-project context isolation.

Manages projects, project settings, and project-specific memories.
Each project represents a workspace/codebase with isolated context.
"""

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field


# ============================================================================
# Pydantic Models
# ============================================================================


class Project(BaseModel):
    """Project data model"""

    project_id: str  # Hash of workspace_path
    workspace_path: str
    project_name: str
    language: Optional[str] = None  # Detected primary language
    framework: Optional[str] = None  # Detected framework
    created_at: datetime
    last_accessed: datetime
    total_sessions: int = 0
    settings: Dict = Field(default_factory=dict)

    class Config:
        from_attributes = True


class ProjectMemory(BaseModel):
    """Project-specific memory"""

    memory_id: str
    project_id: str
    memory_key: str  # e.g., "architecture", "api_endpoints"
    memory_value: str  # Full content
    compressed_value: Optional[str] = None  # Compressed version
    created_at: datetime
    last_accessed: datetime
    accessed_count: int = 0
    ttl_seconds: Optional[int] = None  # Time to live
    metadata: Dict = Field(default_factory=dict)

    class Config:
        from_attributes = True


# ============================================================================
# ProjectManager Class
# ============================================================================


class ProjectManager:
    """
    Manages project settings and per-project context.

    Responsibilities:
    - Create and track projects (workspaces)
    - Manage project-specific settings
    - Store and retrieve project memories
    - Detect project language/framework
    - Isolate context per project
    """

    def __init__(self, db_path: str):
        """
        Initialize ProjectManager.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._ensure_database()

    # ================== PROJECT OPERATIONS ==================

    def create_project(
        self, workspace_path: str, project_id: Optional[str] = None
    ) -> Project:
        """
        Create new project record.

        Args:
            workspace_path: Absolute path to workspace
            project_id: Optional project ID (generated if not provided)

        Returns:
            Created project
        """
        try:
            if not project_id:
                project_id = self._hash_workspace_path(workspace_path)

            # Detect project name from path
            project_name = Path(workspace_path).name

            # Detect language and framework
            language, framework = self._detect_stack(workspace_path)

            project = Project(
                project_id=project_id,
                workspace_path=workspace_path,
                project_name=project_name,
                language=language,
                framework=framework,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                settings={
                    "auto_save_enabled": True,
                    "auto_save_interval_seconds": 300,
                    "memory_limit_mb": 100,
                    "compress_context": True,
                },
            )

            self._save_project(project)
            return project

        except Exception as e:
            raise RuntimeError(f"Failed to create project: {e}") from e

    def get_project(self, project_id: str) -> Optional[Project]:
        """
        Get project by ID.

        Args:
            project_id: Project identifier

        Returns:
            Project if found, None otherwise
        """
        try:
            return self._load_project(project_id)
        except Exception as e:
            print(f"Error loading project {project_id}: {e}")
            return None

    def get_or_create_project(self, workspace_path: str) -> Project:
        """
        Get existing project or create new one.

        Args:
            workspace_path: Absolute path to workspace

        Returns:
            Project (existing or newly created)
        """
        project_id = self._hash_workspace_path(workspace_path)
        project = self.get_project(project_id)

        if project:
            # Update last accessed
            project.last_accessed = datetime.utcnow()
            self._save_project(project)
            return project
        else:
            return self.create_project(workspace_path, project_id)

    def update_project_stats(self, project_id: str, session_created: bool = False):
        """
        Update project statistics.

        Args:
            project_id: Project identifier
            session_created: Whether a new session was created
        """
        try:
            project = self.get_project(project_id)
            if not project:
                return

            project.last_accessed = datetime.utcnow()

            if session_created:
                project.total_sessions += 1

            self._save_project(project)

        except Exception as e:
            print(f"Error updating project stats: {e}")

    # ================== PROJECT CONTEXT (MEMORIES) ==================

    def get_project_memories(
        self, project_id: str, limit: int = 20
    ) -> List[ProjectMemory]:
        """
        Get all memories for project.

        Args:
            project_id: Project identifier
            limit: Maximum number of memories to return

        Returns:
            List of project memories (sorted by last accessed)
        """
        try:
            return self._query_memories(project_id, limit=limit)
        except Exception as e:
            print(f"Error querying memories: {e}")
            return []

    def save_project_memory(
        self,
        project_id: str,
        key: str,
        value: str,
        metadata: Optional[Dict] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """
        Save memory associated with project.

        Args:
            project_id: Project identifier
            key: Memory key (e.g., "architecture")
            value: Memory content
            metadata: Optional metadata dictionary
            ttl_seconds: Optional time-to-live in seconds

        Returns:
            Memory ID
        """
        try:
            memory_id = f"mem_{uuid4().hex[:12]}"

            memory = ProjectMemory(
                memory_id=memory_id,
                project_id=project_id,
                memory_key=key,
                memory_value=value,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                ttl_seconds=ttl_seconds,
                metadata=metadata or {},
            )

            self._save_memory(memory)
            return memory_id

        except Exception as e:
            raise RuntimeError(f"Failed to save project memory: {e}") from e

    def get_project_memory(self, project_id: str, key: str) -> Optional[ProjectMemory]:
        """
        Get specific memory by key.

        Args:
            project_id: Project identifier
            key: Memory key

        Returns:
            ProjectMemory if found, None otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM project_memories
                WHERE project_id = ? AND memory_key = ?
                ORDER BY last_accessed DESC
                LIMIT 1
            """,
                (project_id, key),
            )

            row = cursor.fetchone()
            conn.close()

            if not row:
                return None

            # Update accessed count
            memory = self._row_to_memory(row)
            memory.last_accessed = datetime.utcnow()
            memory.accessed_count += 1
            self._save_memory(memory)

            return memory

        except Exception as e:
            print(f"Error getting project memory: {e}")
            return None

    def delete_expired_memories(self, project_id: str):
        """
        Delete expired memories for project.

        Args:
            project_id: Project identifier
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                DELETE FROM project_memories
                WHERE project_id = ?
                  AND ttl_seconds IS NOT NULL
                  AND datetime(created_at, '+' || ttl_seconds || ' seconds') < datetime('now')
            """,
                (project_id,),
            )

            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()

            if deleted_count > 0:
                print(
                    f"✓ Deleted {deleted_count} expired memories for project {project_id}"
                )

        except Exception as e:
            print(f"Error deleting expired memories: {e}")

    # ================== PROJECT SETTINGS ==================

    def update_project_settings(self, project_id: str, settings: Dict):
        """
        Update project settings.

        Args:
            project_id: Project identifier
            settings: Settings dictionary to merge
        """
        try:
            project = self.get_project(project_id)
            if not project:
                return

            # Merge settings
            project.settings.update(settings)
            self._save_project(project)

        except Exception as e:
            print(f"Error updating project settings: {e}")

    def get_project_settings(self, project_id: str) -> Dict:
        """
        Get project settings.

        Args:
            project_id: Project identifier

        Returns:
            Settings dictionary
        """
        project = self.get_project(project_id)
        return project.settings if project else {}

    # ================== HELPER METHODS ==================

    def _hash_workspace_path(self, workspace_path: str) -> str:
        """Generate project_id from workspace_path."""
        return hashlib.sha256(workspace_path.encode()).hexdigest()[:16]

    def _detect_stack(self, workspace_path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect language and framework from workspace.

        Args:
            workspace_path: Path to workspace

        Returns:
            Tuple of (language, framework)
        """
        try:
            path = Path(workspace_path)

            # Detect by marker files
            if (path / "package.json").exists():
                # Check for framework in package.json
                try:
                    pkg_file = path / "package.json"
                    pkg = json.loads(pkg_file.read_text())
                    deps = {
                        **pkg.get("dependencies", {}),
                        **pkg.get("devDependencies", {}),
                    }

                    if "next" in deps:
                        return ("javascript", "nextjs")
                    elif "react" in deps:
                        return ("javascript", "react")
                    elif "vue" in deps:
                        return ("javascript", "vue")
                    elif "@angular/core" in deps:
                        return ("javascript", "angular")
                    else:
                        return ("javascript", None)
                except Exception:
                    return ("javascript", None)

            elif (path / "pyproject.toml").exists() or (
                path / "requirements.txt"
            ).exists():
                # Check for Django/Flask
                if (path / "manage.py").exists():
                    return ("python", "django")
                elif (
                    any((path / "app").glob("*.py"))
                    if (path / "app").exists()
                    else False
                ):
                    return ("python", "flask")
                else:
                    return ("python", None)

            elif (path / "go.mod").exists():
                return ("go", None)

            elif (path / "Cargo.toml").exists():
                return ("rust", None)

            elif (path / "pom.xml").exists():
                return ("java", "maven")

            elif (path / "build.gradle").exists() or (
                path / "build.gradle.kts"
            ).exists():
                return ("java", "gradle")

            return (None, None)

        except Exception as e:
            print(f"Error detecting stack: {e}")
            return (None, None)

    def _ensure_database(self):
        """Ensure database and tables exist."""
        try:
            # Create parent directory if it doesn't exist
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create projects table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    workspace_path TEXT UNIQUE NOT NULL,
                    project_name TEXT,
                    language TEXT,
                    framework TEXT,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT,
                    total_sessions INTEGER DEFAULT 0,
                    settings_json TEXT
                )
            """
            )

            # Create project_memories table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS project_memories (
                    memory_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    memory_value TEXT,
                    compressed_value TEXT,
                    created_at TEXT NOT NULL,
                    accessed_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    ttl_seconds INTEGER,
                    metadata_json TEXT,
                    FOREIGN KEY (project_id) REFERENCES projects(project_id)
                )
            """
            )

            # Create indexes
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_project_memories_project_id
                ON project_memories(project_id)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_project_memories_key
                ON project_memories(memory_key)
            """
            )

            conn.commit()
            conn.close()

        except Exception as e:
            raise RuntimeError(f"Failed to initialize database: {e}") from e

    def _save_project(self, project: Project):
        """Save project to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO projects (
                    project_id, workspace_path, project_name,
                    language, framework,
                    created_at, last_accessed, total_sessions,
                    settings_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    project.project_id,
                    project.workspace_path,
                    project.project_name,
                    project.language,
                    project.framework,
                    project.created_at.isoformat(),
                    project.last_accessed.isoformat(),
                    project.total_sessions,
                    json.dumps(project.settings),
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            raise RuntimeError(f"Failed to save project: {e}") from e

    def _load_project(self, project_id: str) -> Optional[Project]:
        """Load project from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM projects WHERE project_id = ?
            """,
                (project_id,),
            )

            row = cursor.fetchone()
            conn.close()

            if not row:
                return None

            return Project(
                project_id=row["project_id"],
                workspace_path=row["workspace_path"],
                project_name=row["project_name"],
                language=row["language"],
                framework=row["framework"],
                created_at=datetime.fromisoformat(row["created_at"]),
                last_accessed=datetime.fromisoformat(row["last_accessed"]),
                total_sessions=row["total_sessions"],
                settings=json.loads(row["settings_json"])
                if row["settings_json"]
                else {},
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load project: {e}") from e

    def _save_memory(self, memory: ProjectMemory):
        """Save memory to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO project_memories (
                    memory_id, project_id, memory_key, memory_value,
                    compressed_value, created_at, last_accessed,
                    accessed_count, ttl_seconds, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    memory.memory_id,
                    memory.project_id,
                    memory.memory_key,
                    memory.memory_value,
                    memory.compressed_value,
                    memory.created_at.isoformat(),
                    memory.last_accessed.isoformat(),
                    memory.accessed_count,
                    memory.ttl_seconds,
                    json.dumps(memory.metadata),
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            raise RuntimeError(f"Failed to save memory: {e}") from e

    def _query_memories(self, project_id: str, limit: int = 20) -> List[ProjectMemory]:
        """Query memories for project."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM project_memories
                WHERE project_id = ?
                ORDER BY last_accessed DESC
                LIMIT ?
            """,
                (project_id, limit),
            )

            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_memory(row) for row in rows]

        except Exception as e:
            raise RuntimeError(f"Failed to query memories: {e}") from e

    def _row_to_memory(self, row: sqlite3.Row) -> ProjectMemory:
        """Convert database row to ProjectMemory."""
        return ProjectMemory(
            memory_id=row["memory_id"],
            project_id=row["project_id"],
            memory_key=row["memory_key"],
            memory_value=row["memory_value"],
            compressed_value=row["compressed_value"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed=datetime.fromisoformat(row["last_accessed"]),
            accessed_count=row["accessed_count"],
            ttl_seconds=row["ttl_seconds"],
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        )
