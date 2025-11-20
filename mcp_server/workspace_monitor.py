"""
Workspace Monitor for Automatic Project Switching
Detects when user changes directory/project and loads appropriate context
"""

import os
import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Optional, Callable, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class WorkspaceMonitor:
    """
    Monitors workspace/directory changes and triggers context switching

    Features:
    - Detects when user changes directory
    - Identifies project type (git repo, npm project, Python project, etc.)
    - Triggers context loading for new project
    - Notifies about last session in that project
    """

    def __init__(
        self,
        check_interval: int = 5,  # Check every 5 seconds
        on_switch_callback: Optional[Callable] = None,
    ):
        self.check_interval = check_interval
        self.current_workspace = os.getcwd()
        self.current_project_id = self._detect_project_id(self.current_workspace)
        self.on_switch_callback = on_switch_callback
        self.running = False
        self.monitor_task = None

    def start(self):
        """Start monitoring workspace changes"""
        if not self.running:
            self.running = True

            # Get running event loop (safe in async context)
            try:
                loop = asyncio.get_running_loop()
                self.monitor_task = loop.create_task(self._monitor_loop())
                logger.info(
                    f"✓ Workspace monitor started (checking every {self.check_interval}s)"
                )
            except RuntimeError:
                # No running loop - this shouldn't happen if called from tool
                logger.error("Cannot start workspace monitor: no running event loop")
                self.running = False
                raise

    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
        logger.info("✓ Workspace monitor stopped")

    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                await asyncio.sleep(self.check_interval)

                new_workspace = os.getcwd()

                if new_workspace != self.current_workspace:
                    # Workspace changed!
                    new_project_id = self._detect_project_id(new_workspace)

                    if new_project_id != self.current_project_id:
                        # Project switched!
                        await self._handle_project_switch(
                            old_project=self.current_project_id,
                            new_project=new_project_id,
                            new_workspace=new_workspace,
                        )

                        self.current_project_id = new_project_id

                    self.current_workspace = new_workspace

            except Exception as e:
                logger.error(f"Workspace monitor error: {e}")

    async def _handle_project_switch(
        self, old_project: str, new_project: str, new_workspace: str
    ):
        """Handle project switch event"""
        project_info = self._get_project_info(new_workspace)

        logger.info(f"🔄 Project switched: {old_project} → {new_project}")
        logger.info(f"   Project: {project_info['name']}")
        logger.info(f"   Type: {project_info['type']}")
        logger.info(f"   Path: {new_workspace}")

        # Trigger callback if provided
        if self.on_switch_callback:
            try:
                await self.on_switch_callback(
                    {
                        "old_project": old_project,
                        "new_project": new_project,
                        "project_info": project_info,
                        "workspace_path": new_workspace,
                    }
                )
            except Exception as e:
                logger.error(f"Project switch callback error: {e}")

    def _detect_project_id(self, workspace_path: str) -> str:
        """
        Detect stable project ID from workspace

        Priority:
        1. Git repository (use git root hash)
        2. Node project (package.json)
        3. Python project (setup.py, pyproject.toml)
        4. Directory hash (fallback)
        """
        path = Path(workspace_path)

        # Try git repo
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=path,
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0:
                git_root = result.stdout.strip()
                return f"repo_{hashlib.sha256(git_root.encode()).hexdigest()[:16]}"
        except:
            pass

        # Try Node project
        if (path / "package.json").exists():
            return f"node_{hashlib.sha256(str(path).encode()).hexdigest()[:16]}"

        # Try Python project
        for file in ["setup.py", "pyproject.toml", "requirements.txt"]:
            if (path / file).exists():
                return f"python_{hashlib.sha256(str(path).encode()).hexdigest()[:16]}"

        # Fallback to directory hash
        return f"dir_{hashlib.sha256(str(path).encode()).hexdigest()[:16]}"

    def _get_project_info(self, workspace_path: str) -> Dict[str, str]:
        """Get detailed project information"""
        path = Path(workspace_path)

        info = {"name": path.name, "type": "unknown", "path": str(path)}

        # Detect type
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=path,
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0:
                info["type"] = "git repository"

                # Try to get repo name
                result = subprocess.run(
                    ["git", "config", "--get", "remote.origin.url"],
                    cwd=path,
                    capture_output=True,
                    text=True,
                    timeout=1,
                )
                if result.returncode == 0 and "/" in result.stdout:
                    info["name"] = (
                        result.stdout.strip().split("/")[-1].replace(".git", "")
                    )
        except:
            pass

        if info["type"] == "unknown":
            if (path / "package.json").exists():
                info["type"] = "Node.js project"
            elif (path / "pyproject.toml").exists():
                info["type"] = "Python project"
            elif (path / "Cargo.toml").exists():
                info["type"] = "Rust project"

        return info

    def get_current_project(self) -> Dict[str, str]:
        """Get current project information"""
        return {
            "project_id": self.current_project_id,
            "workspace": self.current_workspace,
            "info": self._get_project_info(self.current_workspace),
        }
