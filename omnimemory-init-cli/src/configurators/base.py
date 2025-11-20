"""Base configurator class for tool integrations."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..utils.file_ops import create_backup


class BaseConfigurator(ABC):
    """Base class for tool configurators."""

    def __init__(self, api_key: str, api_url: str, user_id: Optional[str] = None):
        """
        Initialize configurator.

        Args:
            api_key: OmniMemory API key
            api_url: OmniMemory API base URL
            user_id: User ID for OmniMemory (defaults to "default_user")
        """
        self.api_key = api_key
        self.api_url = api_url
        self.user_id = user_id or "default_user"
        self._changes: List[str] = []

    @abstractmethod
    def get_config_path(self) -> Path:
        """
        Get path to tool's configuration file.

        Returns:
            Path to config file
        """
        pass

    @abstractmethod
    def is_installed(self) -> bool:
        """
        Check if tool is installed on the system.

        Returns:
            True if tool is installed, False otherwise
        """
        pass

    @abstractmethod
    def configure(self, dry_run: bool = False) -> List[str]:
        """
        Apply OmniMemory configuration to the tool.

        Args:
            dry_run: If True, show what would be changed without modifying files

        Returns:
            List of changes made (or that would be made in dry run)
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for automatic memory usage.

        Returns:
            System prompt text
        """
        pass

    def backup_config(self) -> Optional[Path]:
        """
        Backup existing configuration file.

        Returns:
            Path to backup file, or None if no config exists
        """
        config_path = self.get_config_path()
        if not config_path.exists():
            return None

        return create_backup(config_path)

    def get_status(self) -> Dict[str, Any]:
        """
        Get current configuration status.

        Returns:
            Dictionary with status information
        """
        config_path = self.get_config_path()

        status = {
            "installed": self.is_installed(),
            "config_exists": config_path.exists(),
            "config_path": str(config_path),
            "omnimemory_configured": False,
        }

        # Check if OmniMemory is already configured
        if config_path.exists():
            status["omnimemory_configured"] = self._is_omnimemory_configured()

        return status

    def _is_omnimemory_configured(self) -> bool:
        """
        Check if OmniMemory is already configured for this tool.

        Returns:
            True if configured, False otherwise
        """
        # This should be overridden by subclasses
        return False

    def remove_configuration(self, dry_run: bool = False) -> List[str]:
        """
        Remove OmniMemory configuration from the tool.

        Args:
            dry_run: If True, show what would be removed without modifying files

        Returns:
            List of changes made (or that would be made in dry run)
        """
        # This should be overridden by subclasses
        return ["Remove configuration not implemented for this tool"]

    def _record_change(self, change: str) -> None:
        """Record a configuration change."""
        self._changes.append(change)

    def _get_changes(self) -> List[str]:
        """Get list of recorded changes."""
        return self._changes.copy()

    def _clear_changes(self) -> None:
        """Clear recorded changes."""
        self._changes.clear()
