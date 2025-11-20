"""Safe file operations for configuration management."""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def safe_read_json(file_path: Path) -> Dict[str, Any]:
    """
    Safely read JSON file with error handling.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary with file contents, or empty dict if file doesn't exist

    Raises:
        ValueError: If file exists but contains invalid JSON
    """
    if not file_path.exists():
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")


def safe_write_json(file_path: Path, data: Dict[str, Any], indent: int = 2) -> None:
    """
    Safely write JSON file with atomic operation.

    Args:
        file_path: Path to JSON file
        data: Data to write
        indent: JSON indentation level
    """
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file first (atomic operation)
    temp_path = file_path.with_suffix(".tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            f.write("\n")  # Add trailing newline

        # Atomic rename
        temp_path.replace(file_path)
    except Exception as e:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise e


def create_backup(file_path: Path) -> Optional[Path]:
    """
    Create timestamped backup of configuration file.

    Args:
        file_path: Path to file to backup

    Returns:
        Path to backup file, or None if file doesn't exist
    """
    if not file_path.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.with_suffix(f".backup_{timestamp}{file_path.suffix}")

    shutil.copy2(file_path, backup_path)
    return backup_path


def restore_from_backup(backup_path: Path, target_path: Path) -> bool:
    """
    Restore configuration from backup.

    Args:
        backup_path: Path to backup file
        target_path: Path to restore to

    Returns:
        True if restore successful, False otherwise
    """
    if not backup_path.exists():
        return False

    try:
        shutil.copy2(backup_path, target_path)
        return True
    except Exception:
        return False
