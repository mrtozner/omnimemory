"""
Result Cleanup Daemon for OmniMemory
Automatically removes expired cached results to free disk space

Features:
- Runs every 6 hours by default
- Deletes results older than TTL (default 7 days)
- Tracks cleanup metrics
- Reports to metrics service
- Safe file operations with error handling
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CHECK_INTERVAL = 6 * 3600  # 6 hours in seconds
DEFAULT_CACHE_DIR = "~/.omnimemory/cached_results"
RESULT_FILE_PATTERN = "*.result.json.lz4"
METADATA_SUFFIX = ".metadata.json"


class ResultCleanupDaemon:
    """
    Background daemon for cleaning up expired cached results.

    Features:
    - Runs every 6 hours (configurable)
    - Deletes results older than TTL
    - Tracks cleanup metrics
    - Reports to metrics service
    - Safe file operations with error recovery
    """

    def __init__(
        self,
        result_store: Optional[Any] = None,
        check_interval: int = DEFAULT_CHECK_INTERVAL,
        cache_dir: str = DEFAULT_CACHE_DIR,
        metrics_url: str = "http://localhost:8003",
    ):
        """
        Initialize cleanup daemon

        Args:
            result_store: ResultStore instance (optional, can be None)
            check_interval: Seconds between cleanup runs (default 6 hours)
            cache_dir: Directory containing cached results
            metrics_url: URL of metrics service
        """
        self.result_store = result_store
        self.check_interval = check_interval
        self.cache_dir = Path(cache_dir).expanduser()
        self.metrics_url = metrics_url
        self.running = False
        self.cleanup_task = None

        # Statistics
        self.total_deleted = 0
        self.total_freed_bytes = 0
        self.total_errors = 0
        self.cleanup_count = 0

        logger.info(
            f"Initialized ResultCleanupDaemon (interval={check_interval}s, "
            f"cache_dir={self.cache_dir})"
        )

    async def start(self):
        """Start cleanup daemon in background."""
        if not self.running:
            self.running = True

            # Get running event loop (safe in async context)
            try:
                loop = asyncio.get_running_loop()
                self.cleanup_task = loop.create_task(self._cleanup_loop())
                logger.info(
                    f"✓ Result cleanup daemon started (runs every {self.check_interval // 3600}h)"
                )
            except RuntimeError:
                # No running loop - this shouldn't happen if called from async context
                logger.error("Cannot start cleanup daemon: no running event loop")
                self.running = False
                raise

    async def stop(self):
        """Stop cleanup daemon gracefully."""
        self.running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("✓ Result cleanup daemon stopped")

    async def _cleanup_loop(self):
        """
        Main cleanup loop.

        Runs continuously in background, performing cleanup
        at regular intervals.
        """
        logger.info("Cleanup loop started")

        while self.running:
            try:
                # Wait for next cleanup interval
                await asyncio.sleep(self.check_interval)

                # Perform cleanup
                stats = await self._cleanup_expired()

                # Update totals
                self.total_deleted += stats["deleted_count"]
                self.total_freed_bytes += stats["freed_bytes"]
                self.total_errors += stats["errors"]
                self.cleanup_count += 1

                # Report to metrics service
                await self._report_metrics(stats)

                logger.info(
                    f"Cleanup completed: deleted={stats['deleted_count']}, "
                    f"freed={stats['freed_bytes']} bytes, "
                    f"errors={stats['errors']}, "
                    f"duration={stats['duration_ms']:.1f}ms"
                )

            except asyncio.CancelledError:
                logger.info("Cleanup loop cancelled")
                break

            except Exception as e:
                logger.error(f"Cleanup loop error: {e}", exc_info=True)
                # Continue running despite errors
                self.total_errors += 1

        logger.info("Cleanup loop stopped")

    async def _cleanup_expired(self) -> Dict[str, Any]:
        """
        Run cleanup and return statistics.

        Returns:
            {
                "deleted_count": int,
                "freed_bytes": int,
                "checked_count": int,
                "errors": int,
                "duration_ms": float
            }
        """
        start_time = time.time()

        stats = {
            "deleted_count": 0,
            "freed_bytes": 0,
            "checked_count": 0,
            "errors": 0,
            "duration_ms": 0.0,
        }

        # Ensure cache directory exists
        if not self.cache_dir.exists():
            logger.debug(f"Cache directory does not exist: {self.cache_dir}")
            stats["duration_ms"] = (time.time() - start_time) * 1000
            return stats

        try:
            # Find all result files recursively
            result_files = list(self.cache_dir.rglob(RESULT_FILE_PATTERN))
            logger.debug(f"Found {len(result_files)} cached result files")

            for result_file in result_files:
                stats["checked_count"] += 1

                try:
                    # Safety check: ensure file is in cache directory
                    if not self._is_safe_path(result_file):
                        logger.warning(
                            f"Skipping file outside cache directory: {result_file}"
                        )
                        continue

                    # Get companion metadata file
                    # For 'test.result.json.lz4' -> 'test.metadata.json'
                    stem = result_file.name.replace(".result.json.lz4", "")
                    metadata_file = result_file.parent / f"{stem}{METADATA_SUFFIX}"

                    # Check if metadata exists
                    if not metadata_file.exists():
                        logger.debug(f"No metadata file for: {result_file.name}")
                        continue

                    # Read metadata to check expiration
                    try:
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                    except (json.JSONDecodeError, IOError) as e:
                        logger.error(
                            f"Failed to read metadata {metadata_file.name}: {e}"
                        )
                        stats["errors"] += 1
                        continue

                    # Check if expired
                    expires_at_str = metadata.get("expires_at")
                    if not expires_at_str:
                        logger.debug(f"No expires_at in metadata: {metadata_file.name}")
                        continue

                    try:
                        expires_at = datetime.fromisoformat(expires_at_str)
                        now = datetime.now()

                        if now < expires_at:
                            # Not expired yet
                            continue

                    except (ValueError, TypeError) as e:
                        logger.error(
                            f"Invalid expires_at timestamp in {metadata_file.name}: {e}"
                        )
                        stats["errors"] += 1
                        continue

                    # File is expired - delete both result and metadata
                    freed_bytes = 0

                    # Get file sizes before deletion
                    try:
                        if result_file.exists():
                            freed_bytes += result_file.stat().st_size
                        if metadata_file.exists():
                            freed_bytes += metadata_file.stat().st_size
                    except OSError as e:
                        logger.debug(f"Could not get file size: {e}")

                    # Delete files
                    try:
                        result_file.unlink(missing_ok=True)
                        metadata_file.unlink(missing_ok=True)

                        stats["deleted_count"] += 1
                        stats["freed_bytes"] += freed_bytes

                        logger.debug(
                            f"Deleted expired result: {result_file.name} "
                            f"({freed_bytes} bytes)"
                        )

                    except OSError as e:
                        logger.error(f"Failed to delete {result_file.name}: {e}")
                        stats["errors"] += 1

                except Exception as e:
                    logger.error(
                        f"Error processing {result_file.name}: {e}",
                        exc_info=True,
                    )
                    stats["errors"] += 1
                    # Continue with next file

        except Exception as e:
            logger.error(f"Error during cleanup scan: {e}", exc_info=True)
            stats["errors"] += 1

        # Calculate duration
        stats["duration_ms"] = (time.time() - start_time) * 1000

        return stats

    def _is_safe_path(self, file_path: Path) -> bool:
        """
        Verify file is within cache directory (security check).

        Args:
            file_path: File path to check

        Returns:
            True if safe, False otherwise
        """
        try:
            # Resolve to absolute path and check if it's within cache_dir
            resolved = file_path.resolve()
            cache_dir_resolved = self.cache_dir.resolve()

            # Python 3.8 compatible: check if cache_dir is a parent
            # Convert to strings and check if resolved path starts with cache_dir
            try:
                # Try to get relative path - will raise ValueError if not relative
                resolved.relative_to(cache_dir_resolved)
                return True
            except ValueError:
                return False

        except (ValueError, OSError):
            return False

    async def _report_metrics(self, stats: Dict[str, Any]):
        """
        Report cleanup metrics to metrics service.

        Args:
            stats: Cleanup statistics to report
        """
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                payload = {
                    "deleted_count": stats["deleted_count"],
                    "freed_bytes": stats["freed_bytes"],
                    "checked_count": stats["checked_count"],
                    "errors": stats["errors"],
                    "duration_ms": stats["duration_ms"],
                    "timestamp": datetime.now().isoformat(),
                }

                response = await client.post(
                    f"{self.metrics_url}/track/cleanup",
                    json=payload,
                )

                if response.status_code == 200:
                    logger.debug("Cleanup metrics reported successfully")
                else:
                    logger.warning(
                        f"Metrics service returned status {response.status_code}"
                    )

        except Exception as e:
            # Don't crash on metrics reporting errors
            logger.debug(f"Could not report cleanup metrics: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cleanup daemon statistics.

        Returns:
            Dictionary with cleanup statistics
        """
        return {
            "running": self.running,
            "check_interval_hours": self.check_interval / 3600,
            "total_cleanups": self.cleanup_count,
            "total_deleted": self.total_deleted,
            "total_freed_bytes": self.total_freed_bytes,
            "total_freed_mb": round(self.total_freed_bytes / (1024 * 1024), 2),
            "total_errors": self.total_errors,
            "cache_directory": str(self.cache_dir),
        }
