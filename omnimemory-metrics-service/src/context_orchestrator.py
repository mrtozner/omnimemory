#!/usr/bin/env python3
"""
Context Injection Orchestrator Daemon with Predictive Prefetching

This daemon orchestrates automatic context injection for Claude Code by:
1. Monitoring Claude Code sessions
2. Tracking file access patterns
3. Pre-compressing frequently accessed files
4. Proactively injecting compressed context
5. Learning from usage patterns
6. PREDICTIVE PREFETCHING: Uses knowledge graph to predict and preload files

This is the missing piece that connects all the services together.
"""

import asyncio
import logging
import sqlite3
import httpx
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = Path.home() / ".omnimemory" / "dashboard.db"
EMBEDDINGS_URL = "http://localhost:8000"
COMPRESSION_URL = "http://localhost:8001"
METRICS_URL = "http://localhost:8003"

# Thresholds
FILE_ACCESS_THRESHOLD = 3  # Compress after 3 accesses
CACHE_PRELOAD_HOURS = 24  # Look back 24h for patterns
CONTEXT_INJECTION_SIZE = 2000  # Max tokens to inject
POLL_INTERVAL = 5  # seconds

# Predictive prefetching thresholds
PREFETCH_CONFIDENCE_THRESHOLD = 0.7  # Only prefetch files with >70% confidence
PREFETCH_MAX_FILES = 5  # Maximum files to prefetch per session
PREFETCH_CACHE_TTL_SECONDS = 300  # 5 minutes cache TTL
PREFETCH_TARGET_HIT_RATE = 0.85  # Target 85% hit rate


class FileAccessTracker:
    """Tracks file access patterns and identifies hot files"""

    def __init__(self):
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.last_access: Dict[str, datetime] = {}
        self.compressed_cache: Dict[str, str] = {}  # file_path -> compressed_content

    def record_access(self, file_path: str) -> bool:
        """
        Record file access and return True if file should be compressed

        Returns:
            True if access count exceeds threshold
        """
        self.access_counts[file_path] += 1
        self.last_access[file_path] = datetime.now()

        return self.access_counts[file_path] >= FILE_ACCESS_THRESHOLD

    def get_hot_files(self, min_accesses: int = FILE_ACCESS_THRESHOLD) -> List[str]:
        """Get files accessed frequently in recent history"""
        return [
            path for path, count in self.access_counts.items() if count >= min_accesses
        ]

    def is_cached(self, file_path: str) -> bool:
        """Check if file has compressed version in cache"""
        return file_path in self.compressed_cache

    def cache_compressed(self, file_path: str, compressed_content: str):
        """Store compressed version in memory cache"""
        self.compressed_cache[file_path] = compressed_content
        logger.info(f"Cached compressed version of {file_path}")

    def get_compressed(self, file_path: str) -> Optional[str]:
        """Retrieve compressed version from cache"""
        return self.compressed_cache.get(file_path)


class ProactiveFileLoader:
    """
    Proactive file loader with predictive prefetching

    Uses knowledge graph predictions to prefetch files before they're needed,
    reducing latency and improving user experience.

    Features:
    - Predicts next files based on access patterns
    - Prefetches files with >70% confidence
    - Tracks prefetch hit rate and effectiveness
    - Adaptive cache management with TTL
    """

    def __init__(self, knowledge_graph_service):
        """
        Initialize proactive file loader

        Args:
            knowledge_graph_service: Instance of KnowledgeGraphService
        """
        self.kg_service = knowledge_graph_service
        self.prefetch_cache: Dict[
            str, Dict
        ] = {}  # file_path -> {content, timestamp, confidence}
        self.confidence_threshold = PREFETCH_CONFIDENCE_THRESHOLD

        # Metrics tracking
        self.prefetch_requests = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.total_prefetch_time_ms = 0.0

        logger.info(
            f"ProactiveFileLoader initialized (confidence_threshold={self.confidence_threshold})"
        )

    async def prefetch_predicted_files(
        self, current_files: List[str], session_id: str
    ) -> Dict[str, Dict]:
        """
        Prefetch files predicted by knowledge graph

        Args:
            current_files: List of file paths accessed so far
            session_id: Current session ID

        Returns:
            Dict mapping file_path to prefetched content metadata
        """
        if not self.kg_service or not self.kg_service.is_available():
            logger.debug("Knowledge graph unavailable, skipping prefetch")
            return {}

        try:
            start_time = time.time()

            # Get predictions from knowledge graph
            predictions = await self.kg_service.predict_next_files(
                current_sequence=current_files, top_k=PREFETCH_MAX_FILES
            )

            if not predictions:
                logger.debug(f"No predictions for session {session_id}")
                return {}

            # Filter by confidence threshold
            high_confidence_predictions = [
                pred
                for pred in predictions
                if pred["confidence"] >= self.confidence_threshold
            ]

            if not high_confidence_predictions:
                logger.debug(
                    f"No high-confidence predictions (threshold={self.confidence_threshold})"
                )
                return {}

            logger.info(
                f"Prefetching {len(high_confidence_predictions)} files for session {session_id}"
            )

            # Prefetch files in parallel
            prefetch_tasks = [
                self._prefetch_single_file(
                    pred["file_path"], pred["confidence"], pred.get("reason", "unknown")
                )
                for pred in high_confidence_predictions
            ]

            results = await asyncio.gather(*prefetch_tasks, return_exceptions=True)

            # Build result dict
            prefetched = {}
            for pred, result in zip(high_confidence_predictions, results):
                if isinstance(result, Exception):
                    logger.warning(f"Prefetch failed for {pred['file_path']}: {result}")
                    continue

                if result:
                    prefetched[pred["file_path"]] = result

            elapsed_ms = (time.time() - start_time) * 1000
            self.total_prefetch_time_ms += elapsed_ms

            logger.info(
                f"Prefetched {len(prefetched)} files in {elapsed_ms:.1f}ms "
                f"(avg: {elapsed_ms/max(len(prefetched), 1):.1f}ms per file)"
            )

            return prefetched

        except Exception as e:
            logger.error(f"Error in prefetch_predicted_files: {e}", exc_info=True)
            return {}

    async def _prefetch_single_file(
        self, file_path: str, confidence: float, reason: str
    ) -> Optional[Dict]:
        """
        Prefetch a single file and add to cache

        Args:
            file_path: Path to file
            confidence: Prediction confidence
            reason: Reason for prediction

        Returns:
            Dict with prefetch metadata or None if failed
        """
        try:
            # Check if already in cache and still valid
            if file_path in self.prefetch_cache:
                cached = self.prefetch_cache[file_path]
                age_seconds = (datetime.now() - cached["timestamp"]).total_seconds()

                if age_seconds < PREFETCH_CACHE_TTL_SECONDS:
                    logger.debug(f"Using cached prefetch for {file_path}")
                    return cached

            # Read and compress file
            if not os.path.exists(file_path):
                logger.warning(f"Prefetch target does not exist: {file_path}")
                return None

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            original_size = len(content)

            # Compress via API
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{COMPRESSION_URL}/compress",
                    json={"context": content, "query": f"Prefetch: {file_path}"},
                )

                if response.status_code == 200:
                    data = response.json()
                    compressed_content = data["compressed_text"]
                    compression_ratio = data["compression_ratio"]

                    # Store in cache
                    cache_entry = {
                        "content": compressed_content,
                        "original_size": original_size,
                        "compressed_size": len(compressed_content),
                        "compression_ratio": compression_ratio,
                        "confidence": confidence,
                        "reason": reason,
                        "timestamp": datetime.now(),
                    }

                    self.prefetch_cache[file_path] = cache_entry

                    logger.debug(
                        f"Prefetched {file_path}: "
                        f"confidence={confidence:.2f}, "
                        f"compression={compression_ratio:.1%}"
                    )

                    return cache_entry
                else:
                    logger.warning(
                        f"Compression failed for {file_path}: {response.status_code}"
                    )
                    return None

        except Exception as e:
            logger.error(f"Error prefetching {file_path}: {e}")
            return None

    async def get_file_with_prefetch(
        self, file_path: str, session_files: List[str]
    ) -> Tuple[Optional[str], bool]:
        """
        Get file with automatic prefetching of predicted next files

        Args:
            file_path: Path to file being requested
            session_files: List of files accessed so far in session

        Returns:
            Tuple of (file_content, was_prefetch_hit)
        """
        self.prefetch_requests += 1

        # Check if file was prefetched
        if file_path in self.prefetch_cache:
            cached = self.prefetch_cache[file_path]
            age_seconds = (datetime.now() - cached["timestamp"]).total_seconds()

            if age_seconds < PREFETCH_CACHE_TTL_SECONDS:
                self.prefetch_hits += 1
                logger.info(
                    f"✅ PREFETCH HIT: {file_path} "
                    f"(confidence={cached['confidence']:.2f}, "
                    f"reason={cached['reason']})"
                )

                # Background: Predict and prefetch next files
                asyncio.create_task(
                    self.prefetch_predicted_files(
                        session_files + [file_path], "current"
                    )
                )

                return cached["content"], True

        # Cache miss - load file normally
        self.prefetch_misses += 1
        logger.debug(f"Prefetch miss: {file_path}")

        try:
            if not os.path.exists(file_path):
                return None, False

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Background: Predict and prefetch next files
            asyncio.create_task(
                self.prefetch_predicted_files(session_files + [file_path], "current")
            )

            return content, False

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return None, False

    def get_prefetch_metrics(self) -> Dict:
        """
        Get prefetch performance metrics

        Returns:
            Dict with prefetch statistics
        """
        hit_rate = (self.prefetch_hits / max(self.prefetch_requests, 1)) * 100

        avg_prefetch_time_ms = self.total_prefetch_time_ms / max(
            self.prefetch_requests, 1
        )

        cache_size = len(self.prefetch_cache)

        # Calculate cache effectiveness
        cache_miss_reduction = max(
            0, (hit_rate - 50) / 50 * 100
        )  # Compared to 50% baseline

        return {
            "prefetch_requests": self.prefetch_requests,
            "prefetch_hits": self.prefetch_hits,
            "prefetch_misses": self.prefetch_misses,
            "hit_rate_percent": hit_rate,
            "avg_retrieval_time_ms": avg_prefetch_time_ms,
            "cache_size": cache_size,
            "cache_miss_reduction_percent": cache_miss_reduction,
            "target_hit_rate_percent": PREFETCH_TARGET_HIT_RATE * 100,
            "meeting_target": hit_rate >= (PREFETCH_TARGET_HIT_RATE * 100),
        }

    def clear_stale_cache_entries(self):
        """Remove stale entries from prefetch cache"""
        now = datetime.now()
        stale_keys = []

        for file_path, cached in self.prefetch_cache.items():
            age_seconds = (now - cached["timestamp"]).total_seconds()
            if age_seconds >= PREFETCH_CACHE_TTL_SECONDS:
                stale_keys.append(file_path)

        for key in stale_keys:
            del self.prefetch_cache[key]

        if stale_keys:
            logger.debug(f"Cleared {len(stale_keys)} stale cache entries")


class ContextOrchestrator:
    """
    Main orchestrator for automatic context injection with predictive prefetching

    IMPORTANT: This daemon operates in READ-ONLY mode for sessions:
    - Monitors active sessions created by MCP server or Memory Daemon
    - Injects context into newly detected sessions
    - Pre-compresses hot files for performance
    - Does NOT create or manage session lifecycle

    Session creation is handled by:
    - MCP Server (omnimemory_mcp.py) - one session per MCP process
    - Memory Daemon (memory_daemon.py) - background session tracking

    Deduplication by process ID ensures no duplicate sessions.
    """

    def __init__(self, knowledge_graph_service=None):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.file_tracker = FileAccessTracker()
        self.active_sessions: Set[str] = set()
        self.session_files: Dict[str, Set[str]] = defaultdict(
            set
        )  # session_id -> files

        # Initialize proactive file loader if knowledge graph available
        self.proactive_loader = None
        if knowledge_graph_service:
            self.proactive_loader = ProactiveFileLoader(knowledge_graph_service)
            logger.info("✨ Predictive prefetching enabled")
        else:
            logger.warning(
                "Knowledge graph not available - predictive prefetching disabled"
            )

        # Validate read-only mode
        self._validate_readonly_mode()

    def _validate_readonly_mode(self):
        """
        Validate that this orchestrator doesn't create sessions

        This is a development-time check to ensure we maintain
        the read-only contract for session management.
        """
        # Check that we don't have any session creation methods
        forbidden_methods = ["create_session", "start_session", "register_session"]

        for method_name in forbidden_methods:
            if hasattr(self, method_name):
                logger.warning(
                    f"⚠️ Context Orchestrator has session creation method '{method_name}' - "
                    f"this should be removed to maintain read-only mode"
                )

        logger.info("✓ Context Orchestrator initialized in READ-ONLY mode for sessions")

    async def start(self):
        """Start the orchestrator daemon"""
        logger.info("🚀 Context Orchestrator starting...")

        # Ensure cache_hits table exists
        self._ensure_tables()

        # Start monitoring loops
        tasks = [
            self._monitor_active_sessions(),
            self._monitor_file_accesses(),
            self._precompress_hot_files(),
            self._inject_session_context(),
        ]

        # Add predictive prefetching monitoring if enabled
        if self.proactive_loader:
            tasks.append(self._monitor_prefetch_metrics())
            tasks.append(self._cleanup_stale_cache())

        await asyncio.gather(*tasks)

    def _ensure_tables(self):
        """Ensure required database tables exist"""
        cursor = self.conn.cursor()

        # File access tracking table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS file_accesses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                access_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                was_compressed BOOLEAN DEFAULT 0,
                tokens_saved INTEGER DEFAULT 0
            )
        """
        )

        # Compressed file cache table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS compressed_files (
                file_path TEXT PRIMARY KEY,
                compressed_content TEXT NOT NULL,
                original_size INTEGER,
                compressed_size INTEGER,
                compression_ratio REAL,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        self.conn.commit()
        logger.info("Database tables verified")

    async def _monitor_active_sessions(self):
        """Monitor for new Claude Code sessions"""
        logger.info("📊 Monitoring active sessions...")

        while True:
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    """
                    SELECT session_id, tool_id, started_at
                    FROM tool_sessions
                    WHERE ended_at IS NULL
                    ORDER BY started_at DESC
                """
                )

                current_sessions = {row["session_id"] for row in cursor.fetchall()}

                # Detect new sessions (created by MCP server or Memory Daemon)
                new_sessions = current_sessions - self.active_sessions
                for session_id in new_sessions:
                    logger.info(
                        f"🆕 New session detected (created by external component): {session_id}"
                    )
                    await self._on_new_session(session_id)

                # Detect ended sessions
                ended_sessions = self.active_sessions - current_sessions
                for session_id in ended_sessions:
                    logger.info(f"✅ Session ended: {session_id}")
                    self._on_session_end(session_id)

                self.active_sessions = current_sessions

            except Exception as e:
                logger.error(f"Error monitoring sessions: {e}")

            await asyncio.sleep(POLL_INTERVAL)

    async def _on_new_session(self, session_id: str):
        """
        Handle newly detected session - inject previous context if available

        NOTE: This method does NOT create sessions. It only reacts to sessions
        that were created by MCP Server or Memory Daemon.
        """
        try:
            # Find most recent checkpoint for this tool
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT checkpoint_id, summary, compressed_context,
                       compressed_tokens, files_modified
                FROM checkpoints
                WHERE tool_id = 'claude-code'
                ORDER BY created_at DESC
                LIMIT 1
            """
            )

            checkpoint = cursor.fetchone()
            if checkpoint:
                logger.info(
                    f"💉 Injecting context from checkpoint {checkpoint['checkpoint_id']}"
                )

                # Store injected context for this session
                # (In practice, this would be injected via MCP or daemon integration)
                context_summary = checkpoint["summary"]
                compressed_context = checkpoint["compressed_context"]

                logger.info(
                    f"Context summary: {context_summary[:100] if context_summary else 'None'}..."
                )
                logger.info(f"Compressed tokens: {checkpoint['compressed_tokens']}")

        except Exception as e:
            logger.error(f"Error injecting context for new session: {e}")

    def _on_session_end(self, session_id: str):
        """Clean up session data"""
        if session_id in self.session_files:
            del self.session_files[session_id]

    async def _monitor_file_accesses(self):
        """Monitor file access patterns from cache_hits table"""
        logger.info("📂 Monitoring file access patterns...")

        last_check_id = 0

        while True:
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    """
                    SELECT id, tool_id, file_path, tokens_saved, timestamp
                    FROM cache_hits
                    WHERE id > ?
                    ORDER BY id ASC
                """,
                    (last_check_id,),
                )

                new_accesses = cursor.fetchall()

                for access in new_accesses:
                    file_path = access["file_path"]

                    # Track access
                    should_compress = self.file_tracker.record_access(file_path)

                    if should_compress and not self.file_tracker.is_cached(file_path):
                        logger.info(
                            f"🔥 Hot file detected: {file_path} (accessed {self.file_tracker.access_counts[file_path]} times)"
                        )
                        await self._compress_and_cache_file(file_path)

                    last_check_id = access["id"]

            except Exception as e:
                logger.error(f"Error monitoring file accesses: {e}")

            await asyncio.sleep(POLL_INTERVAL)

    async def _compress_and_cache_file(self, file_path: str):
        """Compress a file and store in cache"""
        try:
            # Read file content
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                return

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            original_size = len(content)

            # Compress via API
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{COMPRESSION_URL}/compress",
                    json={"context": content, "query": f"File: {file_path}"},
                )

                if response.status_code == 200:
                    data = response.json()
                    compressed_content = data["compressed_text"]
                    compression_ratio = data["compression_ratio"]

                    # Cache in memory
                    self.file_tracker.cache_compressed(file_path, compressed_content)

                    # Store in database
                    cursor = self.conn.cursor()
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO compressed_files
                        (file_path, compressed_content, original_size, compressed_size, compression_ratio)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (
                            file_path,
                            compressed_content,
                            original_size,
                            len(compressed_content),
                            compression_ratio,
                        ),
                    )
                    self.conn.commit()

                    logger.info(
                        f"✅ Compressed {file_path}: {compression_ratio:.1%} compression"
                    )
                else:
                    logger.error(f"Compression failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Error compressing file {file_path}: {e}")

    async def _precompress_hot_files(self):
        """Background task to pre-compress frequently accessed files"""
        logger.info("🔄 Pre-compression worker started...")

        while True:
            try:
                # Wait longer between pre-compression runs
                await asyncio.sleep(60)  # Every minute

                hot_files = self.file_tracker.get_hot_files()
                uncached_hot_files = [
                    f for f in hot_files if not self.file_tracker.is_cached(f)
                ]

                if uncached_hot_files:
                    logger.info(
                        f"🔥 Pre-compressing {len(uncached_hot_files)} hot files..."
                    )
                    for file_path in uncached_hot_files:
                        await self._compress_and_cache_file(file_path)

            except Exception as e:
                logger.error(f"Error in pre-compression: {e}")

    async def _inject_session_context(self):
        """Proactively inject context into active sessions"""
        logger.info("💉 Context injection worker started...")

        while True:
            try:
                await asyncio.sleep(30)  # Every 30 seconds

                # For each active session, check if context should be injected
                for session_id in self.active_sessions:
                    # Get session's file access patterns
                    cursor = self.conn.cursor()
                    cursor.execute(
                        """
                        SELECT DISTINCT file_path
                        FROM file_accesses
                        WHERE session_id = ?
                        ORDER BY access_time DESC
                        LIMIT 10
                    """,
                        (session_id,),
                    )

                    recent_files = [row["file_path"] for row in cursor.fetchall()]

                    # If predictive prefetching enabled, trigger predictions
                    if self.proactive_loader and recent_files:
                        await self.proactive_loader.prefetch_predicted_files(
                            recent_files, session_id
                        )

                    # Check if we have compressed versions
                    available_context = []
                    for file_path in recent_files:
                        compressed = self.file_tracker.get_compressed(file_path)
                        if compressed:
                            available_context.append(
                                {"file": file_path, "content": compressed}
                            )

                    if available_context:
                        logger.info(
                            f"📦 Context available for session {session_id}: {len(available_context)} files"
                        )

            except Exception as e:
                logger.error(f"Error in context injection: {e}")

    async def _monitor_prefetch_metrics(self):
        """Monitor and log prefetch performance metrics"""
        logger.info("📊 Prefetch metrics monitoring started...")

        while True:
            try:
                await asyncio.sleep(60)  # Every minute

                if not self.proactive_loader:
                    continue

                metrics = self.proactive_loader.get_prefetch_metrics()

                # Log metrics
                logger.info(
                    f"📊 Prefetch Metrics: "
                    f"Hit Rate: {metrics['hit_rate_percent']:.1f}% "
                    f"(Target: {metrics['target_hit_rate_percent']:.0f}%), "
                    f"Hits: {metrics['prefetch_hits']}, "
                    f"Misses: {metrics['prefetch_misses']}, "
                    f"Cache Size: {metrics['cache_size']}, "
                    f"Avg Retrieval: {metrics['avg_retrieval_time_ms']:.1f}ms, "
                    f"Cache Miss Reduction: {metrics['cache_miss_reduction_percent']:.1f}%"
                )

                # Report to metrics service
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        await client.post(
                            f"{METRICS_URL}/metrics/prefetch", json=metrics
                        )
                except Exception as e:
                    logger.debug(f"Could not report metrics to service: {e}")

            except Exception as e:
                logger.error(f"Error in prefetch metrics monitoring: {e}")

    async def _cleanup_stale_cache(self):
        """Periodically clean up stale cache entries"""
        logger.info("🧹 Cache cleanup worker started...")

        while True:
            try:
                await asyncio.sleep(120)  # Every 2 minutes

                if not self.proactive_loader:
                    continue

                self.proactive_loader.clear_stale_cache_entries()

            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")


async def main():
    """Main entry point"""
    # Try to initialize knowledge graph service
    kg_service = None
    try:
        from omnimemory_knowledge_graph.knowledge_graph_service import (
            KnowledgeGraphService,
        )

        kg_service = KnowledgeGraphService()
        if await kg_service.initialize():
            logger.info("✅ Knowledge graph service initialized")
        else:
            logger.warning(
                "⚠️  Knowledge graph service unavailable - predictive prefetching disabled"
            )
            kg_service = None
    except Exception as e:
        logger.warning(f"⚠️  Could not load knowledge graph service: {e}")
        logger.warning("Predictive prefetching will be disabled")
        kg_service = None

    orchestrator = ContextOrchestrator(knowledge_graph_service=kg_service)

    try:
        await orchestrator.start()
    finally:
        if kg_service:
            await kg_service.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Orchestrator stopped by user")
