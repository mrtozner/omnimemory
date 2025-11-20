"""
Autonomous Workflow Learning Service

Monitors user activity from memory daemon and automatically learns
workflow patterns via the procedural memory API.

Features:
- Monitors memory daemon events every 30 seconds
- Groups events into logical workflow sessions
- Automatically learns patterns when sessions become idle
- No manual MCP tool calls required
- Graceful error handling and comprehensive logging
"""

import asyncio
import sqlite3
import httpx
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path.home() / ".omnimemory" / "workflow_learner.log"),
    ],
)
logger = logging.getLogger(__name__)

# Configuration constants
SESSION_IDLE_MINUTES = 5  # Sessions idle for 5 minutes are considered complete
POLL_INTERVAL_SECONDS = 30  # Check for new events every 30 seconds
MIN_COMMANDS_FOR_LEARNING = 3  # Minimum commands needed to learn a pattern
PROCEDURAL_API_URL = "http://localhost:8002"
MEMORY_DAEMON_DB_PATH = Path.home() / ".memory-daemon" / "storage" / "memory_data.db"


class WorkflowLearner:
    """
    Autonomous workflow pattern learning from user activity

    Monitors the memory daemon database for user activity and automatically
    learns workflow patterns without requiring manual intervention.
    """

    def __init__(
        self,
        memory_db_path: Path = MEMORY_DAEMON_DB_PATH,
        procedural_url: str = PROCEDURAL_API_URL,
        session_idle_minutes: int = SESSION_IDLE_MINUTES,
        poll_interval_seconds: int = POLL_INTERVAL_SECONDS,
    ):
        """
        Initialize workflow learner

        Args:
            memory_db_path: Path to memory daemon SQLite database
            procedural_url: URL of procedural memory API
            session_idle_minutes: Minutes of idle time before session is considered complete
            poll_interval_seconds: Seconds between database polls
        """
        self.memory_db_path = memory_db_path
        self.procedural_url = procedural_url
        self.session_idle_threshold = timedelta(minutes=session_idle_minutes)
        self.poll_interval = poll_interval_seconds

        # Track active sessions (tool_id -> {events, last_activity})
        self.active_sessions: Dict[str, Dict] = {}

        # Track last processed event ID
        self.last_processed_id = 0

        # Track learned patterns to avoid duplicates
        self.learned_patterns: Set[str] = set()

        # Statistics
        self.stats = {
            "events_processed": 0,
            "sessions_created": 0,
            "patterns_learned": 0,
            "errors": 0,
        }

        self.running = False

        logger.info("=" * 60)
        logger.info("OmniMemory Autonomous Workflow Learner")
        logger.info("=" * 60)
        logger.info(f"Memory daemon DB: {self.memory_db_path}")
        logger.info(f"Procedural API: {self.procedural_url}")
        logger.info(f"Session idle threshold: {session_idle_minutes} minutes")
        logger.info(f"Poll interval: {poll_interval_seconds} seconds")
        logger.info(f"Min commands for learning: {MIN_COMMANDS_FOR_LEARNING}")
        logger.info("=" * 60)

    async def start(self):
        """Start the workflow learning service"""
        # Verify memory daemon database exists
        if not self.memory_db_path.exists():
            logger.warning(
                f"Memory daemon database not found at {self.memory_db_path}. "
                f"Waiting for database to be created..."
            )

        # Verify procedural API is accessible
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.procedural_url}/health")
                if response.status_code == 200:
                    logger.info("Procedural API health check passed")
                else:
                    logger.warning(
                        f"Procedural API returned status {response.status_code}"
                    )
        except Exception as e:
            logger.warning(f"Procedural API health check failed: {e}")
            logger.warning("Will continue but pattern learning may fail")

        self.running = True
        logger.info("Workflow learning service started")

        try:
            await self._monitor_loop()
        except Exception as e:
            logger.error(f"Fatal error in workflow learner: {e}", exc_info=True)
            raise
        finally:
            await self.stop()

    async def stop(self):
        """Stop the workflow learning service"""
        self.running = False
        logger.info("Workflow learning service stopped")
        logger.info(f"Statistics: {self.stats}")

    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                await self._process_recent_events()
                await self._check_idle_sessions()
            except Exception as e:
                logger.error(f"Error in learning loop: {e}", exc_info=True)
                self.stats["errors"] += 1

            await asyncio.sleep(self.poll_interval)

    async def _process_recent_events(self):
        """Process recent events from memory daemon database"""
        if not self.memory_db_path.exists():
            logger.debug("Memory daemon database not found, skipping")
            return

        try:
            conn = sqlite3.connect(str(self.memory_db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Query events since last processed
            # Note: Using 'id' column (INTEGER PRIMARY KEY AUTOINCREMENT)
            cursor.execute(
                """
                SELECT id, timestamp, event_type, session_id
                FROM events
                WHERE id > ?
                ORDER BY id ASC
                LIMIT 1000
                """,
                (self.last_processed_id,),
            )

            rows = cursor.fetchall()
            conn.close()

            if rows:
                logger.debug(f"Processing {len(rows)} new events")

                for row in rows:
                    event = dict(row)
                    await self._process_event(event)

                    # Update last processed ID
                    if event["id"] > self.last_processed_id:
                        self.last_processed_id = event["id"]

                    self.stats["events_processed"] += 1

                logger.info(
                    f"Processed {len(rows)} events "
                    f"(total: {self.stats['events_processed']})"
                )

        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            self.stats["errors"] += 1
        except Exception as e:
            logger.error(f"Error processing events: {e}", exc_info=True)
            self.stats["errors"] += 1

    async def _process_event(self, event: Dict):
        """
        Process a single event and add to active sessions

        Args:
            event: Event dictionary with id, timestamp, event_type, session_id
        """
        event_type = event["event_type"]
        timestamp_str = event["timestamp"]
        session_id = event.get("session_id", "default")

        # Parse timestamp
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
        except Exception as e:
            logger.warning(f"Invalid timestamp format: {timestamp_str}")
            timestamp = datetime.now()

        # Convert event type to command
        command = self._event_type_to_command(event_type)

        if not command:
            # Skip irrelevant event types
            return

        # Get or create session
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                "commands": [],
                "last_activity": timestamp,
                "started_at": timestamp,
            }
            self.stats["sessions_created"] += 1
            logger.debug(f"Created new session: {session_id}")

        session = self.active_sessions[session_id]

        # Add command to session
        session["commands"].append(
            {
                "command": command,
                "timestamp": timestamp.timestamp(),
                "context": {"event_type": event_type},
            }
        )
        session["last_activity"] = timestamp

        logger.debug(
            f"Added command to session {session_id}: {command} "
            f"(total commands: {len(session['commands'])})"
        )

    def _event_type_to_command(self, event_type: str) -> Optional[str]:
        """
        Convert memory daemon event type to a command string

        Args:
            event_type: Event type from memory daemon

        Returns:
            Command string or None if event type is not relevant
        """
        # Map event types to meaningful command strings
        event_mapping = {
            "file_open": "open_file",
            "file_save": "save_file",
            "file_create": "create_file",
            "file_delete": "delete_file",
            "file_modify": "modify_file",
            "process_start": "execute_command",
            "process_end": "command_completed",
            "editor_open": "open_editor",
            "editor_close": "close_editor",
            "code_edit": "edit_code",
            "test_run": "run_tests",
            "build_start": "build_project",
            "git_commit": "commit_changes",
            "git_push": "push_changes",
            "git_pull": "pull_changes",
        }

        return event_mapping.get(event_type)

    async def _check_idle_sessions(self):
        """Check for idle sessions and learn from them"""
        now = datetime.now()
        idle_sessions = []

        for session_id, session in self.active_sessions.items():
            time_since_activity = now - session["last_activity"]

            if time_since_activity > self.session_idle_threshold:
                idle_sessions.append(session_id)

        # Process idle sessions
        for session_id in idle_sessions:
            await self._learn_from_session(session_id)
            del self.active_sessions[session_id]

    async def _learn_from_session(self, session_id: str):
        """
        Learn workflow patterns from a completed session

        Args:
            session_id: Session identifier
        """
        session = self.active_sessions[session_id]
        commands = session["commands"]

        if len(commands) < MIN_COMMANDS_FOR_LEARNING:
            logger.debug(
                f"Session {session_id} too short to learn "
                f"({len(commands)} commands, need {MIN_COMMANDS_FOR_LEARNING})"
            )
            return

        # Create pattern hash to avoid duplicate learning
        command_sequence = [cmd["command"] for cmd in commands]
        pattern_hash = hash(tuple(command_sequence))

        if pattern_hash in self.learned_patterns:
            logger.debug(f"Pattern already learned for session {session_id}")
            return

        # Infer session outcome
        outcome = self._infer_session_outcome(commands)

        logger.info(
            f"Learning from session {session_id}: "
            f"{len(commands)} commands, outcome={outcome}"
        )

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.procedural_url}/learn",
                    json={
                        "session_commands": commands,
                        "session_outcome": outcome,
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    pattern_id = data.get("pattern_id")

                    if pattern_id:
                        logger.info(
                            f"Successfully learned pattern {pattern_id} from "
                            f"session {session_id} ({len(commands)} commands)"
                        )
                        self.stats["patterns_learned"] += 1
                        self.learned_patterns.add(pattern_hash)
                    else:
                        logger.debug(
                            f"Session {session_id} did not create a new pattern "
                            f"(may have updated existing pattern)"
                        )
                else:
                    logger.warning(
                        f"Failed to learn from session {session_id}: "
                        f"HTTP {response.status_code}"
                    )
                    self.stats["errors"] += 1

        except httpx.TimeoutException:
            logger.error(
                f"Timeout learning from session {session_id}: "
                f"Procedural API took too long to respond"
            )
            self.stats["errors"] += 1
        except httpx.RequestError as e:
            logger.error(
                f"Failed to connect to procedural API for session {session_id}: {e}"
            )
            self.stats["errors"] += 1
        except Exception as e:
            logger.error(
                f"Error learning from session {session_id}: {e}", exc_info=True
            )
            self.stats["errors"] += 1

    def _infer_session_outcome(self, commands: List[Dict]) -> str:
        """
        Infer whether a session was successful or failed

        Args:
            commands: List of command dictionaries

        Returns:
            "success" or "failure"
        """
        # Look for success indicators in recent commands
        last_commands = [cmd["command"] for cmd in commands[-3:]]

        success_indicators = [
            "commit_changes",
            "push_changes",
            "save_file",
            "command_completed",
        ]

        failure_indicators = [
            "delete_file",  # Deleting work might indicate failure
        ]

        # Check for success indicators
        for indicator in success_indicators:
            if indicator in last_commands:
                return "success"

        # Check for failure indicators
        for indicator in failure_indicators:
            if indicator in last_commands:
                return "failure"

        # Default to success if session completed normally
        return "success"

    def get_stats(self) -> Dict:
        """
        Get workflow learner statistics

        Returns:
            Dictionary with statistics
        """
        return {
            "running": self.running,
            "active_sessions": len(self.active_sessions),
            "stats": self.stats,
            "session_idle_threshold_minutes": self.session_idle_threshold.total_seconds()
            / 60,
            "poll_interval_seconds": self.poll_interval,
        }


async def main():
    """Main entry point for standalone execution"""
    logger.info("Starting OmniMemory Autonomous Workflow Learner...")

    learner = WorkflowLearner()

    try:
        await learner.start()
    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        await learner.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nExiting...")
        sys.exit(0)
