"""
Autonomous Checkpoint Monitoring Service with Predictive Capabilities

Monitors active tool sessions and automatically creates checkpoints when token
thresholds are reached or predicted to be reached. Runs as a standalone background worker.

Features:
- Monitors all active sessions across multiple AI coding tools
- Predictive checkpointing based on velocity and acceleration
- Creates checkpoints at 100k (warning), 125k (urgent), 150k tokens (critical)
- Generates intelligent summaries from session metrics
- Stores embeddings for semantic search
- Pattern learning for improved predictions
- Graceful error handling and comprehensive logging
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
import math

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from .data_store import MetricsStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path.home() / ".omnimemory" / "checkpoint_monitor.log"),
    ],
)
logger = logging.getLogger(__name__)

# Configuration constants - Original thresholds
TOKEN_THRESHOLD = 150000  # 75% of Claude Code's 200k token limit (must checkpoint)
POLL_INTERVAL_SECONDS = 10  # Check every 10 seconds
ENABLE_AUTO_CHECKPOINT = True

# Predictive checkpointing thresholds (new)
WARNING_THRESHOLD = 100000  # Start monitoring closely
URGENT_THRESHOLD = 125000  # Checkpoint soon
ENABLE_PREDICTIVE_CHECKPOINTING = True  # Feature flag

# Velocity thresholds (new)
HIGH_VELOCITY = 1000  # tokens/minute - indicates heavy usage
ACCELERATION_THRESHOLD = 100  # tokens/minute² - usage speeding up

# Prediction settings (new)
EXPONENTIAL_SMOOTHING_ALPHA = 0.3  # Weight for new observations
PREDICTION_LOOKBACK_MINUTES = 10  # Use last 10 min for predictions
MIN_DATA_POINTS = 3  # Minimum observations before predicting
IDLE_THRESHOLD_SECONDS = 120  # 2 minutes without activity = idle

# Supported tools
SUPPORTED_TOOLS = [
    "claude-code",
    "cursor",
    "continue",
    "aider",
    "windsurf",
    "codex",
]


class CheckpointMonitor:
    """
    Autonomous checkpoint monitoring service with predictive capabilities

    Monitors active sessions and creates checkpoints when token thresholds
    are reached or predicted to be reached, without requiring any external configuration.
    """

    def __init__(
        self,
        token_threshold: int = TOKEN_THRESHOLD,
        poll_interval: int = POLL_INTERVAL_SECONDS,
        metrics_store: Optional[MetricsStore] = None,
        enable_predictive: bool = ENABLE_PREDICTIVE_CHECKPOINTING,
    ):
        """
        Initialize checkpoint monitor

        Args:
            token_threshold: Token count that triggers checkpoint creation
            poll_interval: Seconds between session checks
            metrics_store: Optional pre-initialized MetricsStore (for testing)
            enable_predictive: Enable predictive checkpointing features
        """
        self.token_threshold = token_threshold
        self.poll_interval = poll_interval
        self.checkpointed_sessions: Set[str] = set()
        self.metrics_store = metrics_store
        self.running = False
        self.enable_predictive = enable_predictive

        # Track velocity data for predictions (new)
        self.session_velocities: Dict[str, Dict] = {}
        self.last_token_counts: Dict[str, Tuple[int, datetime]] = {}

        logger.info(
            f"Checkpoint Monitor initialized (threshold: {token_threshold} tokens, "
            f"poll interval: {poll_interval}s, predictive: {enable_predictive})"
        )

    async def start(self):
        """Start the monitoring service"""
        try:
            # Initialize metrics store if not provided
            if self.metrics_store is None:
                # Disable vector store to avoid Qdrant lock conflict with other services
                self.metrics_store = MetricsStore(enable_vector_store=False)
                logger.info(
                    "Connected to metrics database (vector store disabled to avoid lock conflict)"
                )
            else:
                logger.info("Using provided metrics store")

            # Verify vector store is available
            if self.metrics_store.vector_store is None:
                logger.warning(
                    "Vector store not available - checkpoints will be stored "
                    "without embeddings"
                )

            self.running = True
            logger.info("Checkpoint monitoring service started")

            # Start monitoring loop
            await self._monitor_loop()

        except Exception as e:
            logger.error(f"Failed to start checkpoint monitor: {e}", exc_info=True)
            raise

    async def stop(self):
        """Stop the monitoring service"""
        self.running = False
        if self.metrics_store:
            self.metrics_store.close()
        logger.info("Checkpoint monitoring service stopped")

    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                await self._check_sessions()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                # Continue running despite errors

            await asyncio.sleep(self.poll_interval)

    async def _check_sessions(self):
        """Check all active sessions for checkpoint triggers"""
        if not self.metrics_store:
            logger.error("Metrics store not initialized")
            return

        # Get all active sessions
        active_sessions = self.metrics_store.get_active_sessions()

        if not active_sessions:
            logger.debug("No active sessions to monitor")
            return

        logger.debug(f"Checking {len(active_sessions)} active sessions")

        for session in active_sessions:
            await self._process_session(session)

    async def _process_session(self, session: Dict):
        """
        Process a single session with predictive checkpointing

        Args:
            session: Session dictionary from database
        """
        session_id = session.get("session_id")
        tool_id = session.get("tool_id", "unknown")
        tokens_saved = session.get("tokens_saved", 0)

        # Skip if already checkpointed at highest threshold
        if (
            session_id in self.checkpointed_sessions
            and tokens_saved < self.token_threshold
        ):
            return

        # Calculate velocity if predictive is enabled
        velocity_data = None
        if self.enable_predictive:
            velocity_data = self._calculate_velocity(session)
            if velocity_data:
                # Store velocity for pattern learning
                self._learn_pattern(session, velocity_data)

                # Log velocity information
                logger.debug(
                    f"Session {session_id} velocity: {velocity_data['velocity']:.1f} tokens/min, "
                    f"acceleration: {velocity_data.get('acceleration', 0):.1f} tokens/min²"
                )

        # Get checkpoint strategy
        strategy = self._get_checkpoint_strategy(session, tokens_saved, velocity_data)

        # Execute strategy
        if strategy != "none":
            should_checkpoint, reason = self._should_checkpoint(
                session, tokens_saved, velocity_data, strategy
            )

            if should_checkpoint:
                logger.info(
                    f"Session {session_id} ({tool_id}) checkpoint triggered: "
                    f"{tokens_saved} tokens, strategy: {strategy}, reason: {reason}"
                )

                # Make and store prediction if applicable
                if self.enable_predictive and velocity_data:
                    predicted_time = self._predict_checkpoint_time(
                        session, velocity_data
                    )
                    if predicted_time:
                        self.metrics_store.store_checkpoint_prediction(
                            session_id=session_id,
                            predicted_checkpoint_time=predicted_time.isoformat(),
                            predicted_tokens=self.token_threshold,
                            strategy=strategy,
                        )

                await self._create_checkpoint(session, strategy=strategy)

    def _calculate_velocity(self, session: Dict) -> Optional[Dict]:
        """
        Calculate current token velocity and acceleration

        Args:
            session: Session dictionary

        Returns:
            Velocity data dictionary or None if insufficient data
        """
        session_id = session.get("session_id")
        tokens_saved = session.get("tokens_saved", 0)
        now = datetime.now()

        # Get last measurement
        if session_id in self.last_token_counts:
            last_tokens, last_time = self.last_token_counts[session_id]
            time_diff = (now - last_time).total_seconds() / 60  # Minutes

            if time_diff > 0:
                # Calculate velocity
                velocity = (tokens_saved - last_tokens) / time_diff

                # Calculate acceleration if we have previous velocity
                acceleration = 0
                if session_id in self.session_velocities:
                    last_velocity = self.session_velocities[session_id].get(
                        "velocity", 0
                    )
                    acceleration = (velocity - last_velocity) / time_diff

                # Store current velocity
                velocity_data = {
                    "velocity": velocity,
                    "acceleration": acceleration,
                    "time_diff": time_diff,
                    "tokens_diff": tokens_saved - last_tokens,
                    "trend": self._get_trend(velocity, acceleration),
                }

                # Update session velocity tracking
                self.session_velocities[session_id] = velocity_data

                # Store in database for pattern learning
                self.metrics_store.store_session_velocity(
                    session_id=session_id,
                    tokens_saved=tokens_saved,
                    velocity=velocity,
                    acceleration=acceleration,
                    prediction_confidence=self._calculate_confidence(velocity_data),
                )

                # Update last measurement
                self.last_token_counts[session_id] = (tokens_saved, now)

                return velocity_data

        # First measurement for this session
        self.last_token_counts[session_id] = (tokens_saved, now)
        return None

    def _get_trend(self, velocity: float, acceleration: float) -> str:
        """
        Determine velocity trend

        Args:
            velocity: Current velocity
            acceleration: Current acceleration

        Returns:
            Trend description: "accelerating", "steady", "decelerating", "idle"
        """
        if velocity < 10:  # Almost no activity
            return "idle"
        elif acceleration > ACCELERATION_THRESHOLD:
            return "accelerating"
        elif acceleration < -ACCELERATION_THRESHOLD:
            return "decelerating"
        else:
            return "steady"

    def _calculate_confidence(self, velocity_data: Dict) -> float:
        """
        Calculate prediction confidence based on velocity stability

        Args:
            velocity_data: Velocity information

        Returns:
            Confidence score 0-100
        """
        trend = velocity_data.get("trend", "unknown")
        velocity = velocity_data.get("velocity", 0)

        # High confidence for steady patterns
        if trend == "steady":
            return min(90, 50 + (velocity / HIGH_VELOCITY) * 40)
        # Medium confidence for acceleration/deceleration
        elif trend in ["accelerating", "decelerating"]:
            return 60
        # Low confidence for idle or unknown
        else:
            return 30

    def _predict_checkpoint_time(
        self, session: Dict, velocity_data: Dict
    ) -> Optional[datetime]:
        """
        Predict when checkpoint will be needed using exponential smoothing

        Args:
            session: Session dictionary
            velocity_data: Current velocity data

        Returns:
            Predicted checkpoint time or None
        """
        tokens_saved = session.get("tokens_saved", 0)
        velocity = velocity_data.get("velocity", 0)
        acceleration = velocity_data.get("acceleration", 0)

        if velocity <= 0:
            return None

        # Get historical velocity data
        session_id = session.get("session_id")
        history = self.metrics_store.get_session_velocity_history(
            session_id, limit=PREDICTION_LOOKBACK_MINUTES
        )

        if len(history) < MIN_DATA_POINTS:
            # Simple linear prediction
            tokens_remaining = self.token_threshold - tokens_saved
            minutes_to_threshold = tokens_remaining / velocity
        else:
            # Exponential smoothing prediction
            smoothed_velocity = velocity
            for record in history:
                hist_velocity = record.get("velocity", velocity)
                smoothed_velocity = (
                    EXPONENTIAL_SMOOTHING_ALPHA * hist_velocity
                    + (1 - EXPONENTIAL_SMOOTHING_ALPHA) * smoothed_velocity
                )

            # Account for acceleration
            if acceleration > 0:
                # Accelerating - use quadratic prediction
                # tokens = v*t + 0.5*a*t²
                # Solve for t when tokens = threshold
                a = acceleration / 2
                b = smoothed_velocity
                c = tokens_saved - self.token_threshold

                # Quadratic formula
                discriminant = b**2 - 4 * a * c
                if discriminant >= 0:
                    minutes_to_threshold = (-b + math.sqrt(discriminant)) / (2 * a)
                else:
                    minutes_to_threshold = (
                        self.token_threshold - tokens_saved
                    ) / smoothed_velocity
            else:
                # Linear prediction with smoothed velocity
                tokens_remaining = self.token_threshold - tokens_saved
                minutes_to_threshold = tokens_remaining / smoothed_velocity

        # Calculate predicted time
        if minutes_to_threshold > 0 and minutes_to_threshold < 1440:  # Within 24 hours
            predicted_time = datetime.now() + timedelta(minutes=minutes_to_threshold)

            logger.debug(
                f"Predicted checkpoint for {session_id} in {minutes_to_threshold:.1f} minutes "
                f"(velocity: {velocity:.1f}, acceleration: {acceleration:.1f})"
            )

            return predicted_time

        return None

    def _get_checkpoint_strategy(
        self, session: Dict, tokens: int, velocity_data: Optional[Dict]
    ) -> str:
        """
        Determine checkpoint strategy based on current state

        Args:
            session: Session dictionary
            tokens: Current token count
            velocity_data: Velocity information

        Returns:
            Strategy: "immediate", "urgent", "warning", "predictive", "normal", "none"
        """
        # Critical threshold - must checkpoint
        if tokens >= self.token_threshold:
            return "immediate"

        # Urgent threshold
        if tokens >= URGENT_THRESHOLD:
            return "urgent"

        # Warning threshold
        if tokens >= WARNING_THRESHOLD:
            # Check if accelerating rapidly
            if velocity_data:
                if velocity_data.get("velocity", 0) > HIGH_VELOCITY:
                    return "urgent"
                elif velocity_data.get("trend") == "accelerating":
                    return "warning"
            return "warning"

        # Predictive strategy
        if self.enable_predictive and velocity_data:
            predicted_time = self._predict_checkpoint_time(session, velocity_data)
            if predicted_time:
                minutes_to_checkpoint = (
                    predicted_time - datetime.now()
                ).total_seconds() / 60

                # Will hit threshold soon
                if minutes_to_checkpoint < 5:
                    return "predictive"
                # High velocity warning
                elif velocity_data.get("velocity", 0) > HIGH_VELOCITY:
                    return "warning"

        return "none"

    def _should_checkpoint(
        self, session: Dict, tokens: int, velocity_data: Optional[Dict], strategy: str
    ) -> Tuple[bool, str]:
        """
        Determine if checkpoint should be created

        Args:
            session: Session dictionary
            tokens: Current token count
            velocity_data: Velocity information
            strategy: Checkpoint strategy

        Returns:
            Tuple of (should_checkpoint, reason)
        """
        session_id = session.get("session_id")

        # Check if session is idle (good time to checkpoint)
        is_idle = self._is_session_idle(session, velocity_data)

        if strategy == "immediate":
            return True, "threshold_reached"

        elif strategy == "urgent":
            # Checkpoint if idle or very high velocity
            if is_idle:
                return True, "urgent_idle"
            elif (
                velocity_data and velocity_data.get("velocity", 0) > HIGH_VELOCITY * 1.5
            ):
                return True, "urgent_high_velocity"
            return False, "waiting_for_idle"

        elif strategy == "warning":
            # Only checkpoint if idle or accelerating rapidly
            if is_idle:
                return True, "warning_idle"
            elif velocity_data and velocity_data.get("trend") == "accelerating":
                if velocity_data.get("acceleration", 0) > ACCELERATION_THRESHOLD * 2:
                    return True, "rapid_acceleration"
            return False, "monitoring"

        elif strategy == "predictive":
            # Checkpoint based on prediction
            if is_idle:
                return True, "predictive_idle"
            return True, "predicted_threshold"

        return False, "no_action"

    def _is_session_idle(self, session: Dict, velocity_data: Optional[Dict]) -> bool:
        """
        Check if session is idle (good time for checkpointing)

        Args:
            session: Session dictionary
            velocity_data: Velocity information

        Returns:
            True if session appears idle
        """
        if not velocity_data:
            return False

        # Check velocity trend
        if velocity_data.get("trend") == "idle":
            return True

        # Check if velocity is very low
        if velocity_data.get("velocity", 0) < 50:  # Less than 50 tokens/minute
            return True

        # Check time since last activity
        time_diff = velocity_data.get("time_diff", 0) * 60  # Convert to seconds
        if time_diff > IDLE_THRESHOLD_SECONDS:
            return True

        return False

    def _learn_pattern(self, session: Dict, velocity_data: Dict):
        """
        Learn patterns from velocity data for better predictions

        Args:
            session: Session dictionary
            velocity_data: Velocity information
        """
        # This is where we could implement more sophisticated pattern learning
        # For now, we just log the pattern
        session_id = session.get("session_id")
        tool_id = session.get("tool_id", "unknown")
        tokens = session.get("tokens_saved", 0)

        logger.debug(
            f"Learning pattern: {tool_id} session {session_id} at {tokens} tokens, "
            f"velocity: {velocity_data.get('velocity', 0):.1f}, "
            f"trend: {velocity_data.get('trend', 'unknown')}"
        )

    async def _create_checkpoint(self, session: Dict, strategy: str = "normal"):
        """
        Create checkpoint for a session

        Args:
            session: Session dictionary from database
            strategy: Strategy that triggered the checkpoint
        """
        session_id = session.get("session_id")
        tool_id = session.get("tool_id", "unknown")
        tool_version = session.get("tool_version")

        try:
            # Generate checkpoint summary and key facts
            summary = self._generate_summary(session, strategy)
            key_facts = self._extract_key_facts(session)

            logger.info(
                f"Creating checkpoint for session {session_id} (strategy: {strategy})"
            )
            logger.debug(f"Summary: {summary}")
            logger.debug(f"Key facts: {key_facts}")

            # Store checkpoint with embedding
            checkpoint_id = await self.metrics_store.store_checkpoint_async(
                session_id=session_id,
                tool_id=tool_id,
                tool_version=tool_version,
                checkpoint_type=f"auto_{strategy}",
                summary=summary,
                key_facts=key_facts,
                decisions=[],
                patterns=[],
                files_modified=[],
                dependencies_added=[],
                commands_run=[],
                todos=[],
                blockers=[],
            )

            # Mark session as checkpointed (only at highest threshold)
            if session.get("tokens_saved", 0) >= self.token_threshold:
                self.checkpointed_sessions.add(session_id)

            # Update prediction accuracy if we have predictions
            if self.enable_predictive:
                self.metrics_store.update_prediction_accuracy(
                    session_id=session_id,
                    checkpoint_id=checkpoint_id,
                    actual_time=datetime.now().isoformat(),
                )

            logger.info(
                f"Checkpoint {checkpoint_id} created successfully for "
                f"session {session_id} ({tool_id}) with strategy {strategy}"
            )

        except Exception as e:
            logger.error(
                f"Failed to create checkpoint for session {session_id}: {e}",
                exc_info=True,
            )
            # Don't add to checkpointed_sessions so we can retry later

    def _generate_summary(self, session: Dict, strategy: str = "normal") -> str:
        """
        Generate intelligent summary from session metrics

        Args:
            session: Session dictionary from database
            strategy: Strategy that triggered the checkpoint

        Returns:
            Human-readable summary string
        """
        tool_id = session.get("tool_id", "unknown")
        tokens_saved = session.get("tokens_saved", 0)
        total_embeddings = session.get("total_embeddings", 0)
        total_compressions = session.get("total_compressions", 0)
        started_at = session.get("started_at")

        # Calculate session duration
        duration_str = "unknown duration"
        if started_at:
            try:
                start_time = datetime.fromisoformat(started_at)
                duration = datetime.now() - start_time
                hours = duration.total_seconds() / 3600
                if hours < 1:
                    minutes = duration.total_seconds() / 60
                    duration_str = f"{minutes:.0f} minutes"
                else:
                    duration_str = f"{hours:.1f} hours"
            except Exception as e:
                logger.warning(f"Failed to calculate duration: {e}")

        # Strategy-specific summary
        strategy_msg = {
            "immediate": "reached critical token threshold",
            "urgent": "approaching token limit rapidly",
            "warning": "proactively checkpointed at warning threshold",
            "predictive": "predictively checkpointed based on velocity analysis",
            "normal": "reached standard checkpoint threshold",
        }.get(strategy, "triggered checkpoint")

        # Generate summary
        summary = (
            f"{tool_id} session {strategy_msg}: {tokens_saved:,} tokens saved "
            f"across {total_embeddings} embeddings and {total_compressions} compressions "
            f"over {duration_str}. "
        )

        if strategy == "predictive":
            if session.get("session_id") in self.session_velocities:
                velocity = self.session_velocities[session.get("session_id")].get(
                    "velocity", 0
                )
                summary += f"Token velocity: {velocity:.0f} tokens/minute. "

        summary += "Auto-checkpoint created to preserve context."

        return summary

    def _extract_key_facts(self, session: Dict) -> List[str]:
        """
        Extract key facts from session metrics

        Args:
            session: Session dictionary from database

        Returns:
            List of key fact strings
        """
        facts = []

        # Session metadata
        tool_id = session.get("tool_id", "unknown")
        session_id = session.get("session_id")
        started_at = session.get("started_at")

        facts.append(f"Tool: {tool_id}")
        facts.append(f"Session ID: {session_id}")

        if started_at:
            facts.append(f"Started: {started_at}")

        # Metrics
        total_embeddings = session.get("total_embeddings", 0)
        if total_embeddings > 0:
            facts.append(f"Processed {total_embeddings:,} embeddings")

        total_compressions = session.get("total_compressions", 0)
        if total_compressions > 0:
            facts.append(f"Compressed {total_compressions} contexts")

        tokens_saved = session.get("tokens_saved", 0)
        if tokens_saved > 0:
            facts.append(f"Saved {tokens_saved:,} tokens")

        # Calculate session duration
        if started_at:
            try:
                start_time = datetime.fromisoformat(started_at)
                duration = datetime.now() - start_time
                facts.append(f"Session duration: {duration}")
            except Exception as e:
                logger.warning(f"Failed to calculate duration: {e}")

        # Add velocity information if available
        if session.get("session_id") in self.session_velocities:
            velocity_data = self.session_velocities[session.get("session_id")]
            facts.append(
                f"Token velocity: {velocity_data.get('velocity', 0):.0f} tokens/minute"
            )
            facts.append(f"Velocity trend: {velocity_data.get('trend', 'unknown')}")

        # Checkpoint trigger
        facts.append(f"Checkpoint triggered at {tokens_saved:,} tokens")

        return facts

    def get_stats(self) -> Dict:
        """
        Get monitoring statistics

        Returns:
            Dictionary with monitoring stats
        """
        stats = {
            "running": self.running,
            "token_threshold": self.token_threshold,
            "poll_interval": self.poll_interval,
            "checkpointed_sessions": len(self.checkpointed_sessions),
            "supported_tools": SUPPORTED_TOOLS,
            "predictive_enabled": self.enable_predictive,
            "monitored_sessions": len(self.session_velocities),
        }

        # Add prediction stats if available
        if self.metrics_store and self.enable_predictive:
            prediction_stats = self.metrics_store.get_prediction_stats()
            if prediction_stats:
                stats["prediction_accuracy"] = prediction_stats.get("avg_accuracy", 0)
                stats["total_predictions"] = prediction_stats.get(
                    "total_predictions", 0
                )

        return stats


async def main():
    """Main entry point for standalone execution"""
    logger.info("=" * 60)
    logger.info("OmniMemory Autonomous Checkpoint Monitor")
    logger.info("=" * 60)
    logger.info(f"Token threshold: {TOKEN_THRESHOLD:,} tokens")
    logger.info(f"Warning threshold: {WARNING_THRESHOLD:,} tokens")
    logger.info(f"Urgent threshold: {URGENT_THRESHOLD:,} tokens")
    logger.info(f"Poll interval: {POLL_INTERVAL_SECONDS} seconds")
    logger.info(f"Predictive checkpointing: {ENABLE_PREDICTIVE_CHECKPOINTING}")
    logger.info(f"Supported tools: {', '.join(SUPPORTED_TOOLS)}")
    logger.info("=" * 60)

    if not ENABLE_AUTO_CHECKPOINT:
        logger.warning("Auto-checkpoint is disabled in configuration")
        return

    monitor = CheckpointMonitor(
        token_threshold=TOKEN_THRESHOLD,
        poll_interval=POLL_INTERVAL_SECONDS,
        enable_predictive=ENABLE_PREDICTIVE_CHECKPOINTING,
    )

    try:
        await monitor.start()
    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        await monitor.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nExiting...")
        sys.exit(0)
