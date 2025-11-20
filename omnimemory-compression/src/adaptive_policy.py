"""
Adaptive Compression Policy Engine
Dynamically adjusts compression thresholds based on historical performance metrics
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)


class CompressionGoal(Enum):
    """Optimization goals for adaptive compression"""

    MAX_QUALITY = "max_quality"  # Prioritize quality (slower, less compression)
    BALANCED = "balanced"  # Balance quality, speed, and compression
    MAX_COMPRESSION = "max_compression"  # Prioritize compression ratio
    MAX_SPEED = "max_speed"  # Prioritize speed (less compression)


@dataclass
class CompressionMetrics:
    """Metrics from a compression operation"""

    content_type: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    quality_score: float
    compression_time_ms: float
    timestamp: float


@dataclass
class AdaptiveThresholds:
    """Dynamic compression thresholds per content type"""

    content_type: str
    target_compression: float  # Target compression ratio (0.0-1.0)
    min_quality: float  # Minimum quality threshold (0.0-1.0)
    max_time_ms: float  # Maximum compression time
    sample_rate: float  # How aggressively to sample (0.0-1.0)

    def __repr__(self):
        return (
            f"AdaptiveThresholds(type={self.content_type}, "
            f"target={self.target_compression:.2f}, "
            f"min_quality={self.min_quality:.2f}, "
            f"max_time={self.max_time_ms:.1f}ms, "
            f"sample_rate={self.sample_rate:.2f})"
        )


class AdaptivePolicyEngine:
    """
    Dynamically adjusts compression policies based on:
    - Historical compression metrics
    - Content type characteristics
    - User-defined goals (quality, speed, compression)
    - Real-time performance data
    """

    def __init__(self, goal: CompressionGoal = CompressionGoal.BALANCED):
        """
        Initialize adaptive policy engine

        Args:
            goal: Optimization goal (quality, speed, compression, or balanced)
        """
        self.goal = goal
        self.metrics_history: List[CompressionMetrics] = []
        self.thresholds: Dict[str, AdaptiveThresholds] = {}
        self._initialize_default_thresholds()

        logger.info(f"AdaptivePolicyEngine initialized with goal: {goal.value}")

    def _initialize_default_thresholds(self):
        """
        Set initial thresholds for each content type

        These are conservative defaults that will be adapted based on actual performance.
        """
        defaults = {
            "code": AdaptiveThresholds(
                content_type="code",
                target_compression=0.85,
                min_quality=0.85,
                max_time_ms=50,
                sample_rate=0.5,
            ),
            "json": AdaptiveThresholds(
                content_type="json",
                target_compression=0.88,
                min_quality=0.88,
                max_time_ms=50,
                sample_rate=0.6,
            ),
            "logs": AdaptiveThresholds(
                content_type="logs",
                target_compression=0.90,
                min_quality=0.90,
                max_time_ms=50,
                sample_rate=0.7,
            ),
            "markdown": AdaptiveThresholds(
                content_type="markdown",
                target_compression=0.80,
                min_quality=0.80,
                max_time_ms=50,
                sample_rate=0.4,
            ),
            "text": AdaptiveThresholds(
                content_type="text",
                target_compression=0.75,
                min_quality=0.75,
                max_time_ms=100,
                sample_rate=0.5,
            ),
            "config": AdaptiveThresholds(
                content_type="config",
                target_compression=0.70,
                min_quality=0.70,
                max_time_ms=50,
                sample_rate=0.4,
            ),
            "data": AdaptiveThresholds(
                content_type="data",
                target_compression=0.85,
                min_quality=0.85,
                max_time_ms=75,
                sample_rate=0.6,
            ),
        }

        self.thresholds = defaults
        logger.debug(f"Initialized {len(defaults)} default threshold configurations")

    def get_thresholds(self, content_type: str) -> AdaptiveThresholds:
        """
        Get current thresholds for content type

        Args:
            content_type: Content type identifier

        Returns:
            AdaptiveThresholds for the content type (or default if unknown)
        """
        # Normalize content type
        content_type_normalized = content_type.lower().replace("_", "")

        # Try exact match first
        if content_type_normalized in self.thresholds:
            return self.thresholds[content_type_normalized]

        # Try partial match
        for key in self.thresholds:
            if key in content_type_normalized or content_type_normalized in key:
                return self.thresholds[key]

        # Return generic default for unknown types
        logger.debug(f"Unknown content type '{content_type}', using generic defaults")
        return AdaptiveThresholds(
            content_type=content_type,
            target_compression=0.70,
            min_quality=0.70,
            max_time_ms=100,
            sample_rate=0.5,
        )

    def record_compression(self, metrics: CompressionMetrics):
        """
        Record compression metrics for learning

        Args:
            metrics: CompressionMetrics from a compression operation
        """
        self.metrics_history.append(metrics)

        # Keep only last 1000 metrics to prevent unbounded growth
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
            logger.debug("Trimmed metrics history to last 1000 entries")

        # Trigger adaptation after every 10 compressions
        if len(self.metrics_history) % 10 == 0:
            self._adapt_thresholds(metrics.content_type)
            logger.info(
                f"Adaptation triggered for {metrics.content_type} "
                f"(total metrics: {len(self.metrics_history)})"
            )

    def _adapt_thresholds(self, content_type: str):
        """
        Adapt thresholds based on recent performance

        Strategy:
        - If consistently hitting quality targets but slow → increase sample_rate
        - If quality dropping → decrease sample_rate, increase min_quality
        - If compression ratio low → adjust target_compression
        - If consistently fast → can increase compression target

        Args:
            content_type: Content type to adapt thresholds for
        """
        # Get recent metrics for this content type (last 100)
        recent = [
            m for m in self.metrics_history[-100:] if m.content_type == content_type
        ]

        if len(recent) < 5:
            logger.debug(
                f"Insufficient data for adaptation of {content_type}: "
                f"only {len(recent)} samples"
            )
            return  # Need more data

        current = self.thresholds.get(content_type)
        if not current:
            logger.debug(
                f"No current thresholds for {content_type}, skipping adaptation"
            )
            return

        # Calculate averages
        avg_quality = sum(m.quality_score for m in recent) / len(recent)
        avg_ratio = sum(m.compression_ratio for m in recent) / len(recent)
        avg_time = sum(m.compression_time_ms for m in recent) / len(recent)

        logger.debug(
            f"Adapting {content_type}: "
            f"avg_quality={avg_quality:.3f}, "
            f"avg_ratio={avg_ratio:.3f}, "
            f"avg_time={avg_time:.1f}ms "
            f"(from {len(recent)} samples)"
        )

        # Adapt based on goal
        new_thresholds = self._adapt_for_goal(current, avg_quality, avg_ratio, avg_time)

        # Log changes if significant
        if (
            abs(new_thresholds.sample_rate - current.sample_rate) > 0.01
            or abs(new_thresholds.target_compression - current.target_compression)
            > 0.01
        ):
            logger.info(
                f"Adapted thresholds for {content_type}: "
                f"sample_rate {current.sample_rate:.2f} → {new_thresholds.sample_rate:.2f}, "
                f"target_compression {current.target_compression:.2f} → {new_thresholds.target_compression:.2f}"
            )

        self.thresholds[content_type] = new_thresholds

    def _adapt_for_goal(
        self,
        current: AdaptiveThresholds,
        avg_quality: float,
        avg_ratio: float,
        avg_time: float,
    ) -> AdaptiveThresholds:
        """
        Adjust thresholds based on optimization goal

        Args:
            current: Current threshold configuration
            avg_quality: Average quality score from recent compressions
            avg_ratio: Average compression ratio from recent compressions
            avg_time: Average compression time from recent compressions

        Returns:
            New AdaptiveThresholds with adjusted values
        """
        # Create new thresholds (copy current values)
        new = AdaptiveThresholds(
            content_type=current.content_type,
            target_compression=current.target_compression,
            min_quality=current.min_quality,
            max_time_ms=current.max_time_ms,
            sample_rate=current.sample_rate,
        )

        if self.goal == CompressionGoal.MAX_QUALITY:
            # Prioritize quality over compression
            if avg_quality < current.min_quality:
                # Quality is below target - reduce sample rate to improve quality
                new.sample_rate = max(0.1, current.sample_rate - 0.05)
                logger.debug(
                    f"MAX_QUALITY: Quality below target "
                    f"({avg_quality:.3f} < {current.min_quality:.3f}), "
                    f"reducing sample_rate to {new.sample_rate:.2f}"
                )
            elif avg_quality > current.min_quality + 0.05:
                # Quality is significantly above target - can increase sample rate
                new.sample_rate = min(0.9, current.sample_rate + 0.02)
                logger.debug(
                    f"MAX_QUALITY: Quality above target, "
                    f"increasing sample_rate to {new.sample_rate:.2f}"
                )

        elif self.goal == CompressionGoal.MAX_COMPRESSION:
            # Prioritize compression ratio
            if avg_ratio < current.target_compression:
                # Not achieving target compression - increase sample rate
                new.sample_rate = min(0.9, current.sample_rate + 0.05)
                logger.debug(
                    f"MAX_COMPRESSION: Ratio below target "
                    f"({avg_ratio:.3f} < {current.target_compression:.3f}), "
                    f"increasing sample_rate to {new.sample_rate:.2f}"
                )
            elif avg_quality < current.min_quality - 0.1:
                # Quality dropping too much - reduce sample rate slightly
                new.sample_rate = max(0.3, current.sample_rate - 0.02)
                logger.debug(
                    f"MAX_COMPRESSION: Quality too low, "
                    f"reducing sample_rate to {new.sample_rate:.2f}"
                )

        elif self.goal == CompressionGoal.MAX_SPEED:
            # Prioritize speed
            if avg_time > current.max_time_ms:
                # Taking too long - reduce compression target
                new.target_compression = max(0.5, current.target_compression - 0.05)
                new.sample_rate = min(0.8, current.sample_rate + 0.05)
                logger.debug(
                    f"MAX_SPEED: Time above target "
                    f"({avg_time:.1f}ms > {current.max_time_ms:.1f}ms), "
                    f"reducing target_compression to {new.target_compression:.2f}"
                )
            elif avg_time < current.max_time_ms * 0.5:
                # Running fast - can increase compression target
                new.target_compression = min(0.95, current.target_compression + 0.03)
                logger.debug(
                    f"MAX_SPEED: Running fast, "
                    f"increasing target_compression to {new.target_compression:.2f}"
                )

        else:  # BALANCED
            # Balance all factors
            if avg_quality < current.min_quality - 0.05:
                # Quality significantly below target
                new.sample_rate = max(0.2, current.sample_rate - 0.03)
                logger.debug(
                    f"BALANCED: Quality below target, "
                    f"reducing sample_rate to {new.sample_rate:.2f}"
                )
            elif avg_ratio < current.target_compression - 0.05:
                # Compression ratio below target
                new.sample_rate = min(0.8, current.sample_rate + 0.03)
                logger.debug(
                    f"BALANCED: Ratio below target, "
                    f"increasing sample_rate to {new.sample_rate:.2f}"
                )
            elif avg_time > current.max_time_ms * 1.5:
                # Time significantly above target
                new.target_compression = max(0.6, current.target_compression - 0.03)
                logger.debug(
                    f"BALANCED: Time above target, "
                    f"reducing target_compression to {new.target_compression:.2f}"
                )

        return new

    def get_statistics(self, content_type: Optional[str] = None) -> Dict:
        """
        Get performance statistics

        Args:
            content_type: Optional content type to filter by

        Returns:
            Dictionary with statistics
        """
        if content_type:
            metrics = [
                m for m in self.metrics_history if m.content_type == content_type
            ]
        else:
            metrics = self.metrics_history

        if not metrics:
            return {
                "error": "No metrics available",
                "content_type": content_type,
                "total_compressions": 0,
            }

        total_original = sum(m.original_size for m in metrics)
        total_compressed = sum(m.compressed_size for m in metrics)

        stats = {
            "content_type": content_type or "all",
            "total_compressions": len(metrics),
            "avg_compression_ratio": sum(m.compression_ratio for m in metrics)
            / len(metrics),
            "avg_quality": sum(m.quality_score for m in metrics) / len(metrics),
            "avg_time_ms": sum(m.compression_time_ms for m in metrics) / len(metrics),
            "total_original_size": total_original,
            "total_compressed_size": total_compressed,
            "total_size_saved": total_original - total_compressed,
            "optimization_goal": self.goal.value,
        }

        # Add current thresholds for the content type
        if content_type and content_type in self.thresholds:
            threshold = self.thresholds[content_type]
            stats["current_thresholds"] = {
                "target_compression": threshold.target_compression,
                "min_quality": threshold.min_quality,
                "max_time_ms": threshold.max_time_ms,
                "sample_rate": threshold.sample_rate,
            }

        return stats

    def get_all_thresholds(self) -> Dict[str, Dict]:
        """
        Get all current thresholds for all content types

        Returns:
            Dictionary mapping content types to threshold configurations
        """
        return {
            content_type: {
                "target_compression": threshold.target_compression,
                "min_quality": threshold.min_quality,
                "max_time_ms": threshold.max_time_ms,
                "sample_rate": threshold.sample_rate,
            }
            for content_type, threshold in self.thresholds.items()
        }

    def set_goal(self, goal: CompressionGoal):
        """
        Change the optimization goal

        Args:
            goal: New optimization goal
        """
        old_goal = self.goal
        self.goal = goal
        logger.info(f"Changed optimization goal: {old_goal.value} → {goal.value}")

        # Trigger re-adaptation for all content types with recent data
        content_types = set(m.content_type for m in self.metrics_history[-100:])
        for content_type in content_types:
            self._adapt_thresholds(content_type)

    def reset_thresholds(self):
        """
        Reset all thresholds to defaults

        Useful for testing or recovering from poor adaptations
        """
        logger.info("Resetting all thresholds to defaults")
        self._initialize_default_thresholds()

    def clear_history(self):
        """
        Clear metrics history

        Useful for starting fresh or testing
        """
        logger.info(f"Clearing {len(self.metrics_history)} metrics from history")
        self.metrics_history.clear()
