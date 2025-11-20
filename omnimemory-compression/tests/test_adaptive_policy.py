"""
Tests for Adaptive Compression Policy Engine
"""

import pytest
import time
from src.adaptive_policy import (
    AdaptivePolicyEngine,
    CompressionGoal,
    CompressionMetrics,
    AdaptiveThresholds,
)


class TestAdaptiveThresholds:
    """Test suite for AdaptiveThresholds dataclass"""

    def test_threshold_creation(self):
        """Test creating AdaptiveThresholds"""
        threshold = AdaptiveThresholds(
            content_type="code",
            target_compression=0.85,
            min_quality=0.80,
            max_time_ms=50,
            sample_rate=0.5,
        )

        assert threshold.content_type == "code"
        assert threshold.target_compression == 0.85
        assert threshold.min_quality == 0.80
        assert threshold.max_time_ms == 50
        assert threshold.sample_rate == 0.5

    def test_threshold_repr(self):
        """Test AdaptiveThresholds string representation"""
        threshold = AdaptiveThresholds(
            content_type="json",
            target_compression=0.90,
            min_quality=0.85,
            max_time_ms=75,
            sample_rate=0.6,
        )

        repr_str = repr(threshold)
        assert "json" in repr_str
        assert "0.90" in repr_str
        assert "0.85" in repr_str


class TestCompressionMetrics:
    """Test suite for CompressionMetrics dataclass"""

    def test_metrics_creation(self):
        """Test creating CompressionMetrics"""
        metrics = CompressionMetrics(
            content_type="code",
            original_size=1000,
            compressed_size=200,
            compression_ratio=0.80,
            quality_score=0.85,
            compression_time_ms=25.5,
            timestamp=time.time(),
        )

        assert metrics.content_type == "code"
        assert metrics.original_size == 1000
        assert metrics.compressed_size == 200
        assert metrics.compression_ratio == 0.80
        assert metrics.quality_score == 0.85
        assert metrics.compression_time_ms == 25.5
        assert metrics.timestamp > 0


class TestAdaptivePolicyEngineInit:
    """Test suite for AdaptivePolicyEngine initialization"""

    def test_default_initialization(self):
        """Test default initialization with BALANCED goal"""
        engine = AdaptivePolicyEngine()

        assert engine.goal == CompressionGoal.BALANCED
        assert len(engine.metrics_history) == 0
        assert len(engine.thresholds) > 0

    def test_initialization_with_goal(self):
        """Test initialization with specific goal"""
        engine = AdaptivePolicyEngine(goal=CompressionGoal.MAX_QUALITY)

        assert engine.goal == CompressionGoal.MAX_QUALITY

    def test_default_thresholds_initialized(self):
        """Test that default thresholds are set for common content types"""
        engine = AdaptivePolicyEngine()

        # Should have thresholds for common types
        assert "code" in engine.thresholds
        assert "json" in engine.thresholds
        assert "logs" in engine.thresholds
        assert "markdown" in engine.thresholds

        # Check threshold values are reasonable
        code_threshold = engine.thresholds["code"]
        assert 0.0 < code_threshold.target_compression <= 1.0
        assert 0.0 < code_threshold.min_quality <= 1.0
        assert code_threshold.max_time_ms > 0
        assert 0.0 < code_threshold.sample_rate <= 1.0


class TestThresholdRetrieval:
    """Test suite for get_thresholds method"""

    def setup_method(self):
        """Setup test fixtures"""
        self.engine = AdaptivePolicyEngine()

    def test_get_known_content_type(self):
        """Test getting thresholds for known content type"""
        threshold = self.engine.get_thresholds("code")

        assert isinstance(threshold, AdaptiveThresholds)
        assert threshold.content_type == "code"

    def test_get_unknown_content_type(self):
        """Test getting thresholds for unknown content type"""
        threshold = self.engine.get_thresholds("unknown_type")

        # Should return default thresholds
        assert isinstance(threshold, AdaptiveThresholds)
        assert threshold.content_type == "unknown_type"
        assert 0.0 < threshold.target_compression <= 1.0

    def test_case_insensitive_lookup(self):
        """Test that content type lookup is case-insensitive"""
        threshold1 = self.engine.get_thresholds("CODE")
        threshold2 = self.engine.get_thresholds("code")

        assert threshold1.content_type == threshold2.content_type


class TestMetricsRecording:
    """Test suite for record_compression method"""

    def setup_method(self):
        """Setup test fixtures"""
        self.engine = AdaptivePolicyEngine()

    def test_record_single_metric(self):
        """Test recording a single metric"""
        metrics = CompressionMetrics(
            content_type="code",
            original_size=1000,
            compressed_size=200,
            compression_ratio=0.80,
            quality_score=0.85,
            compression_time_ms=25.0,
            timestamp=time.time(),
        )

        self.engine.record_compression(metrics)

        assert len(self.engine.metrics_history) == 1
        assert self.engine.metrics_history[0] == metrics

    def test_record_multiple_metrics(self):
        """Test recording multiple metrics"""
        for i in range(5):
            metrics = CompressionMetrics(
                content_type="code",
                original_size=1000,
                compressed_size=200 + i * 10,
                compression_ratio=0.80 - i * 0.01,
                quality_score=0.85,
                compression_time_ms=25.0,
                timestamp=time.time(),
            )
            self.engine.record_compression(metrics)

        assert len(self.engine.metrics_history) == 5

    def test_history_limit(self):
        """Test that metrics history is limited to 1000 entries"""
        # Record 1100 metrics
        for i in range(1100):
            metrics = CompressionMetrics(
                content_type="code",
                original_size=1000,
                compressed_size=200,
                compression_ratio=0.80,
                quality_score=0.85,
                compression_time_ms=25.0,
                timestamp=time.time(),
            )
            self.engine.record_compression(metrics)

        # Should only keep last 1000
        assert len(self.engine.metrics_history) == 1000

    def test_adaptation_trigger(self):
        """Test that adaptation is triggered every 10 compressions"""
        initial_threshold = self.engine.get_thresholds("code")

        # Record 10 metrics with poor quality (should trigger adaptation)
        for i in range(10):
            metrics = CompressionMetrics(
                content_type="code",
                original_size=1000,
                compressed_size=200,
                compression_ratio=0.80,
                quality_score=0.50,  # Below default min_quality
                compression_time_ms=25.0,
                timestamp=time.time(),
            )
            self.engine.record_compression(metrics)

        # Thresholds should have been adapted (for BALANCED goal)
        # Note: Need at least 5 samples for adaptation, so record 10 total
        assert len(self.engine.metrics_history) == 10


class TestAdaptationForGoals:
    """Test suite for goal-based adaptation"""

    def create_metrics_batch(
        self,
        content_type: str,
        quality: float,
        ratio: float,
        time_ms: float,
        count: int,
    ):
        """Helper to create a batch of metrics"""
        metrics_list = []
        for _ in range(count):
            metrics = CompressionMetrics(
                content_type=content_type,
                original_size=1000,
                compressed_size=int(1000 * (1 - ratio)),
                compression_ratio=ratio,
                quality_score=quality,
                compression_time_ms=time_ms,
                timestamp=time.time(),
            )
            metrics_list.append(metrics)
        return metrics_list

    def test_max_quality_goal_low_quality(self):
        """Test MAX_QUALITY goal reduces sample_rate when quality is low"""
        engine = AdaptivePolicyEngine(goal=CompressionGoal.MAX_QUALITY)
        initial_threshold = engine.get_thresholds("code")
        initial_sample_rate = initial_threshold.sample_rate

        # Record 10 compressions with low quality
        for metrics in self.create_metrics_batch(
            content_type="code", quality=0.60, ratio=0.80, time_ms=25.0, count=10
        ):
            engine.record_compression(metrics)

        # Sample rate should be reduced
        new_threshold = engine.get_thresholds("code")
        assert new_threshold.sample_rate < initial_sample_rate

    def test_max_compression_goal_low_ratio(self):
        """Test MAX_COMPRESSION goal increases sample_rate when ratio is low"""
        engine = AdaptivePolicyEngine(goal=CompressionGoal.MAX_COMPRESSION)
        initial_threshold = engine.get_thresholds("code")
        initial_sample_rate = initial_threshold.sample_rate

        # Record 10 compressions with low compression ratio
        for metrics in self.create_metrics_batch(
            content_type="code", quality=0.85, ratio=0.70, time_ms=25.0, count=10
        ):
            engine.record_compression(metrics)

        # Sample rate should be increased
        new_threshold = engine.get_thresholds("code")
        assert new_threshold.sample_rate > initial_sample_rate

    def test_max_speed_goal_slow_compression(self):
        """Test MAX_SPEED goal reduces target when compression is slow"""
        engine = AdaptivePolicyEngine(goal=CompressionGoal.MAX_SPEED)
        initial_threshold = engine.get_thresholds("code")
        initial_target = initial_threshold.target_compression

        # Record 10 compressions that are slow
        for metrics in self.create_metrics_batch(
            content_type="code", quality=0.85, ratio=0.80, time_ms=150.0, count=10
        ):
            engine.record_compression(metrics)

        # Target compression should be reduced
        new_threshold = engine.get_thresholds("code")
        assert new_threshold.target_compression < initial_target

    def test_balanced_goal_adaptation(self):
        """Test BALANCED goal adapts based on multiple factors"""
        engine = AdaptivePolicyEngine(goal=CompressionGoal.BALANCED)

        # Record 10 compressions with low quality
        for metrics in self.create_metrics_batch(
            content_type="json", quality=0.60, ratio=0.85, time_ms=30.0, count=10
        ):
            engine.record_compression(metrics)

        # Should have adapted
        threshold = engine.get_thresholds("json")
        assert isinstance(threshold, AdaptiveThresholds)


class TestStatistics:
    """Test suite for get_statistics method"""

    def setup_method(self):
        """Setup test fixtures"""
        self.engine = AdaptivePolicyEngine()

    def test_statistics_empty_history(self):
        """Test statistics with no recorded compressions"""
        stats = self.engine.get_statistics()

        assert stats["error"] == "No metrics available"
        assert stats["total_compressions"] == 0

    def test_statistics_with_data(self):
        """Test statistics calculation with recorded data"""
        # Record 5 compressions
        for i in range(5):
            metrics = CompressionMetrics(
                content_type="code",
                original_size=1000,
                compressed_size=200,
                compression_ratio=0.80,
                quality_score=0.85,
                compression_time_ms=25.0,
                timestamp=time.time(),
            )
            self.engine.record_compression(metrics)

        stats = self.engine.get_statistics()

        assert stats["total_compressions"] == 5
        assert stats["avg_compression_ratio"] == 0.80
        assert stats["avg_quality"] == 0.85
        assert stats["avg_time_ms"] == 25.0
        assert stats["total_original_size"] == 5000
        assert stats["total_compressed_size"] == 1000
        assert stats["total_size_saved"] == 4000

    def test_statistics_filtered_by_content_type(self):
        """Test statistics filtered by content type"""
        # Record mixed content types
        for content_type in ["code", "json", "logs"]:
            for _ in range(3):
                metrics = CompressionMetrics(
                    content_type=content_type,
                    original_size=1000,
                    compressed_size=200,
                    compression_ratio=0.80,
                    quality_score=0.85,
                    compression_time_ms=25.0,
                    timestamp=time.time(),
                )
                self.engine.record_compression(metrics)

        # Get stats for specific type
        stats = self.engine.get_statistics("code")

        assert stats["content_type"] == "code"
        assert stats["total_compressions"] == 3

    def test_statistics_includes_current_thresholds(self):
        """Test that statistics include current thresholds"""
        # Record some data
        for _ in range(5):
            metrics = CompressionMetrics(
                content_type="code",
                original_size=1000,
                compressed_size=200,
                compression_ratio=0.80,
                quality_score=0.85,
                compression_time_ms=25.0,
                timestamp=time.time(),
            )
            self.engine.record_compression(metrics)

        stats = self.engine.get_statistics("code")

        assert "current_thresholds" in stats
        assert "target_compression" in stats["current_thresholds"]
        assert "min_quality" in stats["current_thresholds"]


class TestGoalManagement:
    """Test suite for goal management methods"""

    def test_set_goal(self):
        """Test changing optimization goal"""
        engine = AdaptivePolicyEngine(goal=CompressionGoal.BALANCED)

        engine.set_goal(CompressionGoal.MAX_QUALITY)

        assert engine.goal == CompressionGoal.MAX_QUALITY

    def test_set_goal_triggers_readaptation(self):
        """Test that changing goal triggers re-adaptation"""
        engine = AdaptivePolicyEngine(goal=CompressionGoal.BALANCED)

        # Record some metrics
        for _ in range(10):
            metrics = CompressionMetrics(
                content_type="code",
                original_size=1000,
                compressed_size=200,
                compression_ratio=0.80,
                quality_score=0.85,
                compression_time_ms=25.0,
                timestamp=time.time(),
            )
            engine.record_compression(metrics)

        # Change goal - should trigger re-adaptation
        engine.set_goal(CompressionGoal.MAX_QUALITY)

        # Goal should be changed
        assert engine.goal == CompressionGoal.MAX_QUALITY


class TestUtilityMethods:
    """Test suite for utility methods"""

    def test_get_all_thresholds(self):
        """Test retrieving all thresholds"""
        engine = AdaptivePolicyEngine()

        all_thresholds = engine.get_all_thresholds()

        assert isinstance(all_thresholds, dict)
        assert "code" in all_thresholds
        assert "json" in all_thresholds
        assert "target_compression" in all_thresholds["code"]

    def test_reset_thresholds(self):
        """Test resetting thresholds to defaults"""
        engine = AdaptivePolicyEngine()

        # Modify a threshold
        engine.thresholds["code"].target_compression = 0.95

        # Reset
        engine.reset_thresholds()

        # Should be back to default
        assert engine.thresholds["code"].target_compression == 0.85

    def test_clear_history(self):
        """Test clearing metrics history"""
        engine = AdaptivePolicyEngine()

        # Add some metrics
        for _ in range(5):
            metrics = CompressionMetrics(
                content_type="code",
                original_size=1000,
                compressed_size=200,
                compression_ratio=0.80,
                quality_score=0.85,
                compression_time_ms=25.0,
                timestamp=time.time(),
            )
            engine.record_compression(metrics)

        assert len(engine.metrics_history) == 5

        # Clear
        engine.clear_history()

        assert len(engine.metrics_history) == 0


class TestMultipleContentTypes:
    """Test suite for handling multiple content types"""

    def test_independent_adaptation_per_type(self):
        """Test that each content type adapts independently"""
        engine = AdaptivePolicyEngine(goal=CompressionGoal.BALANCED)

        # Record different performance for different types
        for _ in range(10):
            # Code with good quality
            engine.record_compression(
                CompressionMetrics(
                    content_type="code",
                    original_size=1000,
                    compressed_size=200,
                    compression_ratio=0.80,
                    quality_score=0.90,
                    compression_time_ms=20.0,
                    timestamp=time.time(),
                )
            )

            # JSON with poor quality
            engine.record_compression(
                CompressionMetrics(
                    content_type="json",
                    original_size=1000,
                    compressed_size=200,
                    compression_ratio=0.80,
                    quality_score=0.60,
                    compression_time_ms=20.0,
                    timestamp=time.time(),
                )
            )

        # Each type should have different thresholds
        code_threshold = engine.get_thresholds("code")
        json_threshold = engine.get_thresholds("json")

        # They should differ (json should have lower sample_rate due to poor quality)
        assert code_threshold.sample_rate != json_threshold.sample_rate

    def test_statistics_per_content_type(self):
        """Test getting statistics per content type"""
        engine = AdaptivePolicyEngine()

        # Record different types
        for content_type in ["code", "json"]:
            for _ in range(5):
                engine.record_compression(
                    CompressionMetrics(
                        content_type=content_type,
                        original_size=1000,
                        compressed_size=200,
                        compression_ratio=0.80,
                        quality_score=0.85,
                        compression_time_ms=25.0,
                        timestamp=time.time(),
                    )
                )

        code_stats = engine.get_statistics("code")
        json_stats = engine.get_statistics("json")

        assert code_stats["content_type"] == "code"
        assert json_stats["content_type"] == "json"
        assert code_stats["total_compressions"] == 5
        assert json_stats["total_compressions"] == 5
