"""
Tests for benchmark suites
"""

import pytest
import asyncio
from src.benchmarks import LocomoBenchmark, LongMemEvalBenchmark, OmniMemoryBenchmark
from src.metrics.accuracy import AccuracyMetrics


@pytest.mark.asyncio
async def test_locomo_benchmark():
    """Test LOCOMO benchmark suite"""
    benchmark = LocomoBenchmark()
    result = await benchmark.run()

    assert result.suite_name == "LOCOMO"
    assert 0.0 <= result.overall_score <= 1.0
    assert result.tests_total > 0
    assert len(result.test_results) == result.tests_total


@pytest.mark.asyncio
async def test_longmemeval_benchmark():
    """Test LongMemEval benchmark suite"""
    benchmark = LongMemEvalBenchmark()
    result = await benchmark.run()

    assert result.suite_name == "LongMemEval"
    assert 0.0 <= result.overall_score <= 1.0
    assert result.tests_total > 0


@pytest.mark.asyncio
async def test_omnimemory_benchmark():
    """Test OmniMemory custom benchmark suite"""
    benchmark = OmniMemoryBenchmark()
    result = await benchmark.run()

    assert result.suite_name == "OmniMemory"
    assert 0.0 <= result.overall_score <= 1.0
    assert result.tests_total > 0


def test_accuracy_metrics():
    """Test accuracy metric calculations"""
    metrics = AccuracyMetrics()

    # Test precision/recall/F1
    retrieved = ["a", "b", "c"]
    relevant = ["a", "b", "d"]

    result = metrics.calculate_precision_recall_f1(retrieved, relevant)

    assert result["precision"] == 2 / 3  # 2 correct out of 3 retrieved
    assert result["recall"] == 2 / 3  # 2 correct out of 3 relevant
    assert 0.0 <= result["f1"] <= 1.0


def test_accuracy_metrics_perfect():
    """Test perfect accuracy"""
    metrics = AccuracyMetrics()

    retrieved = ["a", "b", "c"]
    relevant = ["a", "b", "c"]

    result = metrics.calculate_precision_recall_f1(retrieved, relevant)

    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_accuracy_metrics_empty():
    """Test with empty results"""
    metrics = AccuracyMetrics()

    retrieved = []
    relevant = ["a", "b"]

    result = metrics.calculate_precision_recall_f1(retrieved, relevant)

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
