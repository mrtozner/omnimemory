"""
Base Benchmark Class
Provides common interface for all benchmark suites
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark test case"""

    test_name: str
    passed: bool
    score: float  # 0.0 to 1.0
    metrics: Dict[str, Any]
    error: Optional[str] = None
    duration_ms: Optional[float] = None


@dataclass
class BenchmarkSuiteResult:
    """Result from running an entire benchmark suite"""

    suite_name: str
    timestamp: datetime
    overall_score: float  # 0.0 to 1.0
    test_results: List[BenchmarkResult]
    config: Dict[str, Any]
    summary: Dict[str, Any]

    @property
    def tests_passed(self) -> int:
        return sum(1 for r in self.test_results if r.passed)

    @property
    def tests_total(self) -> int:
        return len(self.test_results)

    @property
    def pass_rate(self) -> float:
        if self.tests_total == 0:
            return 0.0
        return self.tests_passed / self.tests_total


class BaseBenchmark(ABC):
    """Base class for all benchmark suites"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"benchmark.{name}")

    @abstractmethod
    async def run(
        self, config: Optional[Dict[str, Any]] = None
    ) -> BenchmarkSuiteResult:
        """
        Run the benchmark suite

        Args:
            config: Optional configuration for the benchmark

        Returns:
            BenchmarkSuiteResult with all test results
        """
        pass

    @abstractmethod
    def get_test_cases(self) -> List[str]:
        """Get list of test case names in this benchmark"""
        pass

    def _create_result(
        self,
        test_results: List[BenchmarkResult],
        config: Dict[str, Any],
    ) -> BenchmarkSuiteResult:
        """Helper to create benchmark suite result"""
        # Calculate overall score (average of all test scores)
        if test_results:
            overall_score = sum(r.score for r in test_results) / len(test_results)
        else:
            overall_score = 0.0

        # Create summary
        summary = {
            "tests_passed": sum(1 for r in test_results if r.passed),
            "tests_total": len(test_results),
            "pass_rate": (
                sum(1 for r in test_results if r.passed) / len(test_results)
                if test_results
                else 0.0
            ),
            "avg_score": overall_score,
            "avg_duration_ms": (
                sum(r.duration_ms for r in test_results if r.duration_ms)
                / len([r for r in test_results if r.duration_ms])
                if any(r.duration_ms for r in test_results)
                else None
            ),
        }

        return BenchmarkSuiteResult(
            suite_name=self.name,
            timestamp=datetime.now(),
            overall_score=overall_score,
            test_results=test_results,
            config=config,
            summary=summary,
        )
