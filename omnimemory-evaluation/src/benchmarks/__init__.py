"""Benchmark suites for OmniMemory evaluation"""

from .base import BaseBenchmark, BenchmarkResult, BenchmarkSuiteResult
from .locomo import LocomoBenchmark
from .longmemeval import LongMemEvalBenchmark
from .omnimemory import OmniMemoryBenchmark

__all__ = [
    "BaseBenchmark",
    "BenchmarkResult",
    "BenchmarkSuiteResult",
    "LocomoBenchmark",
    "LongMemEvalBenchmark",
    "OmniMemoryBenchmark",
]
