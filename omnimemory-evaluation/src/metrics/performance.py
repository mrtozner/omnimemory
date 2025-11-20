"""
Performance Metrics
Measures latency, throughput, and other performance characteristics
"""

import time
import asyncio
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
import logging
import statistics

logger = logging.getLogger(__name__)


@dataclass
class LatencyStats:
    """Statistics for latency measurements"""

    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    count: int


class PerformanceMetrics:
    """Measure performance characteristics"""

    @staticmethod
    async def measure_latency(
        operation: Callable,
        *args,
        **kwargs,
    ) -> tuple[Any, float]:
        """
        Measure latency of an async operation

        Args:
            operation: Async function to measure
            *args, **kwargs: Arguments to pass to operation

        Returns:
            Tuple of (result, latency_ms)
        """
        start = time.perf_counter()
        result = await operation(*args, **kwargs)
        end = time.perf_counter()
        latency_ms = (end - start) * 1000
        return result, latency_ms

    @staticmethod
    async def measure_multiple_latencies(
        operation: Callable,
        iterations: int,
        *args,
        **kwargs,
    ) -> LatencyStats:
        """
        Measure latency over multiple iterations

        Args:
            operation: Async function to measure
            iterations: Number of times to run
            *args, **kwargs: Arguments to pass to operation

        Returns:
            LatencyStats with aggregated measurements
        """
        latencies = []

        for _ in range(iterations):
            _, latency = await PerformanceMetrics.measure_latency(
                operation, *args, **kwargs
            )
            latencies.append(latency)

        latencies.sort()
        return LatencyStats(
            min_ms=min(latencies),
            max_ms=max(latencies),
            mean_ms=statistics.mean(latencies),
            median_ms=statistics.median(latencies),
            p95_ms=latencies[int(len(latencies) * 0.95)],
            p99_ms=latencies[int(len(latencies) * 0.99)],
            count=len(latencies),
        )

    @staticmethod
    async def measure_throughput(
        operation: Callable,
        duration_seconds: float = 10.0,
        *args,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Measure throughput (operations per second)

        Args:
            operation: Async function to measure
            duration_seconds: How long to measure for
            *args, **kwargs: Arguments to pass to operation

        Returns:
            Dict with throughput metrics
        """
        start = time.perf_counter()
        end_time = start + duration_seconds
        count = 0
        errors = 0

        while time.perf_counter() < end_time:
            try:
                await operation(*args, **kwargs)
                count += 1
            except Exception as e:
                logger.error(f"Error in throughput test: {e}")
                errors += 1

        elapsed = time.perf_counter() - start
        ops_per_second = count / elapsed if elapsed > 0 else 0

        return {
            "ops_per_second": ops_per_second,
            "total_operations": count,
            "duration_seconds": elapsed,
            "errors": errors,
            "error_rate": errors / (count + errors) if (count + errors) > 0 else 0,
        }

    @staticmethod
    async def measure_concurrent_latency(
        operation: Callable,
        concurrency: int,
        iterations_per_worker: int,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Measure latency under concurrent load

        Args:
            operation: Async function to measure
            concurrency: Number of concurrent workers
            iterations_per_worker: Iterations per worker
            *args, **kwargs: Arguments to pass to operation

        Returns:
            Dict with concurrent latency stats
        """

        async def worker():
            latencies = []
            for _ in range(iterations_per_worker):
                _, latency = await PerformanceMetrics.measure_latency(
                    operation, *args, **kwargs
                )
                latencies.append(latency)
            return latencies

        # Run workers concurrently
        start = time.perf_counter()
        results = await asyncio.gather(*[worker() for _ in range(concurrency)])
        total_time = time.perf_counter() - start

        # Aggregate all latencies
        all_latencies = []
        for worker_latencies in results:
            all_latencies.extend(worker_latencies)

        all_latencies.sort()
        total_ops = len(all_latencies)

        return {
            "concurrency": concurrency,
            "total_operations": total_ops,
            "total_time_seconds": total_time,
            "throughput_ops_per_second": total_ops / total_time
            if total_time > 0
            else 0,
            "latency_stats": {
                "min_ms": min(all_latencies),
                "max_ms": max(all_latencies),
                "mean_ms": statistics.mean(all_latencies),
                "median_ms": statistics.median(all_latencies),
                "p95_ms": all_latencies[int(len(all_latencies) * 0.95)],
                "p99_ms": all_latencies[int(len(all_latencies) * 0.99)],
            },
        }

    @staticmethod
    def calculate_token_savings(
        original_tokens: int,
        compressed_tokens: int,
    ) -> Dict[str, float]:
        """
        Calculate token savings metrics

        Args:
            original_tokens: Number of tokens before compression
            compressed_tokens: Number of tokens after compression

        Returns:
            Dict with savings metrics
        """
        tokens_saved = original_tokens - compressed_tokens
        compression_ratio = (
            compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        )
        savings_pct = (
            (tokens_saved / original_tokens * 100) if original_tokens > 0 else 0
        )

        return {
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "tokens_saved": tokens_saved,
            "compression_ratio": compression_ratio,
            "savings_percentage": savings_pct,
        }
