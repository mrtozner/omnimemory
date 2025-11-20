"""
FastAPI Evaluation Server
Comprehensive evaluation framework for OmniMemory
Runs benchmarks, A/B tests, and performance regression detection
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import logging
from contextlib import asynccontextmanager
import httpx
import uuid
from datetime import datetime

from .data_store import EvaluationStore
from .benchmarks import LocomoBenchmark, LongMemEvalBenchmark, OmniMemoryBenchmark
from .metrics.accuracy import AccuracyMetrics
from .metrics.performance import PerformanceMetrics
from .metrics.quality import QualityMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Service URLs
COMPRESSION_URL = "http://localhost:8001"
EMBEDDINGS_URL = "http://localhost:8000"
METRICS_URL = "http://localhost:8003"

# Global state
eval_store: Optional[EvaluationStore] = None
benchmarks: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global eval_store, benchmarks

    # Startup
    logger.info("Starting OmniMemory Evaluation Service...")
    eval_store = EvaluationStore()

    # Initialize benchmarks
    benchmarks = {
        "locomo": LocomoBenchmark(COMPRESSION_URL, EMBEDDINGS_URL, METRICS_URL),
        "longmemeval": LongMemEvalBenchmark(),
        "omnimemory": OmniMemoryBenchmark(),
    }

    logger.info("Evaluation service initialized")
    yield

    # Shutdown
    logger.info("Shutting down Evaluation Service...")
    if eval_store:
        eval_store.close()
    logger.info("Service shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="OmniMemory Evaluation Service",
    description="Comprehensive evaluation framework with benchmarks and A/B testing",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class MemoryEvaluationRequest(BaseModel):
    """Request for evaluating memory operations"""

    retrieved_memories: List[str] = Field(..., description="Retrieved memory IDs")
    relevant_memories: List[str] = Field(..., description="Ground truth relevant IDs")
    test_set: str = Field(..., description="Test set name")
    strategy: Optional[str] = Field(None, description="Memory strategy used")


class BenchmarkRunRequest(BaseModel):
    """Request to run a benchmark suite"""

    config: Optional[Dict[str, Any]] = Field(
        None, description="Benchmark configuration"
    )


class ABTestStartRequest(BaseModel):
    """Request to start an A/B test"""

    test_name: str = Field(..., description="Name of the test")
    variant_a: str = Field(..., description="Description of variant A")
    variant_b: str = Field(..., description="Description of variant B")
    metric_name: str = Field(..., description="Metric to compare")


class ABTestUpdateRequest(BaseModel):
    """Request to update A/B test with results"""

    variant: str = Field(..., description="Variant name (a or b)")
    value: float = Field(..., description="Metric value")


class RegressionCheckRequest(BaseModel):
    """Request to check for performance regression"""

    metric_name: str = Field(..., description="Metric name")
    current_value: float = Field(..., description="Current value")
    threshold_pct: Optional[float] = Field(5.0, description="Threshold percentage")


# API Endpoints


@app.get("/")
async def root():
    """Health check"""
    return {
        "service": "OmniMemory Evaluation Service",
        "status": "running",
        "version": "1.0.0",
    }


@app.post("/evaluate/memory")
async def evaluate_memory(request: MemoryEvaluationRequest):
    """
    Evaluate memory retrieval accuracy

    Calculates precision, recall, F1, and other accuracy metrics
    """
    try:
        accuracy_metrics = AccuracyMetrics()

        # Calculate precision, recall, F1
        metrics = accuracy_metrics.calculate_precision_recall_f1(
            request.retrieved_memories,
            request.relevant_memories,
        )

        # Store in database
        eval_store.store_memory_accuracy(
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            test_set=request.test_set,
            strategy=request.strategy,
            metadata={
                "retrieved_count": len(request.retrieved_memories),
                "relevant_count": len(request.relevant_memories),
            },
        )

        logger.info(
            f"Memory evaluation: F1={metrics['f1']:.3f}, "
            f"Precision={metrics['precision']:.3f}, "
            f"Recall={metrics['recall']:.3f}"
        )

        return {
            "metrics": metrics,
            "test_set": request.test_set,
            "strategy": request.strategy,
        }

    except Exception as e:
        logger.error(f"Memory evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/benchmark/run/{suite}")
async def run_benchmark(suite: str, request: BenchmarkRunRequest):
    """
    Run a specific benchmark suite

    Available suites: locomo, longmemeval, omnimemory
    """
    try:
        if suite not in benchmarks:
            raise HTTPException(
                status_code=404,
                detail=f"Benchmark '{suite}' not found. Available: {list(benchmarks.keys())}",
            )

        benchmark = benchmarks[suite]
        logger.info(f"Running benchmark: {suite}")

        # Run benchmark
        result = await benchmark.run(request.config)

        # Store results
        metrics_dict = {
            "overall_score": result.overall_score,
            "pass_rate": result.pass_rate,
            "summary": result.summary,
            "test_results": [
                {
                    "test_name": tr.test_name,
                    "passed": tr.passed,
                    "score": tr.score,
                    "metrics": tr.metrics,
                }
                for tr in result.test_results
            ],
        }

        eval_store.store_benchmark_result(
            benchmark_suite=suite,
            benchmark_name=benchmark.name,
            overall_score=result.overall_score,
            metrics=metrics_dict,
            test_cases_passed=result.tests_passed,
            test_cases_total=result.tests_total,
            config=result.config,
        )

        logger.info(
            f"Benchmark {suite} completed: "
            f"Score={result.overall_score:.3f}, "
            f"Passed={result.tests_passed}/{result.tests_total}"
        )

        return {
            "suite": suite,
            "overall_score": result.overall_score,
            "tests_passed": result.tests_passed,
            "tests_total": result.tests_total,
            "pass_rate": result.pass_rate,
            "summary": result.summary,
            "test_results": [
                {
                    "test_name": tr.test_name,
                    "passed": tr.passed,
                    "score": tr.score,
                    "duration_ms": tr.duration_ms,
                }
                for tr in result.test_results
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Benchmark run failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/benchmark/results")
async def get_benchmark_results(
    suite: Optional[str] = None,
    limit: int = 10,
):
    """
    Get historical benchmark results

    Optionally filter by suite name
    """
    try:
        if suite:
            results = eval_store.get_benchmark_history(suite, limit)
        else:
            # Get latest from all suites
            all_results = []
            for suite_name in benchmarks.keys():
                suite_results = eval_store.get_benchmark_history(suite_name, limit)
                all_results.extend(suite_results)
            results = sorted(all_results, key=lambda x: x["timestamp"], reverse=True)[
                :limit
            ]

        return {"results": results, "count": len(results)}

    except Exception as e:
        logger.error(f"Failed to get benchmark results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ab-test/start")
async def start_ab_test(request: ABTestStartRequest):
    """
    Start a new A/B test

    Compare two variants on a specific metric
    """
    try:
        test_id = f"ab_{uuid.uuid4().hex[:8]}"

        eval_store.create_ab_test(
            test_id=test_id,
            test_name=request.test_name,
            variant_a=request.variant_a,
            variant_b=request.variant_b,
            initial_results={
                "metric_name": request.metric_name,
                "variant_a_samples": [],
                "variant_b_samples": [],
            },
        )

        logger.info(f"Started A/B test: {test_id} - {request.test_name}")

        return {
            "test_id": test_id,
            "test_name": request.test_name,
            "status": "running",
            "message": f"A/B test started. Use /ab-test/update/{test_id} to add samples.",
        }

    except Exception as e:
        logger.error(f"Failed to start A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ab-test/update/{test_id}")
async def update_ab_test(test_id: str, request: ABTestUpdateRequest):
    """
    Update A/B test with new sample

    Add measurements for variant A or B
    """
    try:
        # Get existing test
        test = eval_store.get_ab_test(test_id)
        if not test:
            raise HTTPException(status_code=404, detail=f"Test {test_id} not found")

        # Add sample
        results = test["results"]
        if request.variant == "a":
            results["variant_a_samples"].append(request.value)
        elif request.variant == "b":
            results["variant_b_samples"].append(request.value)
        else:
            raise HTTPException(status_code=400, detail="Variant must be 'a' or 'b'")

        # Update test
        eval_store.update_ab_test(test_id, results=results)

        logger.info(
            f"Updated A/B test {test_id}: variant_{request.variant} = {request.value}"
        )

        return {
            "test_id": test_id,
            "variant": request.variant,
            "value": request.value,
            "samples_a": len(results["variant_a_samples"]),
            "samples_b": len(results["variant_b_samples"]),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ab-test/results/{test_id}")
async def get_ab_test_results(test_id: str):
    """
    Get A/B test results

    Returns statistical comparison of variants
    """
    try:
        test = eval_store.get_ab_test(test_id)
        if not test:
            raise HTTPException(status_code=404, detail=f"Test {test_id} not found")

        results = test["results"]
        samples_a = results.get("variant_a_samples", [])
        samples_b = results.get("variant_b_samples", [])

        # Calculate statistics
        if samples_a and samples_b:
            import statistics

            mean_a = statistics.mean(samples_a)
            mean_b = statistics.mean(samples_b)
            std_a = statistics.stdev(samples_a) if len(samples_a) > 1 else 0
            std_b = statistics.stdev(samples_b) if len(samples_b) > 1 else 0

            # Simple winner determination (can be improved with t-test)
            if mean_a > mean_b * 1.05:  # 5% threshold
                winner = "a"
                confidence = 0.85
            elif mean_b > mean_a * 1.05:
                winner = "b"
                confidence = 0.85
            else:
                winner = "tie"
                confidence = 0.5

            # Update test with winner
            eval_store.update_ab_test(
                test_id, status="completed", winner=winner, confidence=confidence
            )

            return {
                "test_id": test_id,
                "test_name": test["test_name"],
                "status": "completed",
                "winner": winner,
                "confidence": confidence,
                "variant_a": {
                    "description": test["variant_a"],
                    "mean": mean_a,
                    "std": std_a,
                    "samples": len(samples_a),
                },
                "variant_b": {
                    "description": test["variant_b"],
                    "mean": mean_b,
                    "std": std_b,
                    "samples": len(samples_b),
                },
            }
        else:
            return {
                "test_id": test_id,
                "test_name": test["test_name"],
                "status": "running",
                "message": "Insufficient samples for analysis",
                "samples_a": len(samples_a),
                "samples_b": len(samples_b),
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get A/B test results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/regression/check")
async def check_regression(request: RegressionCheckRequest):
    """
    Check for performance regression

    Compare current value against historical baseline
    """
    try:
        result = eval_store.check_regression(
            metric_name=request.metric_name,
            current_value=request.current_value,
            threshold_pct=request.threshold_pct,
        )

        # Store the current metric
        eval_store.store_performance_metric(
            metric_type=request.metric_name,
            value=request.current_value,
        )

        logger.info(
            f"Regression check for {request.metric_name}: "
            f"{'REGRESSION' if result['is_regression'] else 'OK'} "
            f"(current={request.current_value:.2f}, "
            f"baseline={result['baseline_value']:.2f}, "
            f"change={result['pct_change']:.1f}%)"
        )

        return result

    except Exception as e:
        logger.error(f"Regression check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/dashboard")
async def get_metrics_dashboard():
    """
    Get aggregated metrics for dashboard

    Returns latest benchmarks, A/B tests, regressions, and accuracy
    """
    try:
        dashboard = eval_store.get_metrics_dashboard()
        return dashboard

    except Exception as e:
        logger.error(f"Failed to get dashboard metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Run server
if __name__ == "__main__":
    uvicorn.run(
        "evaluation_server:app",
        host="0.0.0.0",
        port=8005,
        reload=True,
        log_level="info",
    )
