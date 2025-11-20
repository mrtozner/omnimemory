"""
SQLite Data Store for Evaluation Results
Stores benchmark results, A/B test data, and performance metrics over time
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class EvaluationStore:
    """SQLite store for evaluation results with time-series tracking"""

    def __init__(self, db_path: str = "evaluation_results.db"):
        """Initialize the evaluation store"""
        self.db_path = db_path
        self._init_database()
        logger.info(f"Evaluation store initialized at {db_path}")

    def _init_database(self):
        """Create database tables if they don't exist"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Benchmark results table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    benchmark_suite TEXT NOT NULL,
                    benchmark_name TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    overall_score REAL NOT NULL,
                    metrics TEXT NOT NULL,  -- JSON
                    test_cases_passed INTEGER,
                    test_cases_total INTEGER,
                    config TEXT,  -- JSON
                    notes TEXT
                )
            """
            )

            # A/B test results table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS ab_test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL UNIQUE,
                    test_name TEXT NOT NULL,
                    variant_a TEXT NOT NULL,
                    variant_b TEXT NOT NULL,
                    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    end_time DATETIME,
                    status TEXT DEFAULT 'running',  -- running, completed, failed
                    winner TEXT,  -- a, b, or tie
                    confidence REAL,
                    results TEXT NOT NULL  -- JSON
                )
            """
            )

            # Performance metrics table (time-series)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_type TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    value REAL NOT NULL,
                    metadata TEXT,  -- JSON
                    service_name TEXT,
                    operation_name TEXT
                )
            """
            )

            # Memory accuracy metrics
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_accuracy (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    precision_score REAL NOT NULL,
                    recall_score REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    test_set TEXT NOT NULL,
                    strategy TEXT,
                    metadata TEXT  -- JSON
                )
            """
            )

            # Regression detection
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS regression_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric_name TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    baseline_value REAL NOT NULL,
                    threshold_pct REAL NOT NULL,
                    is_regression BOOLEAN NOT NULL,
                    severity TEXT,  -- low, medium, high, critical
                    details TEXT  -- JSON
                )
            """
            )

            conn.commit()
            logger.info("Database schema initialized")

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def store_benchmark_result(
        self,
        benchmark_suite: str,
        benchmark_name: str,
        overall_score: float,
        metrics: Dict[str, Any],
        test_cases_passed: int,
        test_cases_total: int,
        config: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> int:
        """Store a benchmark result"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO benchmark_results (
                    benchmark_suite, benchmark_name, overall_score, metrics,
                    test_cases_passed, test_cases_total, config, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    benchmark_suite,
                    benchmark_name,
                    overall_score,
                    json.dumps(metrics),
                    test_cases_passed,
                    test_cases_total,
                    json.dumps(config) if config else None,
                    notes,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def create_ab_test(
        self,
        test_id: str,
        test_name: str,
        variant_a: str,
        variant_b: str,
        initial_results: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a new A/B test"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO ab_test_results (
                    test_id, test_name, variant_a, variant_b, results
                ) VALUES (?, ?, ?, ?, ?)
            """,
                (
                    test_id,
                    test_name,
                    variant_a,
                    variant_b,
                    json.dumps(initial_results or {}),
                ),
            )
            conn.commit()

    def update_ab_test(
        self,
        test_id: str,
        status: Optional[str] = None,
        winner: Optional[str] = None,
        confidence: Optional[float] = None,
        results: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update an A/B test"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            updates = []
            params = []

            if status:
                updates.append("status = ?")
                params.append(status)
                if status == "completed":
                    updates.append("end_time = CURRENT_TIMESTAMP")

            if winner:
                updates.append("winner = ?")
                params.append(winner)

            if confidence is not None:
                updates.append("confidence = ?")
                params.append(confidence)

            if results:
                updates.append("results = ?")
                params.append(json.dumps(results))

            if updates:
                params.append(test_id)
                query = (
                    f"UPDATE ab_test_results SET {', '.join(updates)} WHERE test_id = ?"
                )
                cursor.execute(query, params)
                conn.commit()

    def get_ab_test(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get A/B test results"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM ab_test_results WHERE test_id = ?", (test_id,)
            )
            row = cursor.fetchone()
            if row:
                return {
                    "test_id": row["test_id"],
                    "test_name": row["test_name"],
                    "variant_a": row["variant_a"],
                    "variant_b": row["variant_b"],
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "status": row["status"],
                    "winner": row["winner"],
                    "confidence": row["confidence"],
                    "results": json.loads(row["results"]) if row["results"] else {},
                }
            return None

    def store_performance_metric(
        self,
        metric_type: str,
        value: float,
        service_name: Optional[str] = None,
        operation_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a performance metric"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO performance_metrics (
                    metric_type, value, service_name, operation_name, metadata
                ) VALUES (?, ?, ?, ?, ?)
            """,
                (
                    metric_type,
                    value,
                    service_name,
                    operation_name,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            conn.commit()

    def store_memory_accuracy(
        self,
        precision: float,
        recall: float,
        f1: float,
        test_set: str,
        strategy: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store memory accuracy metrics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO memory_accuracy (
                    precision_score, recall_score, f1_score, test_set, strategy, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    precision,
                    recall,
                    f1,
                    test_set,
                    strategy,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            conn.commit()

    def check_regression(
        self,
        metric_name: str,
        current_value: float,
        threshold_pct: float = 5.0,
        lookback_days: int = 7,
    ) -> Dict[str, Any]:
        """Check for performance regression"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get baseline (average of last N days, excluding today)
            cursor.execute(
                """
                SELECT AVG(value) as baseline
                FROM performance_metrics
                WHERE metric_type = ?
                AND timestamp >= datetime('now', '-' || ? || ' days')
                AND timestamp < datetime('now', 'start of day')
            """,
                (metric_name, lookback_days),
            )
            row = cursor.fetchone()
            baseline = row["baseline"] if row and row["baseline"] else current_value

            # Calculate regression
            if baseline > 0:
                pct_change = ((current_value - baseline) / baseline) * 100
            else:
                pct_change = 0

            is_regression = abs(pct_change) > threshold_pct and pct_change < 0

            # Determine severity
            if not is_regression:
                severity = None
            elif abs(pct_change) > 20:
                severity = "critical"
            elif abs(pct_change) > 10:
                severity = "high"
            elif abs(pct_change) > 5:
                severity = "medium"
            else:
                severity = "low"

            # Store regression check
            cursor.execute(
                """
                INSERT INTO regression_checks (
                    metric_name, current_value, baseline_value,
                    threshold_pct, is_regression, severity, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metric_name,
                    current_value,
                    baseline,
                    threshold_pct,
                    is_regression,
                    severity,
                    json.dumps({"pct_change": pct_change}),
                ),
            )
            conn.commit()

            return {
                "metric_name": metric_name,
                "current_value": current_value,
                "baseline_value": baseline,
                "pct_change": pct_change,
                "is_regression": is_regression,
                "severity": severity,
                "threshold_pct": threshold_pct,
            }

    def get_benchmark_history(
        self, benchmark_suite: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get historical benchmark results"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM benchmark_results
                WHERE benchmark_suite = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (benchmark_suite, limit),
            )
            results = []
            for row in cursor.fetchall():
                results.append(
                    {
                        "id": row["id"],
                        "benchmark_name": row["benchmark_name"],
                        "timestamp": row["timestamp"],
                        "overall_score": row["overall_score"],
                        "metrics": json.loads(row["metrics"]) if row["metrics"] else {},
                        "test_cases_passed": row["test_cases_passed"],
                        "test_cases_total": row["test_cases_total"],
                    }
                )
            return results

    def get_metrics_dashboard(self) -> Dict[str, Any]:
        """Get aggregated metrics for dashboard"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Latest benchmark results
            cursor.execute(
                """
                SELECT benchmark_suite, benchmark_name, overall_score, timestamp
                FROM benchmark_results
                WHERE id IN (
                    SELECT MAX(id) FROM benchmark_results
                    GROUP BY benchmark_suite
                )
                ORDER BY timestamp DESC
            """
            )
            latest_benchmarks = [dict(row) for row in cursor.fetchall()]

            # Active A/B tests
            cursor.execute(
                """
                SELECT test_id, test_name, status, winner, confidence
                FROM ab_test_results
                WHERE status = 'running'
            """
            )
            active_ab_tests = [dict(row) for row in cursor.fetchall()]

            # Recent regressions
            cursor.execute(
                """
                SELECT metric_name, severity, current_value, baseline_value, timestamp
                FROM regression_checks
                WHERE is_regression = 1
                ORDER BY timestamp DESC
                LIMIT 5
            """
            )
            recent_regressions = [dict(row) for row in cursor.fetchall()]

            # Latest memory accuracy
            cursor.execute(
                """
                SELECT precision_score, recall_score, f1_score, timestamp
                FROM memory_accuracy
                ORDER BY timestamp DESC
                LIMIT 1
            """
            )
            row = cursor.fetchone()
            latest_accuracy = dict(row) if row else None

            return {
                "latest_benchmarks": latest_benchmarks,
                "active_ab_tests": active_ab_tests,
                "recent_regressions": recent_regressions,
                "latest_accuracy": latest_accuracy,
            }

    def close(self):
        """Close the database connection"""
        logger.info("Evaluation store closed")
