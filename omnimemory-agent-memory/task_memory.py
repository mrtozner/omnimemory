"""
Task Completion Memory System - Learns from task patterns to improve future performance

Features:
- Pattern mining from successful/failed tasks
- Approach optimization suggestions
- Time and token consumption tracking
- Task similarity detection
- Predictive task approach recommendations
"""

import asyncio
import json
import logging
import sqlite3
import hashlib
import httpx
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from enum import Enum

logger = logging.getLogger(__name__)


class TaskOutcome(Enum):
    """Task completion outcomes"""

    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"


class PatternType(Enum):
    """Types of patterns extracted from tasks"""

    WORKFLOW = "workflow"
    TOOL_SEQUENCE = "tool_sequence"
    FILE_ACCESS = "file_access"
    DECISION_SEQUENCE = "decision_sequence"
    ERROR_RECOVERY = "error_recovery"


@dataclass
class TaskContext:
    """Context information for a completed task"""

    task_id: str
    session_id: str
    task_description: str
    approach_taken: str
    files_modified: List[str]
    tools_used: List[str]
    decisions_made: List[str]
    time_taken: int  # seconds
    tokens_consumed: int
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "task_description": self.task_description,
            "approach_taken": self.approach_taken,
            "files_modified": json.dumps(self.files_modified),
            "tools_used": json.dumps(self.tools_used),
            "decisions_made": json.dumps(self.decisions_made),
            "time_taken": self.time_taken,
            "tokens_consumed": self.tokens_consumed,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TaskOutcomeData:
    """Outcome information for a completed task"""

    success: bool
    error_message: Optional[str] = None
    user_satisfaction: float = 0.0  # 0-1
    rework_needed: bool = False
    outcome_type: TaskOutcome = TaskOutcome.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "success": self.success,
            "error_message": self.error_message,
            "user_satisfaction": self.user_satisfaction,
            "rework_needed": self.rework_needed,
            "outcome_type": self.outcome_type.value,
        }


@dataclass
class TaskPattern:
    """Represents a discovered pattern from task completions"""

    pattern_id: str
    pattern_type: PatternType
    pattern_description: str
    pattern_data: Dict[str, Any]
    frequency: int
    success_rate: float
    avg_time: int
    avg_tokens: int
    confidence: float
    last_seen: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.value,
            "pattern_description": self.pattern_description,
            "pattern_data": json.dumps(self.pattern_data),
            "frequency": self.frequency,
            "success_rate": self.success_rate,
            "avg_time": self.avg_time,
            "avg_tokens": self.avg_tokens,
            "confidence": self.confidence,
            "last_seen": self.last_seen.isoformat(),
        }


@dataclass
class OptimizationSuggestion:
    """Optimization suggestion for task approach"""

    suggestion_id: str
    task_pattern_id: str
    suggestion_type: str
    suggestion_text: str
    expected_improvement: float
    times_applied: int
    success_rate: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "suggestion_id": self.suggestion_id,
            "task_pattern_id": self.task_pattern_id,
            "suggestion_type": self.suggestion_type,
            "suggestion_text": self.suggestion_text,
            "expected_improvement": self.expected_improvement,
            "times_applied": self.times_applied,
            "success_rate": self.success_rate,
        }


@dataclass
class TaskPrediction:
    """Prediction for a new task"""

    recommended_approach: str
    similar_tasks: List[Dict[str, Any]]
    predicted_time: int
    predicted_tokens: int
    confidence: float
    optimization_suggestions: List[str]
    potential_issues: List[str]


class TaskPatternMiner:
    """
    Mines patterns from task completions.

    Extracts:
    - Tool usage sequences
    - File access patterns
    - Decision sequences
    - Error recovery patterns
    - Successful workflow patterns
    """

    def __init__(self):
        logger.info("TaskPatternMiner initialized")

    def extract_success_patterns(
        self, tasks: List[Dict[str, Any]]
    ) -> List[TaskPattern]:
        """
        Mine patterns from successful tasks.

        Args:
            tasks: List of successful task completions

        Returns:
            List of discovered patterns
        """
        patterns = []

        # Extract tool sequence patterns
        tool_patterns = self._extract_tool_sequences(tasks)
        patterns.extend(tool_patterns)

        # Extract file access patterns
        file_patterns = self._extract_file_patterns(tasks)
        patterns.extend(file_patterns)

        # Extract workflow patterns
        workflow_patterns = self._extract_workflow_patterns(tasks)
        patterns.extend(workflow_patterns)

        logger.info(
            f"Extracted {len(patterns)} success patterns from {len(tasks)} tasks"
        )
        return patterns

    def extract_failure_patterns(
        self, tasks: List[Dict[str, Any]]
    ) -> List[TaskPattern]:
        """
        Mine patterns from failed tasks.

        Args:
            tasks: List of failed task completions

        Returns:
            List of discovered failure patterns
        """
        patterns = []

        # Extract error sequences
        error_patterns = self._extract_error_patterns(tasks)
        patterns.extend(error_patterns)

        # Extract problematic tool combinations
        problem_patterns = self._extract_problematic_sequences(tasks)
        patterns.extend(problem_patterns)

        logger.info(
            f"Extracted {len(patterns)} failure patterns from {len(tasks)} tasks"
        )
        return patterns

    def find_workflow_patterns(self, tasks: List[Dict[str, Any]]) -> List[TaskPattern]:
        """
        Discover common workflows across tasks.

        Args:
            tasks: List of task completions

        Returns:
            List of workflow patterns
        """
        workflows = defaultdict(list)

        for task in tasks:
            # Create workflow signature from tools and decisions
            tools = json.loads(task.get("tools_used", "[]"))
            workflow_sig = " -> ".join(tools[:5])  # First 5 tools
            workflows[workflow_sig].append(task)

        patterns = []
        for workflow_sig, workflow_tasks in workflows.items():
            if len(workflow_tasks) >= 2:  # Pattern needs at least 2 occurrences
                success_count = sum(
                    1 for t in workflow_tasks if t.get("success", False)
                )
                pattern = TaskPattern(
                    pattern_id=self._generate_pattern_id(workflow_sig),
                    pattern_type=PatternType.WORKFLOW,
                    pattern_description=f"Common workflow: {workflow_sig}",
                    pattern_data={
                        "workflow": workflow_sig,
                        "tasks": len(workflow_tasks),
                    },
                    frequency=len(workflow_tasks),
                    success_rate=success_count / len(workflow_tasks),
                    avg_time=int(
                        np.mean([t.get("time_taken", 0) for t in workflow_tasks])
                    ),
                    avg_tokens=int(
                        np.mean([t.get("tokens_consumed", 0) for t in workflow_tasks])
                    ),
                    confidence=min(len(workflow_tasks) / 10.0, 1.0),
                    last_seen=datetime.now(),
                )
                patterns.append(pattern)

        logger.info(f"Found {len(patterns)} workflow patterns")
        return patterns

    def _extract_tool_sequences(self, tasks: List[Dict[str, Any]]) -> List[TaskPattern]:
        """Extract common tool usage sequences"""
        sequences = defaultdict(list)

        for task in tasks:
            tools = json.loads(task.get("tools_used", "[]"))
            # Create n-grams of tool sequences (length 3)
            for i in range(len(tools) - 2):
                seq = " -> ".join(tools[i : i + 3])
                sequences[seq].append(task)

        patterns = []
        for seq, seq_tasks in sequences.items():
            if len(seq_tasks) >= 2:
                success_count = sum(1 for t in seq_tasks if t.get("success", False))
                pattern = TaskPattern(
                    pattern_id=self._generate_pattern_id(seq),
                    pattern_type=PatternType.TOOL_SEQUENCE,
                    pattern_description=f"Tool sequence: {seq}",
                    pattern_data={"sequence": seq, "tasks": len(seq_tasks)},
                    frequency=len(seq_tasks),
                    success_rate=success_count / len(seq_tasks),
                    avg_time=int(np.mean([t.get("time_taken", 0) for t in seq_tasks])),
                    avg_tokens=int(
                        np.mean([t.get("tokens_consumed", 0) for t in seq_tasks])
                    ),
                    confidence=min(len(seq_tasks) / 5.0, 1.0),
                    last_seen=datetime.now(),
                )
                patterns.append(pattern)

        return patterns

    def _extract_file_patterns(self, tasks: List[Dict[str, Any]]) -> List[TaskPattern]:
        """Extract file access patterns"""
        file_patterns = defaultdict(list)

        for task in tasks:
            files = json.loads(task.get("files_modified", "[]"))
            # Group by file type/directory
            for file in files:
                file_type = Path(file).suffix
                if file_type:
                    file_patterns[file_type].append(task)

        patterns = []
        for file_type, type_tasks in file_patterns.items():
            if len(type_tasks) >= 2:
                success_count = sum(1 for t in type_tasks if t.get("success", False))
                pattern = TaskPattern(
                    pattern_id=self._generate_pattern_id(f"files_{file_type}"),
                    pattern_type=PatternType.FILE_ACCESS,
                    pattern_description=f"File type pattern: {file_type}",
                    pattern_data={"file_type": file_type, "tasks": len(type_tasks)},
                    frequency=len(type_tasks),
                    success_rate=success_count / len(type_tasks),
                    avg_time=int(np.mean([t.get("time_taken", 0) for t in type_tasks])),
                    avg_tokens=int(
                        np.mean([t.get("tokens_consumed", 0) for t in type_tasks])
                    ),
                    confidence=min(len(type_tasks) / 5.0, 1.0),
                    last_seen=datetime.now(),
                )
                patterns.append(pattern)

        return patterns

    def _extract_workflow_patterns(
        self, tasks: List[Dict[str, Any]]
    ) -> List[TaskPattern]:
        """Extract high-level workflow patterns"""
        # Categorize tasks by approach
        approaches = defaultdict(list)

        for task in tasks:
            approach = task.get("approach_taken", "unknown")
            # Normalize approach to key phrases
            approach_key = self._normalize_approach(approach)
            approaches[approach_key].append(task)

        patterns = []
        for approach_key, approach_tasks in approaches.items():
            if len(approach_tasks) >= 2:
                success_count = sum(
                    1 for t in approach_tasks if t.get("success", False)
                )
                pattern = TaskPattern(
                    pattern_id=self._generate_pattern_id(f"workflow_{approach_key}"),
                    pattern_type=PatternType.WORKFLOW,
                    pattern_description=f"Approach: {approach_key}",
                    pattern_data={
                        "approach": approach_key,
                        "tasks": len(approach_tasks),
                    },
                    frequency=len(approach_tasks),
                    success_rate=success_count / len(approach_tasks),
                    avg_time=int(
                        np.mean([t.get("time_taken", 0) for t in approach_tasks])
                    ),
                    avg_tokens=int(
                        np.mean([t.get("tokens_consumed", 0) for t in approach_tasks])
                    ),
                    confidence=min(len(approach_tasks) / 5.0, 1.0),
                    last_seen=datetime.now(),
                )
                patterns.append(pattern)

        return patterns

    def _extract_error_patterns(self, tasks: List[Dict[str, Any]]) -> List[TaskPattern]:
        """Extract common error patterns"""
        error_patterns = defaultdict(list)

        for task in tasks:
            error_msg = task.get("error_message", "")
            if error_msg:
                # Extract error type (first line or first 50 chars)
                error_type = error_msg.split("\n")[0][:50]
                error_patterns[error_type].append(task)

        patterns = []
        for error_type, error_tasks in error_patterns.items():
            if len(error_tasks) >= 2:
                pattern = TaskPattern(
                    pattern_id=self._generate_pattern_id(f"error_{error_type}"),
                    pattern_type=PatternType.ERROR_RECOVERY,
                    pattern_description=f"Error pattern: {error_type}",
                    pattern_data={"error": error_type, "occurrences": len(error_tasks)},
                    frequency=len(error_tasks),
                    success_rate=0.0,  # These are failures
                    avg_time=int(
                        np.mean([t.get("time_taken", 0) for t in error_tasks])
                    ),
                    avg_tokens=int(
                        np.mean([t.get("tokens_consumed", 0) for t in error_tasks])
                    ),
                    confidence=min(len(error_tasks) / 3.0, 1.0),
                    last_seen=datetime.now(),
                )
                patterns.append(pattern)

        return patterns

    def _extract_problematic_sequences(
        self, tasks: List[Dict[str, Any]]
    ) -> List[TaskPattern]:
        """Extract tool sequences that commonly fail"""
        sequences = defaultdict(list)

        for task in tasks:
            tools = json.loads(task.get("tools_used", "[]"))
            # Create sequences leading to failure
            if len(tools) >= 2:
                seq = " -> ".join(tools[-3:])  # Last 3 tools before failure
                sequences[seq].append(task)

        patterns = []
        for seq, seq_tasks in sequences.items():
            if len(seq_tasks) >= 2:
                pattern = TaskPattern(
                    pattern_id=self._generate_pattern_id(f"problem_{seq}"),
                    pattern_type=PatternType.TOOL_SEQUENCE,
                    pattern_description=f"Problematic sequence: {seq}",
                    pattern_data={"sequence": seq, "failures": len(seq_tasks)},
                    frequency=len(seq_tasks),
                    success_rate=0.0,
                    avg_time=int(np.mean([t.get("time_taken", 0) for t in seq_tasks])),
                    avg_tokens=int(
                        np.mean([t.get("tokens_consumed", 0) for t in seq_tasks])
                    ),
                    confidence=min(len(seq_tasks) / 3.0, 1.0),
                    last_seen=datetime.now(),
                )
                patterns.append(pattern)

        return patterns

    def _normalize_approach(self, approach: str) -> str:
        """Normalize approach text to key phrases"""
        approach_lower = approach.lower()

        # Key approach patterns
        if "incremental" in approach_lower or "step by step" in approach_lower:
            return "incremental"
        elif "refactor" in approach_lower:
            return "refactor"
        elif "debug" in approach_lower or "fix" in approach_lower:
            return "debug"
        elif "test" in approach_lower:
            return "test_driven"
        elif "research" in approach_lower:
            return "research_first"
        else:
            return "direct_implementation"

    def _generate_pattern_id(self, text: str) -> str:
        """Generate unique pattern ID"""
        return hashlib.md5(text.encode()).hexdigest()[:16]


class SuccessFailureAnalyzer:
    """
    Analyzes what makes tasks succeed or fail.
    """

    def __init__(self):
        logger.info("SuccessFailureAnalyzer initialized")

    def analyze_success_factors(
        self, successful_tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Identify success factors.

        Args:
            successful_tasks: List of successful task completions

        Returns:
            Analysis of success factors
        """
        if not successful_tasks:
            return {"factors": [], "insights": "No successful tasks to analyze"}

        factors = {
            "avg_time": int(
                np.mean([t.get("time_taken", 0) for t in successful_tasks])
            ),
            "avg_tokens": int(
                np.mean([t.get("tokens_consumed", 0) for t in successful_tasks])
            ),
            "avg_satisfaction": np.mean(
                [t.get("user_satisfaction", 0.0) for t in successful_tasks]
            ),
            "common_tools": self._find_common_tools(successful_tasks),
            "common_approaches": self._find_common_approaches(successful_tasks),
            "time_efficiency": self._calculate_efficiency(successful_tasks, "time"),
            "token_efficiency": self._calculate_efficiency(successful_tasks, "tokens"),
        }

        logger.info(f"Analyzed success factors for {len(successful_tasks)} tasks")
        return factors

    def analyze_failure_causes(
        self, failed_tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Identify failure causes.

        Args:
            failed_tasks: List of failed task completions

        Returns:
            Analysis of failure causes
        """
        if not failed_tasks:
            return {"causes": [], "insights": "No failed tasks to analyze"}

        causes = {
            "avg_time_before_failure": int(
                np.mean([t.get("time_taken", 0) for t in failed_tasks])
            ),
            "avg_tokens_before_failure": int(
                np.mean([t.get("tokens_consumed", 0) for t in failed_tasks])
            ),
            "common_errors": self._find_common_errors(failed_tasks),
            "problematic_tools": self._find_problematic_tools(failed_tasks),
            "missing_patterns": self._identify_missing_steps(failed_tasks),
        }

        logger.info(f"Analyzed failure causes for {len(failed_tasks)} tasks")
        return causes

    def calculate_approach_effectiveness(
        self, approach: str, tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Score effectiveness of a specific approach.

        Args:
            approach: Approach name
            tasks: List of tasks using this approach

        Returns:
            Effectiveness metrics
        """
        if not tasks:
            return {"effectiveness": 0.0, "metrics": {}}

        success_count = sum(1 for t in tasks if t.get("success", False))
        success_rate = success_count / len(tasks)

        metrics = {
            "success_rate": success_rate,
            "avg_time": int(np.mean([t.get("time_taken", 0) for t in tasks])),
            "avg_tokens": int(np.mean([t.get("tokens_consumed", 0) for t in tasks])),
            "avg_satisfaction": np.mean(
                [t.get("user_satisfaction", 0.0) for t in tasks]
            ),
            "rework_rate": sum(1 for t in tasks if t.get("rework_needed", False))
            / len(tasks),
            "sample_size": len(tasks),
        }

        # Calculate overall effectiveness score (0-1)
        effectiveness = (
            metrics["success_rate"] * 0.4
            + metrics["avg_satisfaction"] * 0.3
            + (1 - metrics["rework_rate"]) * 0.2
            + min(metrics["sample_size"] / 10.0, 1.0) * 0.1
        )

        return {
            "approach": approach,
            "effectiveness": effectiveness,
            "metrics": metrics,
        }

    def _find_common_tools(self, tasks: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """Find most common tools in successful tasks"""
        tool_counter = Counter()
        for task in tasks:
            tools = json.loads(task.get("tools_used", "[]"))
            tool_counter.update(tools)
        return tool_counter.most_common(5)

    def _find_common_approaches(
        self, tasks: List[Dict[str, Any]]
    ) -> List[Tuple[str, int]]:
        """Find most common approaches in successful tasks"""
        approach_counter = Counter()
        for task in tasks:
            approach = task.get("approach_taken", "unknown")
            approach_counter[approach] += 1
        return approach_counter.most_common(5)

    def _find_common_errors(self, tasks: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """Find most common error types"""
        error_counter = Counter()
        for task in tasks:
            error_msg = task.get("error_message", "")
            if error_msg:
                error_type = error_msg.split("\n")[0][:50]
                error_counter[error_type] += 1
        return error_counter.most_common(5)

    def _find_problematic_tools(
        self, tasks: List[Dict[str, Any]]
    ) -> List[Tuple[str, int]]:
        """Find tools commonly present in failed tasks"""
        tool_counter = Counter()
        for task in tasks:
            tools = json.loads(task.get("tools_used", "[]"))
            tool_counter.update(tools)
        return tool_counter.most_common(5)

    def _identify_missing_steps(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """Identify commonly missing steps in failed tasks"""
        # Simple heuristic: check for common tool sequences that might be missing
        missing = []

        for task in tasks:
            tools = json.loads(task.get("tools_used", "[]"))

            # Check for common missing patterns
            if "Write" in tools and "Read" not in tools:
                missing.append("Missing Read before Write")
            if "Edit" in tools and "Read" not in tools:
                missing.append("Missing Read before Edit")
            if "Bash" in tools and len(tools) < 2:
                missing.append("Single Bash command without verification")

        return list(set(missing))[:5]

    def _calculate_efficiency(self, tasks: List[Dict[str, Any]], metric: str) -> float:
        """Calculate efficiency score for time or tokens"""
        if metric == "time":
            values = [t.get("time_taken", 0) for t in tasks]
        else:
            values = [t.get("tokens_consumed", 0) for t in tasks]

        if not values or max(values) == 0:
            return 0.0

        # Efficiency = inverse of normalized values (lower is better)
        normalized = [v / max(values) for v in values]
        return 1.0 - np.mean(normalized)


class TaskOptimizationEngine:
    """
    Suggests optimizations based on learned patterns.
    """

    def __init__(self):
        logger.info("TaskOptimizationEngine initialized")

    def suggest_improvements(
        self, task_approach: str, historical_data: List[Dict[str, Any]]
    ) -> List[OptimizationSuggestion]:
        """
        Generate optimization suggestions.

        Args:
            task_approach: Current approach description
            historical_data: Historical task data for analysis

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # Analyze token usage
        token_suggestions = self._suggest_token_optimizations(historical_data)
        suggestions.extend(token_suggestions)

        # Analyze time usage
        time_suggestions = self._suggest_time_optimizations(historical_data)
        suggestions.extend(time_suggestions)

        # Analyze tool sequences
        tool_suggestions = self._suggest_tool_optimizations(historical_data)
        suggestions.extend(tool_suggestions)

        logger.info(f"Generated {len(suggestions)} optimization suggestions")
        return suggestions

    def identify_bottlenecks(
        self, task_execution: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Find performance bottlenecks.

        Args:
            task_execution: Task execution data

        Returns:
            List of identified bottlenecks
        """
        bottlenecks = []

        time_taken = task_execution.get("time_taken", 0)
        tokens_used = task_execution.get("tokens_consumed", 0)
        tools = json.loads(task_execution.get("tools_used", "[]"))

        # Time bottlenecks
        if time_taken > 300:  # > 5 minutes
            bottlenecks.append(
                {
                    "type": "time",
                    "severity": "high",
                    "description": f"Task took {time_taken}s (>{300}s threshold)",
                    "suggestion": "Consider breaking into smaller tasks or parallelizing operations",
                }
            )

        # Token bottlenecks
        if tokens_used > 10000:
            bottlenecks.append(
                {
                    "type": "tokens",
                    "severity": "high",
                    "description": f"Task consumed {tokens_used} tokens (>10000 threshold)",
                    "suggestion": "Use OmniMemory compression or reduce context size",
                }
            )

        # Tool usage bottlenecks
        if len(tools) > 20:
            bottlenecks.append(
                {
                    "type": "tool_usage",
                    "severity": "medium",
                    "description": f"Task used {len(tools)} tools (>20 threshold)",
                    "suggestion": "Batch operations or use more efficient tool sequences",
                }
            )

        # Repetitive operations
        tool_counts = Counter(tools)
        for tool, count in tool_counts.items():
            if count > 5:
                bottlenecks.append(
                    {
                        "type": "repetition",
                        "severity": "medium",
                        "description": f"Tool {tool} used {count} times",
                        "suggestion": f"Consider batching {tool} operations",
                    }
                )

        logger.info(f"Identified {len(bottlenecks)} bottlenecks")
        return bottlenecks

    def _suggest_token_optimizations(
        self, historical_data: List[Dict[str, Any]]
    ) -> List[OptimizationSuggestion]:
        """Suggest ways to reduce token consumption"""
        suggestions = []

        # Find high token tasks
        high_token_tasks = [
            t for t in historical_data if t.get("tokens_consumed", 0) > 5000
        ]

        if high_token_tasks:
            avg_tokens = int(
                np.mean([t.get("tokens_consumed", 0) for t in high_token_tasks])
            )
            suggestion = OptimizationSuggestion(
                suggestion_id=f"token_opt_{len(suggestions)}",
                task_pattern_id="general",
                suggestion_type="token_reduction",
                suggestion_text=(
                    f"High token usage detected (avg {avg_tokens}). "
                    "Consider: 1) Use OmniMemory compression, "
                    "2) Read specific symbols instead of full files, "
                    "3) Use semantic search to reduce file reads"
                ),
                expected_improvement=0.3,
                times_applied=0,
                success_rate=0.0,
            )
            suggestions.append(suggestion)

        return suggestions

    def _suggest_time_optimizations(
        self, historical_data: List[Dict[str, Any]]
    ) -> List[OptimizationSuggestion]:
        """Suggest ways to reduce time consumption"""
        suggestions = []

        # Find slow tasks
        slow_tasks = [t for t in historical_data if t.get("time_taken", 0) > 180]

        if slow_tasks:
            avg_time = int(np.mean([t.get("time_taken", 0) for t in slow_tasks]))
            suggestion = OptimizationSuggestion(
                suggestion_id=f"time_opt_{len(suggestions)}",
                task_pattern_id="general",
                suggestion_type="time_reduction",
                suggestion_text=(
                    f"Slow execution detected (avg {avg_time}s). "
                    "Consider: 1) Parallel tool calls, "
                    "2) Reduce sequential dependencies, "
                    "3) Use caching for repeated operations"
                ),
                expected_improvement=0.4,
                times_applied=0,
                success_rate=0.0,
            )
            suggestions.append(suggestion)

        return suggestions

    def _suggest_tool_optimizations(
        self, historical_data: List[Dict[str, Any]]
    ) -> List[OptimizationSuggestion]:
        """Suggest tool usage optimizations"""
        suggestions = []

        # Analyze tool sequences
        all_tools = []
        for task in historical_data:
            tools = json.loads(task.get("tools_used", "[]"))
            all_tools.extend(tools)

        tool_counts = Counter(all_tools)

        # Check for Read/Grep overuse
        if tool_counts.get("Read", 0) > len(historical_data) * 3:
            suggestion = OptimizationSuggestion(
                suggestion_id=f"tool_opt_{len(suggestions)}",
                task_pattern_id="general",
                suggestion_type="tool_optimization",
                suggestion_text=(
                    "Frequent Read operations detected. "
                    "Consider: 1) Use omnimemory_smart_read for compression, "
                    "2) Use symbol_overview before full reads, "
                    "3) Batch file reads"
                ),
                expected_improvement=0.5,
                times_applied=0,
                success_rate=0.0,
            )
            suggestions.append(suggestion)

        if tool_counts.get("Grep", 0) > len(historical_data) * 2:
            suggestion = OptimizationSuggestion(
                suggestion_id=f"tool_opt_{len(suggestions)}",
                task_pattern_id="general",
                suggestion_type="tool_optimization",
                suggestion_text=(
                    "Frequent Grep operations detected. "
                    "Consider: Use omnimemory_semantic_search instead of Grep "
                    "to find relevant files faster with higher accuracy"
                ),
                expected_improvement=0.6,
                times_applied=0,
                success_rate=0.0,
            )
            suggestions.append(suggestion)

        return suggestions


class TaskCompletionPredictor:
    """
    Predicts task completion likelihood and optimal approach.
    """

    def __init__(self, embedding_service_url: str = "http://localhost:8000"):
        self.embedding_url = embedding_service_url
        logger.info("TaskCompletionPredictor initialized")

    async def predict(
        self, new_task: str, historical_tasks: List[Dict[str, Any]]
    ) -> TaskPrediction:
        """
        Predict task completion.

        Args:
            new_task: New task description
            historical_tasks: Historical task data

        Returns:
            Task prediction with recommendations
        """
        # Find similar tasks
        similar_tasks = await self.find_similar_tasks(
            new_task, historical_tasks, limit=5
        )

        if not similar_tasks:
            # No historical data, return default prediction
            return TaskPrediction(
                recommended_approach="direct_implementation",
                similar_tasks=[],
                predicted_time=300,
                predicted_tokens=5000,
                confidence=0.1,
                optimization_suggestions=["No historical data available"],
                potential_issues=["First time completing this type of task"],
            )

        # Calculate predictions based on similar tasks
        avg_time = int(np.mean([t.get("time_taken", 0) for t in similar_tasks]))
        avg_tokens = int(np.mean([t.get("tokens_consumed", 0) for t in similar_tasks]))
        success_rate = sum(1 for t in similar_tasks if t.get("success", False)) / len(
            similar_tasks
        )

        # Find best approach
        approach_effectiveness = {}
        for task in similar_tasks:
            approach = task.get("approach_taken", "unknown")
            if approach not in approach_effectiveness:
                approach_effectiveness[approach] = []
            approach_effectiveness[approach].append(task.get("success", False))

        best_approach = max(
            approach_effectiveness.items(),
            key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 0,
        )[0]

        # Generate optimization suggestions
        optimizer = TaskOptimizationEngine()
        suggestions = optimizer.suggest_improvements(best_approach, similar_tasks)

        # Identify potential issues
        potential_issues = self._identify_potential_issues(similar_tasks)

        return TaskPrediction(
            recommended_approach=best_approach,
            similar_tasks=[
                {
                    "description": t.get("task_description", ""),
                    "success": t.get("success", False),
                    "time": t.get("time_taken", 0),
                    "tokens": t.get("tokens_consumed", 0),
                }
                for t in similar_tasks
            ],
            predicted_time=avg_time,
            predicted_tokens=avg_tokens,
            confidence=min(len(similar_tasks) / 10.0, 1.0) * success_rate,
            optimization_suggestions=[s.suggestion_text for s in suggestions],
            potential_issues=potential_issues,
        )

    async def find_similar_tasks(
        self,
        task_description: str,
        historical_tasks: List[Dict[str, Any]],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find similar completed tasks using embeddings.

        Args:
            task_description: New task description
            historical_tasks: Historical task data
            limit: Maximum number of similar tasks to return

        Returns:
            List of similar tasks
        """
        if not historical_tasks:
            return []

        try:
            # Get embedding for new task
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.embedding_url}/embed", json={"text": task_description}
                )
                new_embedding = np.array(response.json()["embedding"])

            # Calculate similarity with historical tasks
            similarities = []
            for task in historical_tasks:
                # Get or create embedding for historical task
                task_embedding_str = task.get("task_embedding")
                if task_embedding_str:
                    task_embedding = np.array(json.loads(task_embedding_str))
                    similarity = self._cosine_similarity(new_embedding, task_embedding)
                    similarities.append((similarity, task))

            # Sort by similarity and return top N
            similarities.sort(reverse=True, key=lambda x: x[0])
            similar_tasks = [task for _, task in similarities[:limit]]

            logger.info(f"Found {len(similar_tasks)} similar tasks")
            return similar_tasks

        except Exception as e:
            logger.error(f"Error finding similar tasks: {e}")
            # Fallback: simple text matching
            return self._simple_text_matching(task_description, historical_tasks, limit)

    def recommend_approach(self, similar_tasks: List[Dict[str, Any]]) -> str:
        """
        Recommend approach based on similar successful tasks.

        Args:
            similar_tasks: List of similar tasks

        Returns:
            Recommended approach description
        """
        if not similar_tasks:
            return "direct_implementation"

        # Count approaches in successful tasks
        successful_tasks = [t for t in similar_tasks if t.get("success", False)]

        if not successful_tasks:
            return "incremental (no successful patterns found, use caution)"

        approach_counter = Counter()
        for task in successful_tasks:
            approach = task.get("approach_taken", "unknown")
            approach_counter[approach] += 1

        most_common_approach = approach_counter.most_common(1)[0][0]
        return most_common_approach

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        if norm_product == 0:
            return 0.0

        return float(dot_product / norm_product)

    def _simple_text_matching(
        self, task_description: str, historical_tasks: List[Dict[str, Any]], limit: int
    ) -> List[Dict[str, Any]]:
        """Fallback simple text matching"""
        task_words = set(task_description.lower().split())

        scores = []
        for task in historical_tasks:
            desc = task.get("task_description", "")
            desc_words = set(desc.lower().split())
            overlap = len(task_words & desc_words)
            scores.append((overlap, task))

        scores.sort(reverse=True, key=lambda x: x[0])
        return [task for _, task in scores[:limit]]

    def _identify_potential_issues(
        self, similar_tasks: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify potential issues based on similar tasks"""
        issues = []

        # Check failure rate
        failed_tasks = [t for t in similar_tasks if not t.get("success", True)]
        if len(failed_tasks) > len(similar_tasks) * 0.3:
            issues.append(
                f"High failure rate in similar tasks ({len(failed_tasks)}/{len(similar_tasks)})"
            )

        # Check for common errors
        common_errors = Counter()
        for task in failed_tasks:
            error_msg = task.get("error_message", "")
            if error_msg:
                error_type = error_msg.split("\n")[0][:50]
                common_errors[error_type] += 1

        if common_errors:
            most_common_error = common_errors.most_common(1)[0]
            issues.append(
                f"Common error: {most_common_error[0]} (occurred {most_common_error[1]} times)"
            )

        # Check for rework
        rework_count = sum(1 for t in similar_tasks if t.get("rework_needed", False))
        if rework_count > len(similar_tasks) * 0.3:
            issues.append(
                f"High rework rate ({rework_count}/{len(similar_tasks)} tasks)"
            )

        return issues


class TaskCompletionMemory:
    """
    Learns from task completion patterns to improve future performance.

    Main entry point for task learning system.
    """

    def __init__(
        self,
        db_path: str = "~/.omnimemory/task_memory.db",
        embedding_service_url: str = "http://localhost:8000",
    ):
        """
        Initialize task completion memory system.

        Args:
            db_path: Path to SQLite database
            embedding_service_url: URL for embedding service
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.embedding_url = embedding_service_url

        # Initialize database connection
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Create schema
        self._create_schema()

        # Initialize components
        self.pattern_miner = TaskPatternMiner()
        self.success_analyzer = SuccessFailureAnalyzer()
        self.optimization_engine = TaskOptimizationEngine()
        self.predictor = TaskCompletionPredictor(embedding_service_url)

        logger.info(f"Initialized TaskCompletionMemory at {self.db_path}")

    def _create_schema(self):
        """Create database schema for task memory"""
        cursor = self.conn.cursor()

        # Task completions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS task_completions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE NOT NULL,
                session_id TEXT NOT NULL,
                task_description TEXT NOT NULL,
                task_embedding TEXT,
                approach_taken TEXT NOT NULL,
                files_modified TEXT,
                tools_used TEXT,
                decisions_made TEXT,
                time_taken INTEGER NOT NULL,
                tokens_consumed INTEGER NOT NULL,
                success INTEGER NOT NULL,
                error_message TEXT,
                user_satisfaction REAL DEFAULT 0.0,
                rework_needed INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Task patterns table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS task_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT UNIQUE NOT NULL,
                pattern_type TEXT NOT NULL,
                pattern_description TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                frequency INTEGER NOT NULL,
                success_rate REAL NOT NULL,
                avg_time INTEGER NOT NULL,
                avg_tokens INTEGER NOT NULL,
                last_seen TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Optimization suggestions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS optimization_suggestions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                suggestion_id TEXT UNIQUE NOT NULL,
                task_pattern_id TEXT NOT NULL,
                suggestion_type TEXT NOT NULL,
                suggestion TEXT NOT NULL,
                expected_improvement REAL NOT NULL,
                times_applied INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (task_pattern_id) REFERENCES task_patterns(pattern_id)
            )
        """
        )

        # Create indexes for performance
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_session
            ON task_completions(session_id, timestamp DESC)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_success
            ON task_completions(success, timestamp DESC)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_patterns_type
            ON task_patterns(pattern_type, frequency DESC)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_patterns_success
            ON task_patterns(success_rate DESC)
        """
        )

        self.conn.commit()
        logger.info("Task memory schema created/verified")

    async def learn_from_task_completion(
        self, task_context: TaskContext, outcome: TaskOutcomeData
    ) -> Dict[str, Any]:
        """
        Learn from a completed task.

        Args:
            task_context: Context information about the task
            outcome: Outcome information

        Returns:
            Learning summary
        """
        try:
            # Generate embedding for task description
            embedding = await self._get_embedding(task_context.task_description)

            # Store task completion
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO task_completions (
                    task_id, session_id, task_description, task_embedding,
                    approach_taken, files_modified, tools_used, decisions_made,
                    time_taken, tokens_consumed, success, error_message,
                    user_satisfaction, rework_needed, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    task_context.task_id,
                    task_context.session_id,
                    task_context.task_description,
                    json.dumps(embedding.tolist()) if embedding is not None else None,
                    task_context.approach_taken,
                    json.dumps(task_context.files_modified),
                    json.dumps(task_context.tools_used),
                    json.dumps(task_context.decisions_made),
                    task_context.time_taken,
                    task_context.tokens_consumed,
                    1 if outcome.success else 0,
                    outcome.error_message,
                    outcome.user_satisfaction,
                    1 if outcome.rework_needed else 0,
                    task_context.timestamp.isoformat(),
                ),
            )
            self.conn.commit()

            # Extract patterns from recent tasks
            recent_tasks = self._get_recent_tasks(limit=20)

            if outcome.success:
                patterns = self.pattern_miner.extract_success_patterns(recent_tasks)
            else:
                patterns = self.pattern_miner.extract_failure_patterns(recent_tasks)

            # Store patterns
            for pattern in patterns:
                self._store_pattern(pattern)

            # Generate optimization suggestions
            suggestions = self.optimization_engine.suggest_improvements(
                task_context.approach_taken, recent_tasks
            )

            # Store suggestions
            for suggestion in suggestions:
                self._store_suggestion(suggestion)

            logger.info(
                f"Learned from task {task_context.task_id}: {len(patterns)} patterns, {len(suggestions)} suggestions"
            )

            return {
                "task_id": task_context.task_id,
                "patterns_discovered": len(patterns),
                "suggestions_generated": len(suggestions),
                "success": outcome.success,
            }

        except Exception as e:
            logger.error(f"Error learning from task completion: {e}")
            return {"error": str(e)}

    async def predict_task_approach(self, new_task: str) -> TaskPrediction:
        """
        Predict best approach for a new task.

        Args:
            new_task: New task description

        Returns:
            Task prediction with recommendations
        """
        try:
            # Get historical tasks
            historical_tasks = self._get_all_tasks()

            # Generate prediction
            prediction = await self.predictor.predict(new_task, historical_tasks)

            logger.info(
                f"Generated prediction for task: confidence={prediction.confidence}"
            )
            return prediction

        except Exception as e:
            logger.error(f"Error predicting task approach: {e}")
            # Return default prediction on error
            return TaskPrediction(
                recommended_approach="direct_implementation",
                similar_tasks=[],
                predicted_time=300,
                predicted_tokens=5000,
                confidence=0.0,
                optimization_suggestions=[],
                potential_issues=[f"Error generating prediction: {str(e)}"],
            )

    def get_task_statistics(self) -> Dict[str, Any]:
        """Get overall task completion statistics"""
        cursor = self.conn.cursor()

        # Overall stats
        cursor.execute(
            """
            SELECT
                COUNT(*) as total_tasks,
                SUM(success) as successful_tasks,
                AVG(time_taken) as avg_time,
                AVG(tokens_consumed) as avg_tokens,
                AVG(user_satisfaction) as avg_satisfaction
            FROM task_completions
        """
        )
        overall = dict(cursor.fetchone())

        # Pattern stats
        cursor.execute(
            """
            SELECT
                pattern_type,
                COUNT(*) as count,
                AVG(success_rate) as avg_success_rate
            FROM task_patterns
            GROUP BY pattern_type
        """
        )
        patterns = [dict(row) for row in cursor.fetchall()]

        return {
            "overall": overall,
            "patterns": patterns,
            "success_rate": overall["successful_tasks"] / overall["total_tasks"]
            if overall["total_tasks"] > 0
            else 0.0,
        }

    async def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.embedding_url}/embed", json={"text": text}, timeout=10.0
                )
                return np.array(response.json()["embedding"])
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None

    def _get_recent_tasks(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent tasks from database"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM task_completions
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def _get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all tasks from database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM task_completions ORDER BY timestamp DESC")
        return [dict(row) for row in cursor.fetchall()]

    def _store_pattern(self, pattern: TaskPattern):
        """Store or update pattern in database"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO task_patterns (
                pattern_id, pattern_type, pattern_description, pattern_data,
                frequency, success_rate, avg_time, avg_tokens,
                last_seen, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                pattern.pattern_id,
                pattern.pattern_type.value,
                pattern.pattern_description,
                json.dumps(pattern.pattern_data),
                pattern.frequency,
                pattern.success_rate,
                pattern.avg_time,
                pattern.avg_tokens,
                pattern.last_seen.isoformat(),
                pattern.confidence,
            ),
        )
        self.conn.commit()

    def _store_suggestion(self, suggestion: OptimizationSuggestion):
        """Store or update suggestion in database"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO optimization_suggestions (
                suggestion_id, task_pattern_id, suggestion_type,
                suggestion, expected_improvement, times_applied, success_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                suggestion.suggestion_id,
                suggestion.task_pattern_id,
                suggestion.suggestion_type,
                suggestion.suggestion_text,
                suggestion.expected_improvement,
                suggestion.times_applied,
                suggestion.success_rate,
            ),
        )
        self.conn.commit()

    def close(self):
        """Close database connection"""
        self.conn.close()
        logger.info("TaskCompletionMemory closed")


# Convenience functions for easy usage
async def learn_from_task(
    task_description: str,
    approach: str,
    files_modified: List[str],
    tools_used: List[str],
    decisions_made: List[str],
    time_taken: int,
    tokens_consumed: int,
    success: bool,
    error_message: Optional[str] = None,
    user_satisfaction: float = 0.0,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to learn from a task completion.

    Args:
        task_description: Description of the task
        approach: Approach taken
        files_modified: List of files modified
        tools_used: List of tools used
        decisions_made: List of decisions made
        time_taken: Time taken in seconds
        tokens_consumed: Tokens consumed
        success: Whether task succeeded
        error_message: Error message if failed
        user_satisfaction: User satisfaction score (0-1)
        session_id: Session ID (auto-generated if not provided)

    Returns:
        Learning summary
    """
    memory = TaskCompletionMemory()

    task_id = hashlib.md5(f"{task_description}{datetime.now()}".encode()).hexdigest()[
        :16
    ]

    if session_id is None:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    context = TaskContext(
        task_id=task_id,
        session_id=session_id,
        task_description=task_description,
        approach_taken=approach,
        files_modified=files_modified,
        tools_used=tools_used,
        decisions_made=decisions_made,
        time_taken=time_taken,
        tokens_consumed=tokens_consumed,
        timestamp=datetime.now(),
    )

    outcome = TaskOutcomeData(
        success=success,
        error_message=error_message,
        user_satisfaction=user_satisfaction,
        rework_needed=False,
        outcome_type=TaskOutcome.SUCCESS if success else TaskOutcome.FAILED,
    )

    result = await memory.learn_from_task_completion(context, outcome)
    memory.close()

    return result


async def predict_task(task_description: str) -> TaskPrediction:
    """
    Convenience function to get prediction for a new task.

    Args:
        task_description: Description of the task

    Returns:
        Task prediction with recommendations
    """
    memory = TaskCompletionMemory()
    prediction = await memory.predict_task_approach(task_description)
    memory.close()

    return prediction


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    async def demo():
        print("Task Completion Memory System Demo\n")

        # Learn from a task
        print("1. Learning from a completed task...")
        result = await learn_from_task(
            task_description="Implement user authentication endpoint",
            approach="Test-driven incremental implementation",
            files_modified=["src/auth.py", "tests/test_auth.py"],
            tools_used=["Read", "Write", "Bash", "pytest"],
            decisions_made=["Use JWT tokens", "Add rate limiting"],
            time_taken=450,
            tokens_consumed=8000,
            success=True,
            user_satisfaction=0.9,
        )
        print(f"Learning result: {result}\n")

        # Predict for a new task
        print("2. Predicting approach for a new task...")
        prediction = await predict_task("Add OAuth2 support to authentication")
        print(f"Recommended approach: {prediction.recommended_approach}")
        print(f"Predicted time: {prediction.predicted_time}s")
        print(f"Predicted tokens: {prediction.predicted_tokens}")
        print(f"Confidence: {prediction.confidence:.2f}")
        print(f"Similar tasks: {len(prediction.similar_tasks)}")
        print(f"Suggestions: {len(prediction.optimization_suggestions)}\n")

        # Get statistics
        print("3. Getting task statistics...")
        memory = TaskCompletionMemory()
        stats = memory.get_task_statistics()
        print(f"Statistics: {json.dumps(stats, indent=2)}")
        memory.close()

    asyncio.run(demo())
