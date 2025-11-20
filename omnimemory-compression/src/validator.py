"""
Compression Validation System

Validates compression quality using:
- ROUGE-L: String overlap and recall
- BERTScore: Semantic similarity (optional, requires model download)

Ensures compressed text maintains sufficient quality relative to original.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .config import ValidationConfig

logger = logging.getLogger(__name__)


class ValidationMetric(str, Enum):
    """Available validation metrics"""

    ROUGE_L = "rouge-l"
    BERTSCORE = "bertscore"
    BOTH = "both"


@dataclass
class ValidationResult:
    """Result of compression validation"""

    passed: bool
    rouge_l_score: Optional[float] = None
    bertscore_f1: Optional[float] = None
    bertscore_precision: Optional[float] = None
    bertscore_recall: Optional[float] = None
    details: Dict[str, any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class CompressionValidator:
    """
    Validates compression quality using multiple metrics

    Example:
        ```python
        validator = CompressionValidator()

        result = validator.validate(
            original="The quick brown fox jumps over the lazy dog",
            compressed="Quick brown fox jumps lazy dog",
            metrics=["rouge-l"]
        )

        if result.passed:
            print(f"Compression valid! ROUGE-L: {result.rouge_l_score:.2%}")
        else:
            print("Compression quality too low")
        ```
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize validator

        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()

        # Lazy-loaded models
        self._rouge_scorer = None
        self._bertscore = None

        logger.info(
            f"CompressionValidator initialized "
            f"(rouge={'on' if self.config.rouge_enabled else 'off'}, "
            f"bertscore={'on' if self.config.bertscore_enabled else 'off'})"
        )

    def validate(
        self,
        original: str,
        compressed: str,
        metrics: Optional[List[str]] = None,
        custom_thresholds: Optional[Dict[str, float]] = None,
    ) -> ValidationResult:
        """
        Validate compression quality

        Args:
            original: Original text
            compressed: Compressed text
            metrics: List of metrics to use (default: all enabled)
            custom_thresholds: Custom thresholds to override config

        Returns:
            ValidationResult with scores and pass/fail status
        """
        if not original or not compressed:
            return ValidationResult(
                passed=False,
                details={"error": "Empty text provided"},
            )

        # Determine which metrics to use
        use_rouge = False
        use_bertscore = False

        if metrics is None:
            # Use all enabled metrics
            use_rouge = self.config.rouge_enabled
            use_bertscore = self.config.bertscore_enabled
        else:
            # Use specified metrics
            metrics_lower = [m.lower() for m in metrics]
            use_rouge = "rouge-l" in metrics_lower or "both" in metrics_lower
            use_bertscore = "bertscore" in metrics_lower or "both" in metrics_lower

        # Get thresholds
        rouge_threshold = (
            custom_thresholds.get("rouge_l", self.config.rouge_min_score)
            if custom_thresholds
            else self.config.rouge_min_score
        )
        bertscore_threshold = (
            custom_thresholds.get("bertscore", self.config.bertscore_min_score)
            if custom_thresholds
            else self.config.bertscore_min_score
        )

        result = ValidationResult(passed=True)

        # Run ROUGE-L
        if use_rouge:
            try:
                rouge_l = self._compute_rouge_l(original, compressed)
                result.rouge_l_score = rouge_l

                if rouge_l < rouge_threshold:
                    result.passed = False
                    result.details["rouge_l_failed"] = True
                    result.details["rouge_l_threshold"] = rouge_threshold

                logger.debug(f"ROUGE-L: {rouge_l:.3f} (threshold: {rouge_threshold})")

            except Exception as e:
                logger.error(f"ROUGE-L computation failed: {e}")
                result.details["rouge_l_error"] = str(e)

        # Run BERTScore
        if use_bertscore:
            try:
                precision, recall, f1 = self._compute_bertscore(original, compressed)
                result.bertscore_precision = precision
                result.bertscore_recall = recall
                result.bertscore_f1 = f1

                if f1 < bertscore_threshold:
                    result.passed = False
                    result.details["bertscore_failed"] = True
                    result.details["bertscore_threshold"] = bertscore_threshold

                logger.debug(
                    f"BERTScore F1: {f1:.3f} (threshold: {bertscore_threshold})"
                )

            except Exception as e:
                logger.error(f"BERTScore computation failed: {e}")
                result.details["bertscore_error"] = str(e)

        return result

    def validate_batch(
        self,
        pairs: List[Tuple[str, str]],
        metrics: Optional[List[str]] = None,
        custom_thresholds: Optional[Dict[str, float]] = None,
    ) -> List[ValidationResult]:
        """
        Validate multiple compression pairs

        Args:
            pairs: List of (original, compressed) tuples
            metrics: List of metrics to use
            custom_thresholds: Custom thresholds

        Returns:
            List of ValidationResult objects
        """
        results = []

        for original, compressed in pairs:
            result = self.validate(original, compressed, metrics, custom_thresholds)
            results.append(result)

        return results

    def _compute_rouge_l(self, original: str, compressed: str) -> float:
        """
        Compute ROUGE-L score

        ROUGE-L measures longest common subsequence between texts

        Args:
            original: Original text
            compressed: Compressed text

        Returns:
            ROUGE-L F1 score (0-1)
        """
        # Lazy load rouge scorer
        if self._rouge_scorer is None:
            try:
                from rouge_score import rouge_scorer

                self._rouge_scorer = rouge_scorer.RougeScorer(
                    ["rougeL"], use_stemmer=True
                )
            except ImportError:
                raise ImportError(
                    "rouge-score not installed. "
                    "Install with: pip install rouge-score"
                )

        # Compute ROUGE-L
        scores = self._rouge_scorer.score(original, compressed)
        rouge_l_f1 = scores["rougeL"].fmeasure

        return rouge_l_f1

    def _compute_bertscore(
        self, original: str, compressed: str
    ) -> Tuple[float, float, float]:
        """
        Compute BERTScore

        BERTScore measures semantic similarity using contextual embeddings

        Args:
            original: Original text
            compressed: Compressed text

        Returns:
            Tuple of (precision, recall, f1)
        """
        # Lazy load BERTScore
        if self._bertscore is None:
            try:
                import bert_score

                self._bertscore = bert_score
            except ImportError:
                raise ImportError(
                    "bert-score not installed. " "Install with: pip install bert-score"
                )

        # Compute BERTScore
        # Note: This downloads a model on first use (~1GB)
        precision, recall, f1 = self._bertscore.score(
            [compressed],
            [original],
            model_type=self.config.bertscore_model,
            verbose=False,
        )

        return (
            precision.item(),
            recall.item(),
            f1.item(),
        )

    def get_summary(self, results: List[ValidationResult]) -> Dict[str, any]:
        """
        Get summary statistics for multiple validation results

        Args:
            results: List of validation results

        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
            }

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        # Average scores
        rouge_scores = [r.rouge_l_score for r in results if r.rouge_l_score is not None]
        bertscore_f1s = [r.bertscore_f1 for r in results if r.bertscore_f1 is not None]

        avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else None
        avg_bertscore = (
            sum(bertscore_f1s) / len(bertscore_f1s) if bertscore_f1s else None
        )

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / total * 100, 2),
            "avg_rouge_l": round(avg_rouge, 3) if avg_rouge else None,
            "avg_bertscore_f1": round(avg_bertscore, 3) if avg_bertscore else None,
        }


# Convenience functions
def quick_validate(
    original: str,
    compressed: str,
    min_rouge_l: float = 0.5,
) -> bool:
    """
    Quick validation using ROUGE-L only

    Args:
        original: Original text
        compressed: Compressed text
        min_rouge_l: Minimum ROUGE-L score

    Returns:
        True if compression passes validation
    """
    config = ValidationConfig(
        rouge_enabled=True,
        rouge_min_score=min_rouge_l,
        bertscore_enabled=False,
    )

    validator = CompressionValidator(config)
    result = validator.validate(original, compressed)

    return result.passed


def validate_with_bertscore(
    original: str,
    compressed: str,
    min_bertscore: float = 0.85,
) -> Tuple[bool, float]:
    """
    Validate using BERTScore (semantic similarity)

    Args:
        original: Original text
        compressed: Compressed text
        min_bertscore: Minimum BERTScore F1

    Returns:
        Tuple of (passed, bertscore_f1)
    """
    config = ValidationConfig(
        rouge_enabled=False,
        bertscore_enabled=True,
        bertscore_min_score=min_bertscore,
    )

    validator = CompressionValidator(config)
    result = validator.validate(original, compressed)

    return result.passed, result.bertscore_f1
