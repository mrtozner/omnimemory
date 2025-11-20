"""
Quality Metrics for Compression and Memory
Measures ROUGE-L, BERTScore, and other quality metrics
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class QualityMetrics:
    """Measure quality of compression and memory operations"""

    @staticmethod
    def calculate_rouge_l(
        hypothesis: str,
        reference: str,
    ) -> float:
        """
        Calculate ROUGE-L score (simplified version)

        Args:
            hypothesis: Generated/compressed text
            reference: Reference/original text

        Returns:
            ROUGE-L F1 score (0.0 to 1.0)
        """
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            scores = scorer.score(reference, hypothesis)
            return scores["rougeL"].fmeasure
        except ImportError:
            logger.warning("rouge-score not installed, using fallback")
            return QualityMetrics._fallback_rouge_l(hypothesis, reference)

    @staticmethod
    def _fallback_rouge_l(hypothesis: str, reference: str) -> float:
        """Fallback ROUGE-L implementation (basic)"""
        hyp_words = hypothesis.lower().split()
        ref_words = reference.lower().split()

        if not hyp_words or not ref_words:
            return 0.0

        # Find longest common subsequence
        lcs_length = QualityMetrics._lcs_length(hyp_words, ref_words)

        if lcs_length == 0:
            return 0.0

        precision = lcs_length / len(hyp_words)
        recall = lcs_length / len(ref_words)

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    @staticmethod
    def calculate_bert_score(
        hypothesis: str,
        reference: str,
        model_type: str = "microsoft/deberta-xlarge-mnli",
    ) -> Dict[str, float]:
        """
        Calculate BERTScore

        Args:
            hypothesis: Generated/compressed text
            reference: Reference/original text
            model_type: BERT model to use

        Returns:
            Dict with precision, recall, f1
        """
        try:
            from bert_score import score

            P, R, F1 = score(
                [hypothesis],
                [reference],
                model_type=model_type,
                verbose=False,
            )
            return {
                "precision": float(P[0]),
                "recall": float(R[0]),
                "f1": float(F1[0]),
            }
        except ImportError:
            logger.warning("bert-score not installed, using fallback")
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }
        except Exception as e:
            logger.error(f"BERTScore calculation failed: {e}")
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

    @staticmethod
    def calculate_bleu(
        hypothesis: str,
        reference: str,
    ) -> float:
        """
        Calculate BLEU score

        Args:
            hypothesis: Generated text
            reference: Reference text

        Returns:
            BLEU score (0.0 to 1.0)
        """
        try:
            from sacrebleu import corpus_bleu

            score = corpus_bleu([hypothesis], [[reference]])
            return score.score / 100.0  # Normalize to 0-1
        except ImportError:
            logger.warning("sacrebleu not installed, using fallback")
            return QualityMetrics._fallback_bleu(hypothesis, reference)

    @staticmethod
    def _fallback_bleu(hypothesis: str, reference: str) -> float:
        """Fallback BLEU implementation (unigram only)"""
        hyp_words = set(hypothesis.lower().split())
        ref_words = set(reference.lower().split())

        if not hyp_words or not ref_words:
            return 0.0

        matches = len(hyp_words & ref_words)
        precision = matches / len(hyp_words) if hyp_words else 0.0

        return precision

    @staticmethod
    def calculate_semantic_similarity(
        text1: str,
        text2: str,
        embeddings1: Optional[List[float]] = None,
        embeddings2: Optional[List[float]] = None,
    ) -> float:
        """
        Calculate semantic similarity between two texts

        Args:
            text1: First text
            text2: Second text
            embeddings1: Optional precomputed embeddings for text1
            embeddings2: Optional precomputed embeddings for text2

        Returns:
            Cosine similarity (0.0 to 1.0)
        """
        if embeddings1 and embeddings2:
            return QualityMetrics._cosine_similarity(embeddings1, embeddings2)

        # Fallback to simple word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same length")

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    @staticmethod
    def calculate_information_retention(
        original: str,
        compressed: str,
    ) -> Dict[str, Any]:
        """
        Calculate how much information is retained after compression

        Args:
            original: Original text
            compressed: Compressed text

        Returns:
            Dict with various retention metrics
        """
        # Word-level metrics
        original_words = set(original.lower().split())
        compressed_words = set(compressed.lower().split())

        word_retention = (
            len(compressed_words & original_words) / len(original_words)
            if original_words
            else 0.0
        )

        # Character-level metrics
        char_retention = len(compressed) / len(original) if original else 0.0

        # ROUGE-L for structural similarity
        rouge_l = QualityMetrics.calculate_rouge_l(compressed, original)

        return {
            "word_retention": word_retention,
            "character_retention": char_retention,
            "rouge_l": rouge_l,
            "original_words": len(original_words),
            "compressed_words": len(compressed_words),
            "original_chars": len(original),
            "compressed_chars": len(compressed),
        }

    @staticmethod
    def evaluate_compression_quality(
        original: str,
        compressed: str,
        original_tokens: int,
        compressed_tokens: int,
    ) -> Dict[str, Any]:
        """
        Comprehensive compression quality evaluation

        Args:
            original: Original text
            compressed: Compressed text
            original_tokens: Token count before compression
            compressed_tokens: Token count after compression

        Returns:
            Dict with all quality metrics
        """
        info_retention = QualityMetrics.calculate_information_retention(
            original, compressed
        )

        tokens_saved = original_tokens - compressed_tokens
        compression_ratio = (
            compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        )

        # Calculate quality score (weighted average)
        quality_score = (
            0.4 * info_retention["rouge_l"]
            + 0.3 * info_retention["word_retention"]
            + 0.3 * min(1.0, compression_ratio * 2)  # Reward higher compression
        )

        return {
            "quality_score": quality_score,
            "rouge_l": info_retention["rouge_l"],
            "word_retention": info_retention["word_retention"],
            "character_retention": info_retention["character_retention"],
            "tokens_saved": tokens_saved,
            "compression_ratio": compression_ratio,
            "savings_percentage": (
                (tokens_saved / original_tokens * 100) if original_tokens > 0 else 0
            ),
        }
