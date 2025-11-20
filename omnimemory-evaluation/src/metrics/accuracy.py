"""
Accuracy Metrics for Memory Evaluation
Measures precision, recall, F1, and other accuracy metrics
"""

from typing import List, Dict, Any, Set
import logging

logger = logging.getLogger(__name__)


class AccuracyMetrics:
    """Calculate accuracy metrics for memory retrieval"""

    @staticmethod
    def calculate_precision_recall_f1(
        retrieved: List[str],
        relevant: List[str],
    ) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score

        Args:
            retrieved: List of retrieved memory IDs
            relevant: List of relevant memory IDs (ground truth)

        Returns:
            Dict with precision, recall, and f1 scores
        """
        if not retrieved and not relevant:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

        if not retrieved:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        if not relevant:
            return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

        retrieved_set = set(retrieved)
        relevant_set = set(relevant)

        true_positives = len(retrieved_set & relevant_set)
        false_positives = len(retrieved_set - relevant_set)
        false_negatives = len(relevant_set - retrieved_set)

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )

        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )

        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }

    @staticmethod
    def calculate_mean_average_precision(
        rankings: List[List[str]],
        relevant_sets: List[List[str]],
    ) -> float:
        """
        Calculate Mean Average Precision (MAP)

        Args:
            rankings: List of ranked retrieval results
            relevant_sets: List of relevant items for each query

        Returns:
            MAP score (0.0 to 1.0)
        """
        if not rankings or not relevant_sets:
            return 0.0

        average_precisions = []

        for retrieved, relevant in zip(rankings, relevant_sets):
            if not relevant:
                continue

            relevant_set = set(relevant)
            num_relevant = 0
            precision_sum = 0.0

            for i, item in enumerate(retrieved, 1):
                if item in relevant_set:
                    num_relevant += 1
                    precision_at_i = num_relevant / i
                    precision_sum += precision_at_i

            if len(relevant_set) > 0:
                average_precision = precision_sum / len(relevant_set)
                average_precisions.append(average_precision)

        return (
            sum(average_precisions) / len(average_precisions)
            if average_precisions
            else 0.0
        )

    @staticmethod
    def calculate_ndcg(
        retrieved: List[str],
        relevant: List[str],
        k: int = 10,
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@k)

        Args:
            retrieved: Ranked list of retrieved items
            relevant: List of relevant items (ground truth)
            k: Cutoff rank

        Returns:
            NDCG@k score (0.0 to 1.0)
        """
        relevant_set = set(relevant)

        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(retrieved[:k], 1):
            relevance = 1.0 if item in relevant_set else 0.0
            dcg += relevance / (i if i == 1 else (i - 1).bit_length() + 1)

        # Calculate IDCG (ideal DCG)
        ideal_ranking = [1.0] * min(len(relevant), k)
        idcg = 0.0
        for i, relevance in enumerate(ideal_ranking, 1):
            idcg += relevance / (i if i == 1 else (i - 1).bit_length() + 1)

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def calculate_mrr(
        rankings: List[List[str]],
        relevant_sets: List[List[str]],
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)

        Args:
            rankings: List of ranked retrieval results
            relevant_sets: List of relevant items for each query

        Returns:
            MRR score (0.0 to 1.0)
        """
        if not rankings or not relevant_sets:
            return 0.0

        reciprocal_ranks = []

        for retrieved, relevant in zip(rankings, relevant_sets):
            relevant_set = set(relevant)

            for i, item in enumerate(retrieved, 1):
                if item in relevant_set:
                    reciprocal_ranks.append(1.0 / i)
                    break
            else:
                reciprocal_ranks.append(0.0)

        return (
            sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        )

    @staticmethod
    def calculate_hit_rate(
        retrieved: List[str],
        relevant: List[str],
        k: int = 10,
    ) -> float:
        """
        Calculate hit rate (at least one relevant item in top-k)

        Args:
            retrieved: Ranked list of retrieved items
            relevant: List of relevant items
            k: Cutoff rank

        Returns:
            1.0 if hit, 0.0 otherwise
        """
        relevant_set = set(relevant)
        top_k = retrieved[:k]
        return 1.0 if any(item in relevant_set for item in top_k) else 0.0
