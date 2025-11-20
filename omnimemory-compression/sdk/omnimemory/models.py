"""
Data models for OmniMemory SDK
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class CompressionResult:
    """Result from compression operation"""

    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    retained_indices: List[int]
    quality_score: float
    compressed_text: str
    model_id: str
    tokenizer_strategy: Optional[str] = None
    is_exact_tokenization: Optional[bool] = None


@dataclass
class TokenCount:
    """Result from token counting operation"""

    token_count: int
    model_id: str
    strategy_used: str
    is_exact: bool
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Result from validation operation"""

    passed: bool
    rouge_l_score: Optional[float] = None
    bertscore_f1: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
