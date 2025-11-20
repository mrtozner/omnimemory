"""
Memory Layer Configuration Module
Provides layer-specific compression strategies and user-selectable compression modes
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any


class MemoryLayer(Enum):
    """Memory layer types with different retention strategies"""

    SESSION = "session"  # Recent conversation context
    TASK = "task"  # Active work context
    LONG_TERM = "long_term"  # Archived historical context
    GLOBAL = "global"  # System prompts, tool definitions


@dataclass
class LayerConfig:
    """Configuration for each memory layer"""

    quality_threshold: float
    target_compression: float
    preserve_structure: bool
    importance_boost: Dict[str, float]
    max_age_hours: Optional[int] = None  # For automatic archival
    priority: int = 1  # Higher = more important


# Layer-specific configurations optimized for use case
MEMORY_LAYER_CONFIGS = {
    MemoryLayer.SESSION: LayerConfig(
        quality_threshold=0.80,  # More aggressive for recent context
        target_compression=0.85,  # Less compression for active conversation
        preserve_structure=True,
        importance_boost={
            "headers": 1.3,
            "code_blocks": 1.5,
            "error_messages": 2.5,  # Critical in session
            "function_names": 1.5,
            "user_messages": 2.0,  # User input is key
        },
        max_age_hours=24,  # Auto-archive after 24h
        priority=4,  # Highest priority
    ),
    MemoryLayer.TASK: LayerConfig(
        quality_threshold=0.85,  # Balanced for active work
        target_compression=0.90,  # Standard compression
        preserve_structure=True,
        importance_boost={
            "headers": 1.5,
            "code_blocks": 1.5,
            "error_messages": 2.0,
            "function_names": 1.8,
            "todo_items": 1.8,  # Task tracking important
        },
        max_age_hours=7 * 24,  # Archive after a week
        priority=3,
    ),
    MemoryLayer.LONG_TERM: LayerConfig(
        quality_threshold=0.90,  # Conservative for archived content
        target_compression=0.95,  # More aggressive compression OK
        preserve_structure=True,
        importance_boost={
            "headers": 1.5,
            "code_blocks": 1.3,
            "error_messages": 1.5,
            "function_names": 1.5,
            "summaries": 2.0,  # Summaries are key in archives
        },
        max_age_hours=None,  # Never auto-delete
        priority=2,
    ),
    MemoryLayer.GLOBAL: LayerConfig(
        quality_threshold=0.95,  # Minimal compression for system context
        target_compression=0.98,  # Almost no compression
        preserve_structure=True,
        importance_boost={
            "headers": 1.0,  # No boost needed
            "code_blocks": 1.0,
            "error_messages": 1.0,
            "function_names": 1.0,
            "system_prompts": 3.0,  # System prompts critical
        },
        max_age_hours=None,
        priority=5,  # Never compress away
    ),
}


class CompressionMode(Enum):
    """User-selectable compression modes"""

    SPEED = "speed"  # Fast, aggressive (0.75 threshold)
    BALANCED = "balanced"  # Default (0.85 threshold)
    QUALITY = "quality"  # Conservative (0.90 threshold)
    MAXIMUM = "maximum"  # Minimal compression (0.95 threshold)


COMPRESSION_MODE_CONFIGS = {
    CompressionMode.SPEED: {
        "quality_threshold": 0.75,
        "target_compression": 0.95,
        "preserve_structure": False,  # Skip structure checks for speed
    },
    CompressionMode.BALANCED: {
        "quality_threshold": 0.85,
        "target_compression": 0.944,
        "preserve_structure": True,
    },
    CompressionMode.QUALITY: {
        "quality_threshold": 0.90,
        "target_compression": 0.90,
        "preserve_structure": True,
    },
    CompressionMode.MAXIMUM: {
        "quality_threshold": 0.95,
        "target_compression": 0.85,
        "preserve_structure": True,
    },
}


def get_layer_config(
    memory_layer: Optional[MemoryLayer] = None,
    compression_mode: Optional[CompressionMode] = None,
) -> LayerConfig:
    """
    Get configuration based on memory layer and compression mode

    Args:
        memory_layer: Memory layer type (SESSION, TASK, LONG_TERM, GLOBAL)
        compression_mode: Compression mode (SPEED, BALANCED, QUALITY, MAXIMUM)

    Returns:
        LayerConfig with merged settings
    """
    # Start with layer config or default
    if memory_layer:
        config = MEMORY_LAYER_CONFIGS[memory_layer]
    else:
        config = MEMORY_LAYER_CONFIGS[MemoryLayer.TASK]  # Default

    # Create a copy to avoid modifying the original
    config = LayerConfig(
        quality_threshold=config.quality_threshold,
        target_compression=config.target_compression,
        preserve_structure=config.preserve_structure,
        importance_boost=config.importance_boost.copy(),
        max_age_hours=config.max_age_hours,
        priority=config.priority,
    )

    # Override with compression mode if specified
    if compression_mode:
        mode_config = COMPRESSION_MODE_CONFIGS[compression_mode]
        config.quality_threshold = mode_config["quality_threshold"]
        config.target_compression = mode_config["target_compression"]
        config.preserve_structure = mode_config["preserve_structure"]

    return config
