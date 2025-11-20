"""
OmniMemory File Context Package
Cross-tool file caching with Tri-Index support
"""

from .cross_tool_cache import CrossToolFileCache
from .tier_manager import TierManager
from .structure_extractor import FileStructureExtractor
from .witness_selector import WitnessSelector

__all__ = [
    "CrossToolFileCache",
    "TierManager",
    "FileStructureExtractor",
    "WitnessSelector",
]
__version__ = "0.1.0"
