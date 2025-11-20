"""
OmniMemory Compression Module
Provides VisionDrop compression for context reduction
"""

from .visiondrop import VisionDropCompressor, CompressedContext

__version__ = "1.0.0"
__all__ = ["VisionDropCompressor", "CompressedContext"]
