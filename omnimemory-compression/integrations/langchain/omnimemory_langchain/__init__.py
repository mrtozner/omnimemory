"""
OmniMemory integration for LangChain
"""

from .compressor import OmniMemoryDocumentCompressor
from .prompt_compressor import OmniMemoryPromptCompressor

__version__ = "0.1.0"
__all__ = ["OmniMemoryDocumentCompressor", "OmniMemoryPromptCompressor"]
