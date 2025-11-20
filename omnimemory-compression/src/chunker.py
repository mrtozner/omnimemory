"""
FastCDC (Fast Content-Defined Chunking) for Token Counting

Provides stable, content-defined chunking for:
- Deduplication across similar texts
- Caching of chunk token counts
- 10-50x speedup for repeated long contexts

Features:
- FastCDC algorithm for stable boundaries
- BLAKE3 hashing for chunk identification
- Boundary overlap extraction for correction
- Graceful fallback to simple rolling hash
"""

import logging
import hashlib
from typing import List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Content-defined chunk with metadata"""

    data: bytes
    offset: int
    length: int
    hash: str  # BLAKE3 hash for chunk identification


class FastCDCChunker:
    """
    Content-defined chunking for deduplication and caching.
    Uses FastCDC algorithm for stable chunk boundaries.

    Algorithm:
    - Uses rolling hash to find content-defined boundaries
    - Ensures chunks are between min_size and max_size
    - Same content → same chunks → cache hits!

    Example:
        ```python
        chunker = FastCDCChunker(
            min_size=2048,
            avg_size=4096,
            max_size=8192
        )

        if chunker.should_chunk(text):
            chunks = chunker.chunk_text(text)
            # Use chunks for caching...
        ```
    """

    def __init__(
        self,
        min_size: int = 2048,  # 2KB minimum
        avg_size: int = 4096,  # 4KB average
        max_size: int = 8192,  # 8KB maximum
        threshold: int = 16000,  # Use CDC for texts > 16K chars
    ):
        """
        Initialize FastCDC chunker

        Args:
            min_size: Minimum chunk size in bytes
            avg_size: Average target chunk size
            max_size: Maximum chunk size in bytes
            threshold: Minimum text length to trigger chunking
        """
        self.min_size = min_size
        self.avg_size = avg_size
        self.max_size = max_size
        self.threshold = threshold

        # Calculate mask for rolling hash (determines average chunk size)
        # More 1 bits = smaller chunks, fewer 1 bits = larger chunks
        # For avg_size=4096, we want mask that triggers ~every 4096 bytes
        self.mask = self._calculate_mask(avg_size)

        # Check if fastcdc library is available
        self._use_fastcdc = False
        try:
            import fastcdc

            self._fastcdc = fastcdc
            self._use_fastcdc = True
            logger.info(
                f"FastCDC library available, using optimized chunking "
                f"(min={min_size}, avg={avg_size}, max={max_size})"
            )
        except ImportError:
            logger.info(
                f"fastcdc library not available, using fallback rolling hash "
                f"(min={min_size}, avg={avg_size}, max={max_size})"
            )

        # Check if blake3 is available
        self._use_blake3 = False
        try:
            import blake3

            self._blake3 = blake3
            self._use_blake3 = True
            logger.debug("BLAKE3 hashing available")
        except ImportError:
            logger.debug("BLAKE3 not available, using SHA-256")

    def _calculate_mask(self, avg_size: int) -> int:
        """
        Calculate rolling hash mask for target average chunk size

        The mask determines when to create a boundary:
        - Fewer bits = larger chunks
        - More bits = smaller chunks
        """
        # Use logarithmic relationship: avg_size ≈ 2^bits
        import math

        bits = max(1, int(math.log2(avg_size)) - 1)
        mask = (1 << bits) - 1
        return mask

    def should_chunk(self, text: str) -> bool:
        """
        Determine if text should be chunked

        Args:
            text: Input text

        Returns:
            True if text is long enough to benefit from chunking
        """
        return len(text) >= self.threshold

    def chunk_text(self, text: str) -> List[Chunk]:
        """
        Split text into content-defined chunks using FastCDC

        Args:
            text: Input text to chunk

        Returns:
            List of Chunk objects with metadata

        Raises:
            ValueError: If text is empty
        """
        if not text:
            raise ValueError("Cannot chunk empty text")

        # Convert to bytes for chunking
        data = text.encode("utf-8")

        if self._use_fastcdc:
            return self._chunk_with_fastcdc(data)
        else:
            return self._chunk_with_rolling_hash(data)

    def _chunk_with_fastcdc(self, data: bytes) -> List[Chunk]:
        """
        Chunk using fastcdc library (optimized C implementation)

        Args:
            data: Bytes to chunk

        Returns:
            List of Chunk objects
        """
        try:
            # Use fastcdc library
            cdc = self._fastcdc.fastcdc(
                data,
                min_size=self.min_size,
                avg_size=self.avg_size,
                max_size=self.max_size,
            )

            chunks = []
            for chunk_data in cdc:
                # chunk_data is a tuple: (offset, length)
                offset, length = chunk_data
                chunk_bytes = data[offset : offset + length]

                # Generate hash for chunk
                chunk_hash = self._hash_chunk(chunk_bytes)

                chunks.append(
                    Chunk(
                        data=chunk_bytes,
                        offset=offset,
                        length=length,
                        hash=chunk_hash,
                    )
                )

            logger.debug(
                f"FastCDC: {len(data)} bytes → {len(chunks)} chunks "
                f"(avg={len(data) // len(chunks) if chunks else 0} bytes/chunk)"
            )

            return chunks

        except Exception as e:
            logger.warning(f"FastCDC failed: {e}, falling back to rolling hash")
            return self._chunk_with_rolling_hash(data)

    def _chunk_with_rolling_hash(self, data: bytes) -> List[Chunk]:
        """
        Chunk using simple rolling hash (fallback implementation)

        Algorithm:
        1. Compute rolling hash for each byte
        2. When hash & mask == 0, create boundary
        3. Enforce min_size and max_size constraints

        Args:
            data: Bytes to chunk

        Returns:
            List of Chunk objects
        """
        chunks = []
        data_len = len(data)

        if data_len == 0:
            return chunks

        # Rolling hash parameters
        PRIME = 31
        hash_val = 0
        window_size = 32  # Size of rolling window

        chunk_start = 0
        i = 0

        while i < data_len:
            # Update rolling hash
            hash_val = (hash_val * PRIME + data[i]) & 0xFFFFFFFF

            # Check for boundary conditions
            chunk_size = i - chunk_start + 1

            # Force boundary at max_size
            if chunk_size >= self.max_size:
                self._create_chunk(data, chunks, chunk_start, i + 1)
                chunk_start = i + 1
                hash_val = 0

            # Check for content-defined boundary (after min_size)
            elif chunk_size >= self.min_size and (hash_val & self.mask) == 0:
                self._create_chunk(data, chunks, chunk_start, i + 1)
                chunk_start = i + 1
                hash_val = 0

            i += 1

        # Add final chunk
        if chunk_start < data_len:
            self._create_chunk(data, chunks, chunk_start, data_len)

        logger.debug(
            f"Rolling hash: {data_len} bytes → {len(chunks)} chunks "
            f"(avg={data_len // len(chunks) if chunks else 0} bytes/chunk)"
        )

        return chunks

    def _create_chunk(
        self, data: bytes, chunks: List[Chunk], start: int, end: int
    ) -> None:
        """
        Create a chunk and add to list

        Args:
            data: Full data bytes
            chunks: List to append chunk to
            start: Start offset
            end: End offset
        """
        chunk_bytes = data[start:end]
        chunk_hash = self._hash_chunk(chunk_bytes)

        chunks.append(
            Chunk(
                data=chunk_bytes,
                offset=start,
                length=len(chunk_bytes),
                hash=chunk_hash,
            )
        )

    def _hash_chunk(self, data: bytes) -> str:
        """
        Generate hash for chunk data

        Args:
            data: Chunk bytes

        Returns:
            Hash hex string
        """
        if self._use_blake3:
            # BLAKE3 is 10x faster than SHA-256
            return self._blake3.blake3(data).hexdigest()
        else:
            # Fallback to SHA-256
            return hashlib.sha256(data).hexdigest()

    def get_boundary_windows(
        self, chunks: List[Chunk], window_size: int = 128
    ) -> List[Tuple[int, str]]:
        """
        Extract boundary overlap windows for correction

        For each boundary between chunks, extract a window of text that
        spans the boundary. This is used to correct token counts that may
        differ when text is split vs whole (due to BPE merges).

        Args:
            chunks: List of chunks
            window_size: Size of window in characters (half on each side)

        Returns:
            List of (boundary_index, window_text) pairs

        Example:
            If chunks are ["ABC", "DEF"], and window_size=2,
            returns [(0, "BCDE")] - 1 char from chunk 0, 1 from chunk 1
        """
        if len(chunks) <= 1:
            return []

        windows = []
        half_window = window_size // 2

        for i in range(len(chunks) - 1):
            # Get text around boundary
            chunk1 = chunks[i]
            chunk2 = chunks[i + 1]

            try:
                # Decode chunks to text
                text1 = chunk1.data.decode("utf-8")
                text2 = chunk2.data.decode("utf-8")

                # Extract window around boundary
                # Take last half_window chars from chunk1
                # and first half_window chars from chunk2
                window_text = text1[-half_window:] + text2[:half_window]

                windows.append((i, window_text))

            except UnicodeDecodeError as e:
                logger.warning(
                    f"Failed to decode boundary window at chunk {i}: {e}, skipping"
                )
                continue

        return windows
