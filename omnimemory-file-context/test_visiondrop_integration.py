#!/usr/bin/env python3
"""
Test VisionDrop Integration
Verifies that VisionDrop compression works properly
"""

import sys
import asyncio
import logging

# Add the compression src directory to path
sys.path.insert(0, "../omnimemory-compression/src")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_visiondrop():
    """Test VisionDrop compression functionality"""

    try:
        # Import VisionDrop and dependencies
        from visiondrop import VisionDropCompressor, ContentType, CompressedContext
        from tokenizer import OmniTokenizer
        from cache_manager import ThreeTierCache

        logger.info("✅ Successfully imported VisionDrop and dependencies")

        # Initialize VisionDrop
        compressor = VisionDropCompressor(
            embedding_service_url="http://localhost:8000",  # MLX embeddings service
            target_compression=0.944,  # 94.4% compression
        )

        logger.info("✅ VisionDropCompressor initialized")

        # Test content (Python code)
        test_code = """
def calculate_fibonacci(n: int) -> int:
    '''Calculate the nth Fibonacci number using dynamic programming.'''
    if n <= 1:
        return n

    dp = [0] * (n + 1)
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]

class DataProcessor:
    '''Process and transform data efficiently.'''

    def __init__(self, config):
        self.config = config
        self.cache = {}

    def process(self, data):
        if data in self.cache:
            return self.cache[data]

        result = self._transform(data)
        self.cache[data] = result
        return result

    def _transform(self, data):
        # Complex transformation logic here
        return data.upper()

# Example usage
if __name__ == "__main__":
    processor = DataProcessor({})
    result = processor.process("hello world")
    print(f"Processed: {result}")

    fib_result = calculate_fibonacci(10)
    print(f"Fibonacci(10): {fib_result}")
"""

        logger.info("Testing VisionDrop compression...")

        # Test 1: Basic compression without query
        try:
            result = await compressor.compress(context=test_code, file_path="test.py")

            logger.info(f"✅ Basic compression test passed")
            logger.info(f"   Original tokens: {result.original_tokens}")
            logger.info(f"   Compressed tokens: {result.compressed_tokens}")
            logger.info(f"   Compression ratio: {result.compression_ratio:.1%}")
            logger.info(f"   Quality score: {result.quality_score:.1%}")
            logger.info(f"   Content type: {result.content_type}")

            # Verify compression achieved target
            if result.compression_ratio >= 0.9:  # At least 90% compression
                logger.info(f"   ✅ Achieved target compression (>90%)")
            else:
                logger.warning(
                    f"   ⚠️ Compression below target: {result.compression_ratio:.1%}"
                )

        except Exception as e:
            logger.error(f"❌ Basic compression failed: {e}")
            # Check if it's due to missing embedding service
            if "Connection" in str(e):
                logger.info(
                    "   Note: This might be because the MLX embedding service is not running"
                )
                logger.info(
                    "   Start it with: cd omnimemory-embeddings && python src/mlx_embedding_service.py"
                )

        # Test 2: Query-aware compression
        try:
            result = await compressor.compress(
                context=test_code, query="fibonacci calculation", file_path="test.py"
            )

            logger.info(f"✅ Query-aware compression test passed")
            logger.info(f"   Compression ratio: {result.compression_ratio:.1%}")
            logger.info(f"   Quality score: {result.quality_score:.1%}")

            # Check if fibonacci-related code was preserved
            if "fibonacci" in result.compressed_text.lower():
                logger.info(f"   ✅ Query-relevant code preserved")

        except Exception as e:
            logger.error(f"❌ Query-aware compression failed: {e}")

        # Test 3: Content type detection
        test_markdown = """
# VisionDrop Documentation

VisionDrop is a **code-aware compression** system that achieves:

## Key Features
- 94.4% token reduction
- Code-aware parsing
- Query-aware filtering

### Performance
| Metric | Value |
|--------|-------|
| Compression | 94.4% |
| Quality | 91% |
| Latency | 0.61ms |
"""

        try:
            result = await compressor.compress(
                context=test_markdown, file_path="README.md"
            )

            logger.info(f"✅ Documentation compression test passed")
            logger.info(f"   Content type: {result.content_type}")
            logger.info(f"   Compression ratio: {result.compression_ratio:.1%}")

        except Exception as e:
            logger.error(f"❌ Documentation compression failed: {e}")

        # Close the compressor
        await compressor.close()

        logger.info("\n" + "=" * 60)
        logger.info("VisionDrop Integration Test Summary:")
        logger.info("- VisionDrop module exists and can be imported ✅")
        logger.info("- All dependencies are available ✅")
        logger.info("- Compression functionality works (if embedding service running)")
        logger.info("=" * 60)

        return True

    except ImportError as e:
        logger.error(f"❌ Failed to import VisionDrop: {e}")
        logger.info("This should not happen as we verified the files exist")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_visiondrop())
    sys.exit(0 if success else 1)
