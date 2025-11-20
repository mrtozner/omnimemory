import asyncio
from src.tokenizer import OmniTokenizer
from src.cache_manager import ThreeTierCache
from src.config import CacheConfig


async def test_tokenizer():
    # Initialize
    cache = ThreeTierCache(config=CacheConfig(l1_enabled=True, l2_enabled=True))
    tokenizer = OmniTokenizer(cache_manager=cache, enable_cdc=False)

    test_text = "Hello, world! This is a test of the enterprise tokenizer system."

    # Test different models
    models = ["gpt-4", "claude-3-5-sonnet", "gemini-1.5-pro", "qwen2.5"]

    print("\n🧪 Testing Multi-Model Tokenization\n")
    for model in models:
        try:
            count = await tokenizer.count(model, test_text)
            print(f"✓ {model:25} → {count.count} tokens")
        except Exception as e:
            print(f"✗ {model:25} → Error: {e}")

    print("\n✅ Tokenizer tests complete\n")


asyncio.run(test_tokenizer())
