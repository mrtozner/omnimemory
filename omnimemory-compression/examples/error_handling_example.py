"""
Example: Error Handling with OmniMemory SDK

Demonstrates how to handle different types of errors when using the SDK.
"""

import asyncio
from omnimemory import (
    OmniMemory,
    OmniMemoryError,
    QuotaExceededError,
    AuthenticationError,
    CompressionError,
    RateLimitError,
    ServiceUnavailableError,
    InvalidRequestError,
)


async def handle_authentication_error():
    """Example: Handling authentication errors"""
    print("=== Authentication Error Example ===\n")

    try:
        # Try to use invalid API key
        client = OmniMemory(api_key="invalid_key", base_url="http://localhost:8001")
        result = await client.compress(
            context="This will fail due to invalid API key",
            target_compression=0.5,
        )
        await client.close()
    except AuthenticationError as e:
        print(f"Authentication failed: {e}")
        print("Solution: Check your API key and try again\n")
    except Exception as e:
        print(f"Unexpected error: {e}\n")


async def handle_quota_exceeded():
    """Example: Handling quota exceeded errors"""
    print("=== Quota Exceeded Example ===\n")

    try:
        client = OmniMemory(api_key="om_test_key", base_url="http://localhost:8001")
        result = await client.compress(
            context="This might fail if quota is exceeded",
            target_compression=0.5,
        )
        await client.close()
    except QuotaExceededError as e:
        print(f"Quota exceeded: {e}")
        print("Solution: Upgrade your plan or wait until quota resets\n")
    except Exception as e:
        print(f"Note: Quota limit not reached yet (expected in development)\n")


async def handle_rate_limit():
    """Example: Handling rate limit errors"""
    print("=== Rate Limit Example ===\n")

    try:
        client = OmniMemory(base_url="http://localhost:8001")

        # Make multiple rapid requests
        for i in range(100):
            result = await client.compress(
                context=f"Request {i} - testing rate limits",
                target_compression=0.5,
            )

        await client.close()
    except RateLimitError as e:
        print(f"Rate limit exceeded: {e}")
        if hasattr(e, "retry_after") and e.retry_after:
            print(f"Retry after: {e.retry_after} seconds")
        print("Solution: Implement exponential backoff or reduce request rate\n")
    except Exception as e:
        print(f"Note: Rate limit not reached (expected in development)\n")


async def handle_invalid_request():
    """Example: Handling invalid request errors"""
    print("=== Invalid Request Example ===\n")

    try:
        client = OmniMemory(base_url="http://localhost:8001")

        # Try invalid compression ratio
        result = await client.compress(
            context="Test content", target_compression=1.5  # Invalid: > 1.0
        )

        await client.close()
    except InvalidRequestError as e:
        print(f"Invalid request: {e}")
        print("Solution: Check API documentation for valid parameter ranges\n")
    except Exception as e:
        print(f"Error: {e}\n")


async def handle_service_unavailable():
    """Example: Handling service unavailable errors"""
    print("=== Service Unavailable Example ===\n")

    try:
        # Try to connect to non-existent service
        client = OmniMemory(base_url="http://localhost:9999", timeout=2.0)
        result = await client.compress(
            context="This will fail due to unavailable service",
            target_compression=0.5,
        )
        await client.close()
    except ServiceUnavailableError as e:
        print(f"Service unavailable: {e}")
        print("Solution: Check service status or try again later\n")
    except Exception as e:
        print(f"Connection error: {type(e).__name__}: {e}")
        print("Solution: Verify service is running and accessible\n")


async def robust_compression_with_retry():
    """Example: Robust compression with automatic retry"""
    print("=== Robust Compression with Retry ===\n")

    max_retries = 3
    retry_delay = 1.0

    for attempt in range(max_retries):
        try:
            client = OmniMemory(base_url="http://localhost:8001")

            result = await client.compress(
                context="""
                Machine learning is a subset of artificial intelligence that enables
                systems to learn and improve from experience without being explicitly
                programmed. It focuses on developing computer programs that can access
                data and use it to learn for themselves.
                """,
                target_compression=0.5,
            )

            print(f"✓ Compression successful on attempt {attempt + 1}")
            print(f"  Original: {result.original_tokens} tokens")
            print(f"  Compressed: {result.compressed_tokens} tokens")
            print(f"  Ratio: {result.compression_ratio:.2%}\n")

            await client.close()
            break

        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = (
                    e.retry_after
                    if hasattr(e, "retry_after") and e.retry_after
                    else retry_delay
                )
                print(f"Rate limited. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed after {max_retries} attempts: {e}\n")

        except ServiceUnavailableError as e:
            if attempt < max_retries - 1:
                print(f"Service unavailable. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"Service unavailable after {max_retries} attempts: {e}\n")

        except (AuthenticationError, QuotaExceededError, InvalidRequestError) as e:
            # These errors won't be fixed by retrying
            print(f"Non-retryable error: {type(e).__name__}: {e}\n")
            break

        except OmniMemoryError as e:
            print(f"OmniMemory error: {e}\n")
            break

        except Exception as e:
            print(f"Unexpected error: {type(e).__name__}: {e}\n")
            break


async def graceful_degradation_example():
    """Example: Graceful degradation when compression fails"""
    print("=== Graceful Degradation Example ===\n")

    long_context = """
    Natural language processing (NLP) is a subfield of linguistics, computer science,
    and artificial intelligence concerned with the interactions between computers and
    human language. Modern NLP techniques use machine learning to extract meaning from
    text and generate human-like responses.
    """

    try:
        client = OmniMemory(base_url="http://localhost:8001")
        result = await client.compress(
            context=long_context, query="What is NLP?", target_compression=0.5
        )

        # Use compressed version
        final_context = result.compressed_text
        print(f"✓ Using compressed context ({result.compressed_tokens} tokens)")

        await client.close()

    except (CompressionError, ServiceUnavailableError, OmniMemoryError) as e:
        print(f"⚠ Compression failed: {e}")
        print("  Falling back to original context")

        # Fall back to original context
        final_context = long_context

    # Continue with processing
    print(f"\nFinal context length: {len(final_context)} characters")
    print(f"Context preview: {final_context[:100]}...\n")


async def context_manager_error_handling():
    """Example: Error handling with context manager"""
    print("=== Context Manager Error Handling ===\n")

    try:
        async with OmniMemory(base_url="http://localhost:8001") as client:
            result = await client.compress(
                context="Test content for context manager",
                target_compression=0.5,
            )
            print(f"✓ Compression successful: {result.compressed_tokens} tokens")

    except OmniMemoryError as e:
        print(f"Compression failed: {e}")
        print("Note: Client automatically closed due to context manager\n")
    except Exception as e:
        print(f"Unexpected error: {e}\n")


if __name__ == "__main__":
    print("OmniMemory SDK - Error Handling Examples\n")
    print("=" * 60)
    print()

    # Run examples
    asyncio.run(handle_authentication_error())
    asyncio.run(handle_quota_exceeded())
    asyncio.run(handle_rate_limit())
    asyncio.run(handle_invalid_request())
    asyncio.run(handle_service_unavailable())
    asyncio.run(robust_compression_with_retry())
    asyncio.run(graceful_degradation_example())
    asyncio.run(context_manager_error_handling())

    print("=" * 60)
    print("\nAll error handling examples completed!")
