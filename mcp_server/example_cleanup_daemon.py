"""
Example usage of ResultCleanupDaemon

This demonstrates how to integrate the cleanup daemon into your application.
"""

import asyncio
import logging
from result_cleanup_daemon import ResultCleanupDaemon

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


async def main():
    """Example: Start cleanup daemon and let it run"""

    print("=" * 60)
    print("ResultCleanupDaemon Example")
    print("=" * 60)

    # Create cleanup daemon
    daemon = ResultCleanupDaemon(
        result_store=None,  # Can pass ResultStore instance if you have one
        check_interval=60,  # Run cleanup every 60 seconds (for demo)
        cache_dir="~/.omnimemory/cached_results",
        metrics_url="http://localhost:8003",
    )

    print("\n1. Initial stats:")
    stats = daemon.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n2. Starting daemon...")
    await daemon.start()

    print("   Daemon is running in background")
    print(f"   Cache directory: {daemon.cache_dir}")
    print(f"   Check interval: {daemon.check_interval}s")

    # Let it run for a while
    print("\n3. Letting daemon run for 10 seconds...")
    try:
        await asyncio.sleep(10)
    except KeyboardInterrupt:
        print("\n   Interrupted by user")

    print("\n4. Stopping daemon...")
    await daemon.stop()

    print("\n5. Final stats:")
    stats = daemon.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 60)
    print("Example completed")
    print("=" * 60)


async def one_time_cleanup_example():
    """Example: Run cleanup once without background daemon"""

    print("\n" + "=" * 60)
    print("One-Time Cleanup Example")
    print("=" * 60)

    daemon = ResultCleanupDaemon(
        result_store=None,
        cache_dir="~/.omnimemory/cached_results",
    )

    print("\nRunning one-time cleanup...")
    stats = await daemon._cleanup_expired()

    print("\nCleanup results:")
    print(f"   Files checked: {stats['checked_count']}")
    print(f"   Files deleted: {stats['deleted_count']}")
    print(f"   Bytes freed: {stats['freed_bytes']:,}")
    print(f"   MB freed: {stats['freed_bytes'] / (1024 * 1024):.2f}")
    print(f"   Errors: {stats['errors']}")
    print(f"   Duration: {stats['duration_ms']:.1f}ms")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "once":
        # Run one-time cleanup
        asyncio.run(one_time_cleanup_example())
    else:
        # Run background daemon example
        asyncio.run(main())
