#!/usr/bin/env python3
"""
Test the live endpoint with detailed debugging
Assumes metrics service is already running on port 8003
"""

import sys
import tempfile
import os
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).parent / "src"))

from file_hash_cache import FileHashCache

METRICS_SERVICE_URL = "http://localhost:8003"


def test_live_endpoint():
    """Test the live /cache/compressed endpoint"""
    print("\n=== Testing Live Endpoint ===\n")

    # 1. Check if service is running
    try:
        response = requests.get(f"{METRICS_SERVICE_URL}/health", timeout=2)
        print(f"✓ Service is running (status: {response.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"✗ ERROR: Service not running: {e}")
        print("\nPlease start the service first:")
        print("  uvicorn src.metrics_service:app --host 0.0.0.0 --port 8003")
        return False

    # 2. Create a test file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        test_content = """import sys

def hello():
    print("Hello from OmniMemory cache test!")
    return 0

if __name__ == "__main__":
    sys.exit(hello())
"""
        f.write(test_content)
        test_file = f.name

    try:
        print(f"\n✓ Created test file: {test_file}")
        print(f"  Size: {len(test_content)} bytes")

        # 3. Calculate hash and store in cache
        cache = FileHashCache()
        print(f"\n✓ Using cache DB: {cache.db_path}")

        # Read file content (exactly as endpoint will)
        with open(test_file, "r", encoding="utf-8") as f:
            file_content = f.read()

        file_hash = cache.calculate_hash(file_content)
        print(f"✓ File hash: {file_hash}")

        # Simulate 90% compression
        compressed = "COMPRESSED_" + file_content[:30]
        original_size = len(file_content)
        compressed_size = int(original_size * 0.1)

        print(f"\n✓ Storing in cache...")
        print(f"  Original size: {original_size} bytes")
        print(f"  Compressed size: {compressed_size} bytes")

        success = cache.store_compressed_file(
            file_hash=file_hash,
            file_path=test_file,
            compressed_content=compressed,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compressed_size / original_size,
            quality_score=0.95,
        )

        print(f"✓ Stored: {success}")

        # Verify it's in cache
        verify = cache.lookup_compressed_file(file_hash)
        if verify:
            print(f"✓ Verified in cache (access_count={verify['access_count']})")
        else:
            print("✗ ERROR: Not found in cache after storing!")
            return False

        cache.close()

        # 4. Query the endpoint
        print(f"\n✓ Querying endpoint: GET /cache/compressed?path={test_file}")

        response = requests.get(
            f"{METRICS_SERVICE_URL}/cache/compressed",
            params={"path": test_file},
            timeout=5,
        )

        print(f"✓ Response status: {response.status_code}")
        print(f"✓ Response content-type: {response.headers.get('content-type')}")

        if response.status_code == 200:
            data = response.json()

            if data is None:
                print("\n✗ CACHE MISS (Expected cache hit!)")
                print("\nPossible issues:")
                print("  1. Endpoint using different database")
                print("  2. Hash calculation mismatch")
                print("  3. File content changed")

                # Debug: Re-read file and calculate hash
                with open(test_file, "r", encoding="utf-8") as f:
                    current_content = f.read()

                cache_debug = FileHashCache()
                current_hash = cache_debug.calculate_hash(current_content)
                print(f"\nDebug info:")
                print(f"  Original hash: {file_hash}")
                print(f"  Current hash:  {current_hash}")
                print(f"  Match: {file_hash == current_hash}")
                cache_debug.close()

                return False

            print("\n✓ CACHE HIT!")
            print("\nResponse data:")
            print(f"  cached: {data.get('cached')}")
            print(f"  file_hash: {data.get('file_hash')}")
            print(f"  original_size: {data.get('original_size')}")
            print(f"  compressed_size: {data.get('compressed_size')}")
            print(f"  compression_ratio: {data.get('compression_ratio'):.1%}")

            # Calculate savings
            tokens_saved = data["original_size"] - data["compressed_size"]
            savings_percent = (tokens_saved / data["original_size"]) * 100
            print(f"\n✓ Token savings: {tokens_saved} tokens ({savings_percent:.1f}%)")

            # Verify hash matches
            if data.get("file_hash") == file_hash:
                print(f"✓ Hash matches!")
                return True
            else:
                print(f"✗ Hash mismatch!")
                print(f"  Expected: {file_hash}")
                print(f"  Got:      {data.get('file_hash')}")
                return False

        else:
            print(f"\n✗ ERROR: HTTP {response.status_code}")
            print(f"  Response: {response.text}")
            return False

    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)
            print(f"\n✓ Cleaned up test file")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LIVE ENDPOINT TEST FOR /cache/compressed")
    print("=" * 70)
    print("\nNOTE: This test requires the metrics service to be running!")
    print("Start it with: uvicorn src.metrics_service:app --port 8003")
    print("=" * 70)

    success = test_live_endpoint()

    if success:
        print("\n" + "=" * 70)
        print("SUCCESS: Endpoint working correctly!")
        print("=" * 70 + "\n")
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("FAILED: See errors above")
        print("=" * 70 + "\n")
        sys.exit(1)
