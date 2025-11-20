#!/usr/bin/env python3
"""
End-to-End Integration Test for /cache/compressed Endpoint
Tests the actual HTTP endpoint with a running service
"""

import sys
import tempfile
import os
import time
import signal
import subprocess
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).parent / "src"))

from file_hash_cache import FileHashCache


METRICS_SERVICE_URL = "http://localhost:8003"
METRICS_SERVICE_PROCESS = None


def start_metrics_service():
    """Start the metrics service in background"""
    global METRICS_SERVICE_PROCESS

    print("Starting metrics service...")

    # Start uvicorn server
    METRICS_SERVICE_PROCESS = subprocess.Popen(
        ["uvicorn", "src.metrics_service:app", "--host", "0.0.0.0", "--port", "8003"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for service to start
    max_attempts = 30
    for i in range(max_attempts):
        try:
            response = requests.get(f"{METRICS_SERVICE_URL}/health", timeout=1)
            if response.status_code == 200:
                print(f"✓ Metrics service started (pid={METRICS_SERVICE_PROCESS.pid})")
                return True
        except requests.exceptions.RequestException:
            time.sleep(0.5)

    print("✗ Failed to start metrics service")
    return False


def stop_metrics_service():
    """Stop the metrics service"""
    global METRICS_SERVICE_PROCESS

    if METRICS_SERVICE_PROCESS:
        print("Stopping metrics service...")
        METRICS_SERVICE_PROCESS.send_signal(signal.SIGTERM)
        METRICS_SERVICE_PROCESS.wait(timeout=5)
        print("✓ Metrics service stopped")


def test_endpoint_cache_hit():
    """Test /cache/compressed endpoint with cache hit"""
    print("\n=== Testing Endpoint Cache Hit ===\n")

    # Create a test file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        test_content = """import os
import sys

def main():
    print("OmniMemory Cache Test")
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""
        f.write(test_content)
        test_file = f.name

    try:
        print(f"✓ Created test file: {test_file}")
        print(f"  Size: {len(test_content)} bytes")

        # Store in cache using FileHashCache
        cache = FileHashCache()
        file_hash = cache.calculate_hash(test_content)

        # Simulate compression
        compressed = "COMPRESSED_VERSION"
        cache.store_compressed_file(
            file_hash=file_hash,
            file_path=test_file,
            compressed_content=compressed,
            original_size=len(test_content),
            compressed_size=len(compressed),
            compression_ratio=len(compressed) / len(test_content),
            quality_score=0.95,
        )

        print(f"✓ Stored in cache (hash={file_hash[:16]}...)")
        cache.close()

        # Query endpoint
        print(f"\nQuerying endpoint: GET /cache/compressed?path={test_file}")

        response = requests.get(
            f"{METRICS_SERVICE_URL}/cache/compressed",
            params={"path": test_file},
            timeout=5,
        )

        print(f"✓ Response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            if data is None:
                print("✗ FAIL: Got null (cache miss) instead of cache hit")
                return False

            print("✓ Response data:")
            print(f"  cached: {data.get('cached')}")
            print(f"  original_size: {data.get('original_size')}")
            print(f"  compressed_size: {data.get('compressed_size')}")
            print(f"  compression_ratio: {data.get('compression_ratio'):.1%}")
            print(f"  file_hash: {data.get('file_hash')[:16]}...")

            # Verify data
            if data.get("cached") == True and data.get("file_hash") == file_hash:
                tokens_saved = data["original_size"] - data["compressed_size"]
                print(f"\n✓ SUCCESS: Cache hit with {tokens_saved} tokens saved")
                return True
            else:
                print("✗ FAIL: Response data incorrect")
                return False
        else:
            print(f"✗ FAIL: HTTP {response.status_code}")
            return False

    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)


def test_endpoint_cache_miss():
    """Test /cache/compressed endpoint with cache miss"""
    print("\n=== Testing Endpoint Cache Miss ===\n")

    # Create a file that's NOT in cache
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This file is not cached\n")
        test_file = f.name

    try:
        print(f"✓ Created uncached file: {test_file}")

        # Query endpoint (should miss)
        print(f"\nQuerying endpoint: GET /cache/compressed?path={test_file}")

        response = requests.get(
            f"{METRICS_SERVICE_URL}/cache/compressed",
            params={"path": test_file},
            timeout=5,
        )

        print(f"✓ Response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            if data is None:
                print("✓ SUCCESS: Cache miss (returned null as expected)")
                return True
            else:
                print(f"✗ FAIL: Expected null, got: {data}")
                return False
        else:
            print(f"✗ FAIL: HTTP {response.status_code}")
            return False

    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)


def test_endpoint_file_not_found():
    """Test /cache/compressed endpoint with non-existent file"""
    print("\n=== Testing Endpoint File Not Found ===\n")

    non_existent = "/tmp/omnimemory_nonexistent_file_xyz123.txt"
    print(f"Testing with non-existent file: {non_existent}")

    response = requests.get(
        f"{METRICS_SERVICE_URL}/cache/compressed",
        params={"path": non_existent},
        timeout=5,
    )

    print(f"✓ Response status: {response.status_code}")

    if response.status_code == 200 and response.json() is None:
        print("✓ SUCCESS: Returned null for non-existent file")
        return True
    else:
        print(f"✗ FAIL: Unexpected response: {response.json()}")
        return False


def main():
    """Run end-to-end integration tests"""
    print("\n" + "=" * 60)
    print("E2E INTEGRATION TEST FOR /cache/compressed ENDPOINT")
    print("=" * 60)

    # Start metrics service
    if not start_metrics_service():
        print("\n✗ ERROR: Could not start metrics service")
        return 1

    try:
        # Run tests
        results = []
        results.append(("Cache Hit", test_endpoint_cache_hit()))
        results.append(("Cache Miss", test_endpoint_cache_miss()))
        results.append(("File Not Found", test_endpoint_file_not_found()))

        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60 + "\n")

        passed = sum(1 for _, result in results if result)
        total = len(results)

        for name, result in results:
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{status}: {name}")

        print(f"\n{passed}/{total} tests passed\n")

        return 0 if passed == total else 1

    finally:
        stop_metrics_service()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        stop_metrics_service()
        sys.exit(1)
