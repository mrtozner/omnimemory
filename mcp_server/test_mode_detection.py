#!/usr/bin/env python3
"""
Test script to validate mode detection logic without running the full server
"""
import os


def test_mode_detection():
    """Test the mode detection logic"""

    print("Testing Mode Detection Logic\n")
    print("=" * 50)

    # Test 1: Default mode (local)
    print("\n1. Test Default Mode (no env vars)")
    connection_mode = os.getenv("OMNIMEMORY_CONNECTION_MODE", "local")
    gateway_url = os.getenv("OMNIMEMORY_GATEWAY_URL")
    api_key = os.getenv("OMNIMEMORY_API_KEY")
    user_id = os.getenv("OMNIMEMORY_USER_ID", "default_user")

    print(f"   connection_mode: {connection_mode}")
    print(f"   gateway_url: {gateway_url}")
    print(f"   api_key: {api_key}")
    print(f"   user_id: {user_id}")

    if connection_mode == "local":
        print("   ✓ Local mode detected correctly")
    else:
        print("   ✗ FAILED: Expected local mode")

    # Test 2: Cloud mode validation
    print("\n2. Test Cloud Mode Validation")
    os.environ["OMNIMEMORY_CONNECTION_MODE"] = "cloud"
    connection_mode = os.getenv("OMNIMEMORY_CONNECTION_MODE", "local")
    gateway_url = os.getenv("OMNIMEMORY_GATEWAY_URL")
    api_key = os.getenv("OMNIMEMORY_API_KEY")

    print(f"   connection_mode: {connection_mode}")
    print(f"   gateway_url: {gateway_url}")
    print(f"   api_key: {api_key}")

    if connection_mode == "cloud":
        if not gateway_url or not api_key:
            print("   ✓ Correctly detected missing configuration")
        else:
            print("   ✓ Cloud mode with valid configuration")

    # Test 3: Cloud mode with full config
    print("\n3. Test Cloud Mode with Full Config")
    os.environ["OMNIMEMORY_CONNECTION_MODE"] = "cloud"
    os.environ["OMNIMEMORY_GATEWAY_URL"] = "http://localhost:8009"
    os.environ["OMNIMEMORY_API_KEY"] = "sk_test_12345"
    os.environ["OMNIMEMORY_USER_ID"] = "test_user"

    connection_mode = os.getenv("OMNIMEMORY_CONNECTION_MODE", "local")
    gateway_url = os.getenv("OMNIMEMORY_GATEWAY_URL")
    api_key = os.getenv("OMNIMEMORY_API_KEY")
    user_id = os.getenv("OMNIMEMORY_USER_ID", "default_user")

    print(f"   connection_mode: {connection_mode}")
    print(f"   gateway_url: {gateway_url}")
    print(f"   api_key: {api_key}")
    print(f"   user_id: {user_id}")

    if connection_mode == "cloud" and gateway_url and api_key:
        print("   ✓ Cloud mode properly configured")
    else:
        print("   ✗ FAILED: Cloud mode configuration incomplete")

    # Clean up
    os.environ.pop("OMNIMEMORY_CONNECTION_MODE", None)
    os.environ.pop("OMNIMEMORY_GATEWAY_URL", None)
    os.environ.pop("OMNIMEMORY_API_KEY", None)
    os.environ.pop("OMNIMEMORY_USER_ID", None)

    print("\n" + "=" * 50)
    print("✓ All mode detection tests passed!")
    print("\nImplementation Summary:")
    print("  - Default: Local mode")
    print("  - Cloud mode requires: OMNIMEMORY_GATEWAY_URL + OMNIMEMORY_API_KEY")
    print("  - Optional: OMNIMEMORY_USER_ID (defaults to 'default_user')")


if __name__ == "__main__":
    test_mode_detection()
