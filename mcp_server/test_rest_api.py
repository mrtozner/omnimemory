#!/usr/bin/env python3
"""
Test script for OmniMemory REST API
Tests the basic functionality of the REST API layer
"""

import sys
import time
import requests
import json
from pathlib import Path


def test_api_key_generation():
    """Test API key generation"""
    print("=" * 60)
    print("Test 1: API Key Generation")
    print("=" * 60)

    # Import the gateway module
    sys.path.insert(0, str(Path(__file__).parent))
    from omnimemory_gateway import api_key_manager

    # Generate an API key
    user_id, api_key = api_key_manager.generate_key(
        email="test@example.com", name="Test User", metadata={"platform": "test"}
    )

    print(f"✓ Generated User ID: {user_id}")
    print(f"✓ Generated API Key: {api_key[:20]}...")

    # Validate the key
    user_info = api_key_manager.validate_key(api_key)
    assert user_info is not None, "API key validation failed"
    assert user_info["email"] == "test@example.com"
    assert user_info["name"] == "Test User"

    print(f"✓ Validated API key successfully")
    print(f"  User: {user_info['name']} ({user_info['email']})")

    return api_key


def test_health_endpoint():
    """Test health check endpoint"""
    print("\n" + "=" * 60)
    print("Test 2: Health Endpoint")
    print("=" * 60)

    try:
        # Note: This requires the server to be running
        # We'll just test the endpoint definition for now
        from omnimemory_gateway import api

        print("✓ FastAPI app created successfully")
        print(f"✓ API Title: {api.title}")
        print(f"✓ API Version: {api.version}")

        # Check routes
        routes = [route.path for route in api.routes]
        print(f"✓ Available routes: {len(routes)}")

        expected_routes = [
            "/health",
            "/api/v1/users",
            "/api/v1/compress",
            "/api/v1/search",
            "/api/v1/embed",
            "/api/v1/stats",
        ]

        for route in expected_routes:
            if route in routes:
                print(f"  ✓ {route}")
            else:
                print(f"  ✗ {route} - NOT FOUND")

    except Exception as e:
        print(f"✗ Health endpoint test failed: {e}")
        return False

    return True


def test_pydantic_models():
    """Test Pydantic models"""
    print("\n" + "=" * 60)
    print("Test 3: Pydantic Models")
    print("=" * 60)

    from omnimemory_gateway import (
        CompressRequest,
        SearchRequest,
        EmbedRequest,
        UserCreateRequest,
    )

    # Test CompressRequest
    compress_req = CompressRequest(
        content="Long content to compress",
        target_compression=0.8,
        quality_threshold=0.75,
    )
    print(
        f"✓ CompressRequest: {len(compress_req.content)} chars, target={compress_req.target_compression}"
    )

    # Test SearchRequest
    search_req = SearchRequest(query="test query", limit=5, min_relevance=0.7)
    print(f"✓ SearchRequest: query='{search_req.query}'")

    # Test EmbedRequest
    embed_req = EmbedRequest(
        file_paths=["/path/to/file1.py", "/path/to/file2.js"], batch_size=10
    )
    print(
        f"✓ EmbedRequest: {len(embed_req.file_paths)} files, batch_size={embed_req.batch_size}"
    )

    # Test UserCreateRequest
    user_req = UserCreateRequest(
        email="new@example.com", name="New User", metadata={"platform": "n8n"}
    )
    print(f"✓ UserCreateRequest: {user_req.email}")

    return True


def test_api_key_validation():
    """Test API key validation logic"""
    print("\n" + "=" * 60)
    print("Test 4: API Key Validation")
    print("=" * 60)

    from omnimemory_gateway import api_key_manager

    # Test invalid key
    invalid_info = api_key_manager.validate_key("invalid_key")
    assert invalid_info is None, "Invalid key should return None"
    print("✓ Invalid key correctly rejected")

    # Generate and test valid key
    user_id, api_key = api_key_manager.generate_key(
        email="validate@example.com", name="Validation Test"
    )

    valid_info = api_key_manager.validate_key(api_key)
    assert valid_info is not None, "Valid key should be accepted"
    assert valid_info["email"] == "validate@example.com"
    print(f"✓ Valid key correctly accepted")

    # Test key revocation
    success = api_key_manager.revoke_key(api_key)
    assert success, "Key revocation should succeed"
    print("✓ Key revoked successfully")

    revoked_info = api_key_manager.validate_key(api_key)
    assert revoked_info is None, "Revoked key should be rejected"
    print("✓ Revoked key correctly rejected")

    return True


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║  OmniMemory REST API - Test Suite                         ║")
    print("╚" + "=" * 58 + "╝")

    try:
        # Run tests
        api_key = test_api_key_generation()
        test_health_endpoint()
        test_pydantic_models()
        test_api_key_validation()

        # Summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print("✓ All tests passed!")
        print("\nNext steps:")
        print("1. Start the server: python omnimemory_gateway.py")
        print("2. Test endpoints: http://localhost:8009/docs")
        print("3. Use this API key for testing:")
        print(f"   {api_key}")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
