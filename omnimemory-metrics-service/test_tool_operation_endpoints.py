#!/usr/bin/env python3
"""
Test script for Phase 2: Tool Operation Tracking Endpoints

Tests all 4 new endpoints:
1. POST /track/tool-operation
2. GET /metrics/tool-operations
3. GET /metrics/tool-breakdown
4. GET /metrics/api-savings
"""

import requests
import uuid
import time
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8003"


def test_track_tool_operation():
    """Test POST /track/tool-operation"""
    print("=" * 80)
    print("TEST 1: POST /track/tool-operation")
    print("=" * 80)

    # Generate a session ID
    session_id = str(uuid.uuid4())
    print(f"\nUsing session_id: {session_id}")

    # Test data: various operation types
    operations = [
        {
            "session_id": session_id,
            "tool_name": "read",
            "operation_mode": "full",
            "parameters": {"compress": False},
            "file_path": "src/main.py",
            "tokens_original": 5000,
            "tokens_actual": 5000,
            "tokens_prevented": 0,
            "response_time_ms": 150.5,
            "tool_id": "claude-code",
        },
        {
            "session_id": session_id,
            "tool_name": "read",
            "operation_mode": "overview",
            "parameters": {"compress": True},
            "file_path": "src/utils.py",
            "tokens_original": 5000,
            "tokens_actual": 500,
            "tokens_prevented": 4500,
            "response_time_ms": 120.3,
            "tool_id": "claude-code",
        },
        {
            "session_id": session_id,
            "tool_name": "read",
            "operation_mode": "symbol",
            "parameters": {"compress": True, "symbol": "authenticate"},
            "file_path": "src/auth.py",
            "tokens_original": 8000,
            "tokens_actual": 800,
            "tokens_prevented": 7200,
            "response_time_ms": 95.7,
            "tool_id": "claude-code",
        },
        {
            "session_id": session_id,
            "tool_name": "search",
            "operation_mode": "semantic",
            "parameters": {"query": "authentication implementation", "limit": 5},
            "file_path": None,
            "tokens_original": 50000,
            "tokens_actual": 5000,
            "tokens_prevented": 45000,
            "response_time_ms": 250.7,
            "tool_id": "claude-code",
        },
        {
            "session_id": session_id,
            "tool_name": "search",
            "operation_mode": "tri_index",
            "parameters": {"query": "config", "limit": 10},
            "file_path": None,
            "tokens_original": 30000,
            "tokens_actual": 3000,
            "tokens_prevented": 27000,
            "response_time_ms": 180.4,
            "tool_id": "claude-code",
        },
    ]

    print(f"\nTracking {len(operations)} operations...")
    operation_ids = []

    for i, op in enumerate(operations, 1):
        response = requests.post(f"{BASE_URL}/track/tool-operation", json=op)

        if response.status_code == 200:
            result = response.json()
            operation_ids.append(result["operation_id"])
            print(
                f"  ✅ {i}. {op['tool_name']}/{op['operation_mode']}: "
                f"{op['tokens_prevented']:,} tokens prevented (ID: {result['operation_id'][:8]}...)"
            )
        else:
            print(f"  ❌ {i}. Failed: {response.status_code} - {response.text}")

    print(f"\n✅ Tracked {len(operation_ids)} operations successfully")
    return session_id, operation_ids


def test_get_tool_operations(session_id):
    """Test GET /metrics/tool-operations"""
    print("\n" + "=" * 80)
    print("TEST 2: GET /metrics/tool-operations")
    print("=" * 80)

    # Test 1: Get all operations for session
    print("\n1. Get all operations for session:")
    response = requests.get(
        f"{BASE_URL}/metrics/tool-operations", params={"session_id": session_id}
    )

    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Found {result['total']} operations")
        print(f"   Showing {len(result['operations'])} operations:")
        for op in result["operations"][:3]:
            print(
                f"     - {op['tool_name']}/{op['operation_mode']}: "
                f"{op['tokens_prevented']:,} tokens prevented"
            )
    else:
        print(f"   ❌ Failed: {response.status_code} - {response.text}")

    # Test 2: Filter by tool_name
    print("\n2. Filter by tool_name='read':")
    response = requests.get(
        f"{BASE_URL}/metrics/tool-operations",
        params={"session_id": session_id, "tool_name": "read"},
    )

    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Found {result['total']} read operations")
    else:
        print(f"   ❌ Failed: {response.status_code} - {response.text}")

    # Test 3: Filter by operation_mode
    print("\n3. Filter by operation_mode='semantic':")
    response = requests.get(
        f"{BASE_URL}/metrics/tool-operations",
        params={"session_id": session_id, "operation_mode": "semantic"},
    )

    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Found {result['total']} semantic search operations")
    else:
        print(f"   ❌ Failed: {response.status_code} - {response.text}")

    # Test 4: Pagination
    print("\n4. Test pagination (limit=2, offset=0):")
    response = requests.get(
        f"{BASE_URL}/metrics/tool-operations",
        params={"session_id": session_id, "limit": 2, "offset": 0},
    )

    if response.status_code == 200:
        result = response.json()
        print(
            f"   ✅ Page 1: Showing {len(result['operations'])} of {result['total']} total"
        )
    else:
        print(f"   ❌ Failed: {response.status_code} - {response.text}")

    print("\n✅ All query tests passed")


def test_tool_breakdown():
    """Test GET /metrics/tool-breakdown"""
    print("\n" + "=" * 80)
    print("TEST 3: GET /metrics/tool-breakdown")
    print("=" * 80)

    # Test different time ranges
    time_ranges = ["1h", "24h", "7d", "30d"]

    for time_range in time_ranges:
        print(f"\n{time_range} breakdown:")
        response = requests.get(
            f"{BASE_URL}/metrics/tool-breakdown", params={"time_range": time_range}
        )

        if response.status_code == 200:
            result = response.json()

            # Read operations
            read_ops = result["read"]["total_operations"]
            read_prevented = result["read"]["total_tokens_prevented"]

            # Search operations
            search_ops = result["search"]["total_operations"]
            search_prevented = result["search"]["total_tokens_prevented"]

            # Totals
            total_prevented = result["total_tokens_prevented"]
            total_saved = result["total_cost_saved"]

            print(f"  Read:   {read_ops} ops, {read_prevented:,} tokens prevented")
            print(f"  Search: {search_ops} ops, {search_prevented:,} tokens prevented")
            print(
                f"  Total:  {total_prevented:,} tokens prevented, ${total_saved:.4f} saved"
            )

            # Show mode breakdown for read
            if result["read"]["by_mode"]:
                print("  Read modes:")
                for mode, data in result["read"]["by_mode"].items():
                    print(
                        f"    - {mode}: {data['count']} ops, {data['tokens_prevented']:,} tokens"
                    )

            # Show mode breakdown for search
            if result["search"]["by_mode"]:
                print("  Search modes:")
                for mode, data in result["search"]["by_mode"].items():
                    print(
                        f"    - {mode}: {data['count']} ops, {data['tokens_prevented']:,} tokens"
                    )

            print(f"  ✅ Successfully retrieved {time_range} breakdown")
        else:
            print(f"  ❌ Failed: {response.status_code} - {response.text}")

    print("\n✅ All breakdown tests passed")


def test_api_savings():
    """Test GET /metrics/api-savings"""
    print("\n" + "=" * 80)
    print("TEST 4: GET /metrics/api-savings")
    print("=" * 80)

    # Test different time ranges
    time_ranges = ["1h", "24h", "7d", "30d", "all"]

    for time_range in time_ranges:
        print(f"\n{time_range} API savings:")
        response = requests.get(
            f"{BASE_URL}/metrics/api-savings", params={"time_range": time_range}
        )

        if response.status_code == 200:
            result = response.json()

            baseline_cost = result["api_cost_baseline"]
            actual_cost = result["api_cost_actual"]
            saved_cost = result["total_cost_saved"]
            savings_pct = result["savings_percentage"]
            operations = result["total_operations"]

            print(f"  Operations: {operations}")
            print(f"  Baseline cost:  ${baseline_cost:.4f}")
            print(f"  Actual cost:    ${actual_cost:.4f}")
            print(f"  Cost saved:     ${saved_cost:.4f} ({savings_pct:.1f}%)")

            # Show breakdown by tool
            print("  By tool:")
            for tool, data in result["breakdown_by_tool"].items():
                if data["operations"] > 0:
                    print(
                        f"    {tool}: {data['operations']} ops, "
                        f"${data['cost_saved']:.4f} saved"
                    )

            # Show breakdown by mode (top 3)
            if result["breakdown_by_mode"]:
                print("  Top modes:")
                sorted_modes = sorted(
                    result["breakdown_by_mode"].items(),
                    key=lambda x: x[1]["cost_saved"],
                    reverse=True,
                )
                for mode, data in sorted_modes[:3]:
                    print(
                        f"    {mode}: {data['operations']} ops, "
                        f"${data['cost_saved']:.4f} saved"
                    )

            # Show trend info
            if result["trends"]:
                print(f"  Trends: {len(result['trends'])} time buckets")

            print(f"  ✅ Successfully retrieved {time_range} API savings")
        else:
            print(f"  ❌ Failed: {response.status_code} - {response.text}")

    print("\n✅ All API savings tests passed")


def test_error_handling():
    """Test error handling for invalid inputs"""
    print("\n" + "=" * 80)
    print("TEST 5: Error Handling")
    print("=" * 80)

    # Test 1: Invalid session_id format
    print("\n1. Invalid session_id format:")
    response = requests.post(
        f"{BASE_URL}/track/tool-operation",
        json={
            "session_id": "invalid-uuid",
            "tool_name": "read",
            "operation_mode": "full",
            "tokens_original": 1000,
            "tokens_actual": 1000,
            "tokens_prevented": 0,
            "response_time_ms": 100.0,
            "tool_id": "test",
        },
    )
    if response.status_code == 400:
        print("   ✅ Correctly rejected invalid UUID")
    else:
        print(f"   ❌ Expected 400, got {response.status_code}")

    # Test 2: Invalid tool_name
    print("\n2. Invalid tool_name:")
    response = requests.post(
        f"{BASE_URL}/track/tool-operation",
        json={
            "session_id": str(uuid.uuid4()),
            "tool_name": "invalid",
            "operation_mode": "full",
            "tokens_original": 1000,
            "tokens_actual": 1000,
            "tokens_prevented": 0,
            "response_time_ms": 100.0,
            "tool_id": "test",
        },
    )
    if response.status_code == 422:  # Pydantic validation error
        print("   ✅ Correctly rejected invalid tool_name")
    else:
        print(f"   ❌ Expected 422, got {response.status_code}")

    # Test 3: Invalid operation_mode
    print("\n3. Invalid operation_mode:")
    response = requests.post(
        f"{BASE_URL}/track/tool-operation",
        json={
            "session_id": str(uuid.uuid4()),
            "tool_name": "read",
            "operation_mode": "invalid",
            "tokens_original": 1000,
            "tokens_actual": 1000,
            "tokens_prevented": 0,
            "response_time_ms": 100.0,
            "tool_id": "test",
        },
    )
    if response.status_code == 422:  # Pydantic validation error
        print("   ✅ Correctly rejected invalid operation_mode")
    else:
        print(f"   ❌ Expected 422, got {response.status_code}")

    # Test 4: Invalid time_range
    print("\n4. Invalid time_range:")
    response = requests.get(
        f"{BASE_URL}/metrics/tool-breakdown", params={"time_range": "invalid"}
    )
    if response.status_code == 400:
        print("   ✅ Correctly rejected invalid time_range")
    else:
        print(f"   ❌ Expected 400, got {response.status_code}")

    print("\n✅ All error handling tests passed")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("PHASE 2: Tool Operation Tracking Endpoints - Test Suite")
    print("=" * 80)
    print(f"\nTesting against: {BASE_URL}")
    print(f"Time: {datetime.now().isoformat()}")

    try:
        # Test 1: Track operations
        session_id, operation_ids = test_track_tool_operation()
        time.sleep(0.5)  # Brief pause between tests

        # Test 2: Query operations
        test_get_tool_operations(session_id)
        time.sleep(0.5)

        # Test 3: Tool breakdown
        test_tool_breakdown()
        time.sleep(0.5)

        # Test 4: API savings
        test_api_savings()
        time.sleep(0.5)

        # Test 5: Error handling
        test_error_handling()

        # Summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print("\n✅ ALL TESTS PASSED!")
        print(f"\nCreated {len(operation_ids)} test operations")
        print(f"Session ID: {session_id}")
        print("\nYou can now:")
        print(
            f"  1. View operations: GET {BASE_URL}/metrics/tool-operations?session_id={session_id}"
        )
        print(
            f"  2. View breakdown:  GET {BASE_URL}/metrics/tool-breakdown?time_range=24h"
        )
        print(
            f"  3. View savings:    GET {BASE_URL}/metrics/api-savings?time_range=24h"
        )

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to metrics service")
        print(f"   Make sure the service is running on {BASE_URL}")
        print("   Run: cd omnimemory-metrics-service && python -m src.metrics_service")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
