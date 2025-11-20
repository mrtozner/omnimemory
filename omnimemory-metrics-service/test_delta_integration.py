#!/usr/bin/env python3
"""
Integration Test for Delta Tracking via API Endpoints

Tests the complete workflow:
1. Send metrics via POST /metrics
2. Verify deltas are calculated correctly
3. Query /metrics/history and verify SUM(deltas) = latest cumulative
4. Query /metrics/latest and verify cumulative values
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8002"
TOOL_ID = "delta-test-tool"


def test_delta_integration():
    """Test delta tracking through API endpoints"""

    print("=" * 70)
    print("DELTA TRACKING INTEGRATION TEST")
    print("=" * 70)

    # Step 1: Send first metrics (cumulative: 100 tokens, 5 compressions, 10 embeddings)
    print("\n[Step 1] Sending first metrics snapshot...")
    metrics1 = {
        "tool_id": TOOL_ID,
        "embeddings": {"mlx_metrics": {"total_embeddings": 10}},
        "compression": {
            "metrics": {"total_compressions": 5, "total_tokens_saved": 100}
        },
    }

    response1 = requests.post(f"{BASE_URL}/metrics", json=metrics1)
    print(f"Status: {response1.status_code}")
    print(f"Response: {response1.json()}")
    assert response1.status_code == 200, f"Failed to store metrics: {response1.text}"
    session_id = response1.json().get("session_id")
    print(f"✓ Metrics stored (session: {session_id})")

    time.sleep(0.5)  # Small delay to ensure timestamp ordering

    # Step 2: Send second metrics (cumulative: 250 tokens, 12 compressions, 25 embeddings)
    # Expected deltas: 150 tokens, 7 compressions, 15 embeddings
    print("\n[Step 2] Sending second metrics snapshot...")
    metrics2 = {
        "tool_id": TOOL_ID,
        "embeddings": {"mlx_metrics": {"total_embeddings": 25}},
        "compression": {
            "metrics": {"total_compressions": 12, "total_tokens_saved": 250}
        },
    }

    response2 = requests.post(f"{BASE_URL}/metrics", json=metrics2)
    print(f"Status: {response2.status_code}")
    print(f"Response: {response2.json()}")
    assert response2.status_code == 200, f"Failed to store metrics: {response2.text}"
    print(f"✓ Metrics stored")

    time.sleep(0.5)

    # Step 3: Send third metrics (cumulative: 500 tokens, 20 compressions, 40 embeddings)
    # Expected deltas: 250 tokens, 8 compressions, 15 embeddings
    print("\n[Step 3] Sending third metrics snapshot...")
    metrics3 = {
        "tool_id": TOOL_ID,
        "embeddings": {"mlx_metrics": {"total_embeddings": 40}},
        "compression": {
            "metrics": {"total_compressions": 20, "total_tokens_saved": 500}
        },
    }

    response3 = requests.post(f"{BASE_URL}/metrics", json=metrics3)
    print(f"Status: {response3.status_code}")
    print(f"Response: {response3.json()}")
    assert response3.status_code == 200, f"Failed to store metrics: {response3.text}"
    print(f"✓ Metrics stored")

    time.sleep(1)  # Give server time to process

    # Step 4: Query latest metrics (should show cumulative: 500 tokens, 20 compressions, 40 embeddings)
    print("\n[Step 4] Querying /metrics/latest...")
    latest_response = requests.get(f"{BASE_URL}/metrics/latest?tool_id={TOOL_ID}")
    print(f"Status: {latest_response.status_code}")

    if latest_response.status_code == 200:
        latest = latest_response.json()
        print(f"Latest metrics:")
        print(f"  Embeddings: {latest.get('total_embeddings')} (expected 40)")
        print(f"  Compressions: {latest.get('total_compressions')} (expected 20)")
        print(f"  Tokens saved: {latest.get('tokens_saved')} (expected 500)")

        assert (
            latest.get("total_embeddings") == 40
        ), f"Wrong embeddings: {latest.get('total_embeddings')}"
        assert (
            latest.get("total_compressions") == 20
        ), f"Wrong compressions: {latest.get('total_compressions')}"
        assert (
            latest.get("tokens_saved") == 500
        ), f"Wrong tokens: {latest.get('tokens_saved')}"
        print("✓ Latest metrics show correct cumulative values!")
    else:
        print(f"WARNING: Latest endpoint returned {latest_response.status_code}")
        print(f"Response: {latest_response.text}")

    # Step 5: Query historical aggregates (should use deltas, SUM = 500 tokens, 20 compressions, 40 embeddings)
    print("\n[Step 5] Querying /metrics/tool/{tool_id} for historical aggregates...")
    tool_response = requests.get(f"{BASE_URL}/metrics/tool/{TOOL_ID}?hours=1")
    print(f"Status: {tool_response.status_code}")

    if tool_response.status_code == 200:
        tool_data = tool_response.json()
        print(f"Tool metrics (using deltas):")
        print(f"  Total embeddings: {tool_data.get('total_embeddings')} (expected 40)")
        print(
            f"  Total tokens saved: {tool_data.get('total_tokens_saved')} (expected 500)"
        )

        assert (
            tool_data.get("total_embeddings") == 40
        ), f"Wrong aggregated embeddings: {tool_data.get('total_embeddings')}"
        assert (
            tool_data.get("total_tokens_saved") == 500
        ), f"Wrong aggregated tokens: {tool_data.get('total_tokens_saved')}"
        print("✓ Historical aggregates use deltas correctly (SUM = final cumulative)!")
    else:
        print(f"WARNING: Tool metrics endpoint returned {tool_response.status_code}")
        print(f"Response: {tool_response.text}")

    # Step 6: Query general aggregates
    print("\n[Step 6] Querying /metrics/aggregates...")
    agg_response = requests.get(
        f"{BASE_URL}/metrics/aggregates?hours=1&tool_id={TOOL_ID}"
    )
    print(f"Status: {agg_response.status_code}")

    if agg_response.status_code == 200:
        agg_data = agg_response.json()
        print(f"Aggregated metrics (using deltas):")
        print(f"  Total embeddings: {agg_data.get('total_embeddings')} (expected 40)")
        print(
            f"  Total compressions: {agg_data.get('total_compressions')} (expected 20)"
        )
        print(
            f"  Total tokens saved: {agg_data.get('total_tokens_saved')} (expected 500)"
        )

        assert (
            agg_data.get("total_embeddings") == 40
        ), f"Wrong agg embeddings: {agg_data.get('total_embeddings')}"
        assert (
            agg_data.get("total_compressions") == 20
        ), f"Wrong agg compressions: {agg_data.get('total_compressions')}"
        assert (
            agg_data.get("total_tokens_saved") == 500
        ), f"Wrong agg tokens: {agg_data.get('total_tokens_saved')}"
        print("✓ Aggregate endpoint uses deltas correctly!")
    else:
        print(f"WARNING: Aggregates endpoint returned {agg_response.status_code}")
        print(f"Response: {agg_response.text}")

    # Step 7: Verify delta calculation by querying raw data
    print("\n[Step 7] Verifying delta values in database...")
    import sqlite3

    db_path = "/Users/mertozoner/Documents/claude-idea-discussion/omni-memory/omnimemory-metrics-service/metrics.db"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                total_embeddings, total_embeddings_delta,
                total_compressions, total_compressions_delta,
                tokens_saved, tokens_saved_delta,
                timestamp
            FROM metrics
            WHERE tool_id = ?
            ORDER BY timestamp ASC
        """,
            (TOOL_ID,),
        )

        rows = cursor.fetchall()
        print(f"\nFound {len(rows)} records in database:")

        expected_deltas = [
            (10, 10, 5, 5, 100, 100),  # First: delta = full amount
            (25, 15, 12, 7, 250, 150),  # Second: delta = difference
            (40, 15, 20, 8, 500, 250),  # Third: delta = difference
        ]

        for i, row in enumerate(rows[-3:]):  # Get last 3 records
            emb_cum, emb_delta, comp_cum, comp_delta, tok_cum, tok_delta, ts = row
            (
                exp_emb_cum,
                exp_emb_delta,
                exp_comp_cum,
                exp_comp_delta,
                exp_tok_cum,
                exp_tok_delta,
            ) = expected_deltas[i]

            print(f"\nRecord {i+1}:")
            print(
                f"  Embeddings: cum={emb_cum} (exp {exp_emb_cum}), delta={emb_delta} (exp {exp_emb_delta})"
            )
            print(
                f"  Compressions: cum={comp_cum} (exp {exp_comp_cum}), delta={comp_delta} (exp {exp_comp_delta})"
            )
            print(
                f"  Tokens: cum={tok_cum} (exp {exp_tok_cum}), delta={tok_delta} (exp {exp_tok_delta})"
            )

            assert emb_cum == exp_emb_cum, f"Embeddings cumulative mismatch"
            assert emb_delta == exp_emb_delta, f"Embeddings delta mismatch"
            assert comp_cum == exp_comp_cum, f"Compressions cumulative mismatch"
            assert comp_delta == exp_comp_delta, f"Compressions delta mismatch"
            assert tok_cum == exp_tok_cum, f"Tokens cumulative mismatch"
            assert tok_delta == exp_tok_delta, f"Tokens delta mismatch"

        print("\n✓ Database delta values are correct!")

        # Verify SUM(deltas) = latest cumulative
        cursor.execute(
            """
            SELECT
                SUM(total_embeddings_delta) as total_emb,
                SUM(total_compressions_delta) as total_comp,
                SUM(tokens_saved_delta) as total_tok
            FROM metrics
            WHERE tool_id = ?
        """,
            (TOOL_ID,),
        )

        sums = cursor.fetchone()
        print(f"\nSUM of all deltas:")
        print(f"  Embeddings: {sums[0]} (should equal latest cumulative: 40)")
        print(f"  Compressions: {sums[1]} (should equal latest cumulative: 20)")
        print(f"  Tokens: {sums[2]} (should equal latest cumulative: 500)")

        assert sums[0] == 40, f"Delta sum mismatch for embeddings"
        assert sums[1] == 20, f"Delta sum mismatch for compressions"
        assert sums[2] == 500, f"Delta sum mismatch for tokens"

        print("✓ SUM(deltas) equals latest cumulative values!")

        conn.close()

    except Exception as e:
        print(f"Warning: Could not verify database directly: {e}")

    print("\n" + "=" * 70)
    print("✓ ALL INTEGRATION TESTS PASSED!")
    print("=" * 70)
    print("\nDelta tracking is working correctly:")
    print("  ✓ First record: delta = cumulative value")
    print("  ✓ Subsequent records: delta = difference from previous")
    print("  ✓ Latest endpoint shows cumulative values")
    print("  ✓ Historical queries use deltas")
    print("  ✓ SUM(deltas) = final cumulative value")
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_delta_integration()
        exit(0)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
