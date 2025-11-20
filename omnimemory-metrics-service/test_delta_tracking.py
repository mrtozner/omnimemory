"""
Test Delta-Based Metrics Tracking

Verifies that:
1. Delta columns are correctly calculated
2. Historical queries SUM deltas to show accurate totals
3. Latest endpoint shows current cumulative values
4. First record (no previous data) uses full amount as delta
"""

import sys
import os
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_store import MetricsStore
from datetime import datetime, timedelta


def test_delta_calculation():
    """Test that deltas are correctly calculated from cumulative values"""
    print("\n=== Test 1: Delta Calculation ===")

    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        store = MetricsStore(db_path=db_path, enable_vector_store=False)

        # Start a session
        session_id = store.start_session(tool_id="test-tool", tool_version="1.0.0")
        print(f"Started session: {session_id}")

        # Store first metrics (cumulative: 100 tokens, 5 compressions, 10 embeddings)
        metrics1 = {
            "embeddings": {"mlx_metrics": {"total_embeddings": 10}},
            "compression": {
                "metrics": {"total_compressions": 5, "total_tokens_saved": 100}
            },
            "procedural": {},
        }
        store.store_metrics(metrics1, tool_id="test-tool", session_id=session_id)
        print(f"Stored metrics 1: embeddings=10, compressions=5, tokens=100")

        # Store second metrics (cumulative: 250 tokens, 12 compressions, 25 embeddings)
        # Expected deltas: 150 tokens, 7 compressions, 15 embeddings
        metrics2 = {
            "embeddings": {"mlx_metrics": {"total_embeddings": 25}},
            "compression": {
                "metrics": {"total_compressions": 12, "total_tokens_saved": 250}
            },
            "procedural": {},
        }
        store.store_metrics(metrics2, tool_id="test-tool", session_id=session_id)
        print(f"Stored metrics 2: embeddings=25, compressions=12, tokens=250")

        # Store third metrics (cumulative: 500 tokens, 20 compressions, 40 embeddings)
        # Expected deltas: 250 tokens, 8 compressions, 15 embeddings
        metrics3 = {
            "embeddings": {"mlx_metrics": {"total_embeddings": 40}},
            "compression": {
                "metrics": {"total_compressions": 20, "total_tokens_saved": 500}
            },
            "procedural": {},
        }
        store.store_metrics(metrics3, tool_id="test-tool", session_id=session_id)
        print(f"Stored metrics 3: embeddings=40, compressions=20, tokens=500")

        # Verify deltas in database
        cursor = store.conn.cursor()
        cursor.execute(
            """
            SELECT
                total_embeddings, total_embeddings_delta,
                total_compressions, total_compressions_delta,
                tokens_saved, tokens_saved_delta,
                timestamp
            FROM metrics
            WHERE session_id = ?
            ORDER BY timestamp ASC
            """,
            (session_id,),
        )

        rows = cursor.fetchall()
        print(f"\nFound {len(rows)} records:")

        expected_deltas = [
            (10, 10, 5, 5, 100, 100),  # First record: delta = full amount
            (25, 15, 12, 7, 250, 150),  # Second: delta = current - previous
            (40, 15, 20, 8, 500, 250),  # Third: delta = current - previous
        ]

        for i, row in enumerate(rows):
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
                f"  Embeddings: cumulative={emb_cum}, delta={emb_delta} (expected {exp_emb_delta})"
            )
            print(
                f"  Compressions: cumulative={comp_cum}, delta={comp_delta} (expected {exp_comp_delta})"
            )
            print(
                f"  Tokens: cumulative={tok_cum}, delta={tok_delta} (expected {exp_tok_delta})"
            )

            assert (
                emb_cum == exp_emb_cum
            ), f"Embeddings cumulative mismatch: {emb_cum} != {exp_emb_cum}"
            assert (
                emb_delta == exp_emb_delta
            ), f"Embeddings delta mismatch: {emb_delta} != {exp_emb_delta}"
            assert (
                comp_cum == exp_comp_cum
            ), f"Compressions cumulative mismatch: {comp_cum} != {exp_comp_cum}"
            assert (
                comp_delta == exp_comp_delta
            ), f"Compressions delta mismatch: {comp_delta} != {exp_comp_delta}"
            assert (
                tok_cum == exp_tok_cum
            ), f"Tokens cumulative mismatch: {tok_cum} != {exp_tok_cum}"
            assert (
                tok_delta == exp_tok_delta
            ), f"Tokens delta mismatch: {tok_delta} != {exp_tok_delta}"

        print("\n✓ All deltas calculated correctly!")

        store.close()

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_aggregate_queries():
    """Test that aggregate queries use deltas and return accurate totals"""
    print("\n=== Test 2: Aggregate Queries ===")

    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        store = MetricsStore(db_path=db_path, enable_vector_store=False)

        # Start a session
        session_id = store.start_session(tool_id="test-tool", tool_version="1.0.0")

        # Simulate service sending cumulative totals 3 times
        # (mimicking real-world scenario where services report growing totals)
        for i in range(1, 4):
            metrics = {
                "embeddings": {"mlx_metrics": {"total_embeddings": i * 100}},
                "compression": {
                    "metrics": {
                        "total_compressions": i * 10,
                        "total_tokens_saved": i * 1000,
                    }
                },
                "procedural": {},
            }
            store.store_metrics(metrics, tool_id="test-tool", session_id=session_id)
            print(
                f"Stored metrics {i}: embeddings={i*100}, compressions={i*10}, tokens={i*1000}"
            )

        # Test get_aggregates() - should use deltas and return accurate total
        aggregates = store.get_aggregates(hours=24, tool_id="test-tool")

        print(f"\nAggregate results (using deltas):")
        print(f"  Total embeddings: {aggregates['total_embeddings']} (expected 300)")
        print(f"  Total compressions: {aggregates['total_compressions']} (expected 30)")
        print(
            f"  Total tokens saved: {aggregates['total_tokens_saved']} (expected 3000)"
        )

        # With deltas, SUM should equal the latest cumulative value
        # NOT the sum of all cumulative values (which would be 100+200+300 = 600)
        assert (
            aggregates["total_embeddings"] == 300
        ), f"Wrong total: {aggregates['total_embeddings']}"
        assert (
            aggregates["total_compressions"] == 30
        ), f"Wrong total: {aggregates['total_compressions']}"
        assert (
            aggregates["total_tokens_saved"] == 3000
        ), f"Wrong total: {aggregates['total_tokens_saved']}"

        print("\n✓ Aggregates use deltas correctly!")

        # Test get_tool_metrics() - should also use deltas
        tool_metrics = store.get_tool_metrics("test-tool", hours=24)

        print(f"\nTool metrics (using deltas):")
        print(f"  Total embeddings: {tool_metrics['total_embeddings']} (expected 300)")
        print(
            f"  Total tokens saved: {tool_metrics['total_tokens_saved']} (expected 3000)"
        )

        assert (
            tool_metrics["total_embeddings"] == 300
        ), f"Wrong tool total: {tool_metrics['total_embeddings']}"
        assert (
            tool_metrics["total_tokens_saved"] == 3000
        ), f"Wrong tool total: {tool_metrics['total_tokens_saved']}"

        print("\n✓ Tool metrics use deltas correctly!")

        store.close()

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_latest_vs_historical():
    """Test that latest uses cumulative, historical uses deltas"""
    print("\n=== Test 3: Latest vs Historical ===")

    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        store = MetricsStore(db_path=db_path, enable_vector_store=False)

        # Start a session
        session_id = store.start_session(tool_id="test-tool", tool_version="1.0.0")

        # Store 5 snapshots with increasing cumulative values
        for i in range(1, 6):
            metrics = {
                "embeddings": {"mlx_metrics": {"total_embeddings": i * 50}},
                "compression": {
                    "metrics": {
                        "total_compressions": i * 5,
                        "total_tokens_saved": i * 500,
                    }
                },
                "procedural": {},
            }
            store.store_metrics(metrics, tool_id="test-tool", session_id=session_id)

        # Get latest metrics (should show current cumulative: 250 embeddings, 2500 tokens)
        latest = store.get_latest(tool_id="test-tool")

        print(f"\nLatest metrics (cumulative values):")
        print(f"  Embeddings: {latest['total_embeddings']} (expected 250)")
        print(f"  Tokens saved: {latest['tokens_saved']} (expected 2500)")

        assert (
            latest["total_embeddings"] == 250
        ), f"Latest cumulative wrong: {latest['total_embeddings']}"
        assert (
            latest["tokens_saved"] == 2500
        ), f"Latest cumulative wrong: {latest['tokens_saved']}"

        print("\n✓ Latest shows current cumulative values!")

        # Get aggregates (should use deltas and also equal 250 embeddings, 2500 tokens)
        aggregates = store.get_aggregates(hours=24, tool_id="test-tool")

        print(f"\nHistorical aggregates (sum of deltas):")
        print(f"  Embeddings: {aggregates['total_embeddings']} (expected 250)")
        print(f"  Tokens saved: {aggregates['total_tokens_saved']} (expected 2500)")

        assert (
            aggregates["total_embeddings"] == 250
        ), f"Aggregate delta sum wrong: {aggregates['total_embeddings']}"
        assert (
            aggregates["total_tokens_saved"] == 2500
        ), f"Aggregate delta sum wrong: {aggregates['total_tokens_saved']}"

        print(
            "\n✓ Historical aggregates match latest (SUM of deltas = final cumulative)!"
        )

        store.close()

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_backward_compatibility():
    """Test that existing records without deltas are handled gracefully"""
    print("\n=== Test 4: Backward Compatibility ===")

    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        store = MetricsStore(db_path=db_path, enable_vector_store=False)

        # Manually insert a record without delta columns (simulating old data)
        cursor = store.conn.cursor()
        session_id = "old-session"
        cursor.execute(
            """
            INSERT INTO metrics (
                timestamp, service, tool_id, session_id,
                total_embeddings, total_compressions, tokens_saved
            ) VALUES (?, 'combined', 'old-tool', ?, 100, 10, 1000)
            """,
            (datetime.now().isoformat(), session_id),
        )
        store.conn.commit()

        print("Inserted old record without deltas")

        # Store new metrics with deltas
        metrics = {
            "embeddings": {"mlx_metrics": {"total_embeddings": 200}},
            "compression": {
                "metrics": {"total_compressions": 20, "total_tokens_saved": 2000}
            },
            "procedural": {},
        }
        store.store_metrics(metrics, tool_id="old-tool", session_id=session_id)

        print("Stored new record with deltas")

        # Query aggregates - should handle NULL deltas as 0
        aggregates = store.get_aggregates(hours=24, tool_id="old-tool")

        print(f"\nAggregates with mixed old/new records:")
        print(f"  Total embeddings: {aggregates['total_embeddings']}")
        print(f"  Total tokens saved: {aggregates['total_tokens_saved']}")

        # Old record has NULL deltas (treated as 0), new record has deltas
        # Expected: 0 (old) + 100 (new delta) = 100 for embeddings
        # (since new cumulative 200 - old cumulative 100 = delta 100)
        assert (
            aggregates["total_embeddings"] == 100
        ), f"Backward compat failed: {aggregates['total_embeddings']}"

        print("\n✓ Backward compatibility works (NULL deltas treated as 0)!")

        store.close()

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


def main():
    """Run all tests"""
    print("=" * 60)
    print("Delta-Based Metrics Tracking Tests")
    print("=" * 60)

    try:
        test_delta_calculation()
        test_aggregate_queries()
        test_latest_vs_historical()
        test_backward_compatibility()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
