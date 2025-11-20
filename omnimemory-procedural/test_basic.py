#!/usr/bin/env python3
"""
Basic test to verify procedural memory implementation
"""

import sys

sys.path.insert(0, "src")

from procedural_memory import ProceduralMemoryEngine, WorkflowPattern, Prediction
import networkx as nx


def test_initialization():
    """Test that engine initializes correctly"""
    engine = ProceduralMemoryEngine()
    assert isinstance(engine.workflow_graph, nx.DiGraph)
    assert isinstance(engine.patterns, dict)
    assert isinstance(engine.causal_chains, dict)
    print("✓ Initialization test passed")


def test_pattern_creation():
    """Test WorkflowPattern creation"""
    import numpy as np

    pattern = WorkflowPattern(
        pattern_id="test123",
        command_sequence=["cmd1", "cmd2", "cmd3"],
        embeddings=[np.random.randn(768) for _ in range(3)],
        transitions=[np.random.randn(768) for _ in range(2)],
        success_count=5,
        failure_count=1,
    )
    assert pattern.confidence == 5 / 6
    print("✓ WorkflowPattern test passed")


def test_prediction_creation():
    """Test Prediction creation"""
    pred = Prediction(
        next_command="test_cmd",
        confidence=0.85,
        reason="Test reason",
        similar_patterns=["pat1", "pat2"],
        auto_suggestions=["sug1", "sug2"],
    )
    assert pred.next_command == "test_cmd"
    assert pred.confidence == 0.85
    print("✓ Prediction test passed")


def test_command_normalization():
    """Test command normalization"""
    engine = ProceduralMemoryEngine()

    # Test file normalization
    result = engine._normalize_command("cat /path/to/file.txt")
    assert "<FILE>" in result

    # Test URL normalization
    result = engine._normalize_command("curl https://example.com/api")
    assert "<URL>" in result

    # Test number normalization
    result = engine._normalize_command("sleep 100")
    assert "<NUM>" in result

    print("✓ Command normalization test passed")


def test_graph_operations():
    """Test workflow graph operations"""
    engine = ProceduralMemoryEngine()

    commands = ["cmd1", "cmd2", "cmd3"]
    engine._update_workflow_graph(commands, "success")

    assert engine.workflow_graph.number_of_nodes() > 0
    assert engine.workflow_graph.number_of_edges() > 0

    print("✓ Graph operations test passed")


def test_causal_chains():
    """Test causal chain tracking"""
    engine = ProceduralMemoryEngine()

    commands = ["cmd1", "cmd2", "cmd3"]
    engine._update_causal_chains(commands, "success")

    assert len(engine.causal_chains) > 0

    print("✓ Causal chain test passed")


def test_cosine_similarity():
    """Test cosine similarity calculation"""
    import numpy as np

    engine = ProceduralMemoryEngine()

    a = np.array([1, 0, 0])
    b = np.array([1, 0, 0])

    similarity = engine._cosine_similarity(a, b)
    assert abs(similarity - 1.0) < 0.001

    print("✓ Cosine similarity test passed")


def test_fuzzy_match():
    """Test fuzzy string matching"""
    engine = ProceduralMemoryEngine()

    assert engine._fuzzy_match("git add file.txt", "git add file.txt", threshold=0.8)
    assert engine._fuzzy_match("git status", "git commit", threshold=0.8) == False

    print("✓ Fuzzy match test passed")


def test_pattern_id_generation():
    """Test pattern ID generation"""
    engine = ProceduralMemoryEngine()

    commands = ["cmd1", "cmd2", "cmd3"]
    pattern_id = engine._generate_pattern_id(commands)

    assert len(pattern_id) == 16
    assert pattern_id.isalnum()

    # Same commands should generate same ID
    pattern_id2 = engine._generate_pattern_id(commands)
    assert pattern_id == pattern_id2

    print("✓ Pattern ID generation test passed")


if __name__ == "__main__":
    print("Running basic tests for Procedural Memory Engine...\n")

    try:
        test_initialization()
        test_pattern_creation()
        test_prediction_creation()
        test_command_normalization()
        test_graph_operations()
        test_causal_chains()
        test_cosine_similarity()
        test_fuzzy_match()
        test_pattern_id_generation()

        print("\n✓ All basic tests passed!")
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
