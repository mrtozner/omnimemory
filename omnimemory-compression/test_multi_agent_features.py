"""
Test script for multi-agent memory and layer features
"""

import sys

sys.path.insert(0, "src")

from memory_layers import MemoryLayer, CompressionMode, get_layer_config
from agent_memory import (
    AgentContext,
    SharingPolicy,
    SharedMemoryPool,
    MultiAgentMemoryManager,
)


def test_memory_layers():
    """Test memory layer configurations"""
    print("Testing Memory Layers...")

    # Test getting layer configs
    session_config = get_layer_config(MemoryLayer.SESSION)
    print(
        f"  SESSION config: threshold={session_config.quality_threshold}, "
        f"compression={session_config.target_compression}"
    )

    task_config = get_layer_config(MemoryLayer.TASK)
    print(
        f"  TASK config: threshold={task_config.quality_threshold}, "
        f"compression={task_config.target_compression}"
    )

    # Test compression modes
    speed_config = get_layer_config(compression_mode=CompressionMode.SPEED)
    print(f"  SPEED mode: threshold={speed_config.quality_threshold}")

    # Test combined
    combined = get_layer_config(MemoryLayer.TASK, CompressionMode.QUALITY)
    print(
        f"  TASK+QUALITY: threshold={combined.quality_threshold}, "
        f"compression={combined.target_compression}"
    )

    print("✓ Memory layers working correctly\n")


def test_multi_agent_memory():
    """Test multi-agent memory management"""
    print("Testing Multi-Agent Memory...")

    # Create memory manager
    manager = MultiAgentMemoryManager()

    # Create shared pool
    pool = manager.create_pool("test-pool", team_id="team-1")
    print(f"  Created pool: {pool.pool_id}")

    # Create agent contexts
    agent1 = AgentContext(
        agent_id="agent-1",
        shared_pool_id="test-pool",
        team_id="team-1",
        sharing_policy=SharingPolicy.READ_WRITE,
        tags=["researcher", "python"],
    )

    agent2 = AgentContext(
        agent_id="agent-2",
        shared_pool_id="test-pool",
        team_id="team-1",
        sharing_policy=SharingPolicy.READ_ONLY,
        tags=["coder", "javascript"],
    )

    # Add memories
    entry1 = pool.add_memory(
        agent_context=agent1,
        content="This is important research data about VisionDrop compression.",
        memory_layer="task",
        compressed_content="Important VisionDrop research.",
    )
    print(f"  Agent 1 created memory: {entry1}")

    entry2 = pool.add_memory(
        agent_context=agent2,
        content="Implemented the MMR selection algorithm.",
        memory_layer="task",
        compressed_content="Implemented MMR algorithm.",
        dependencies=[entry1],  # Depends on previous research
    )
    print(f"  Agent 2 created memory: {entry2} (depends on {entry1})")

    # Retrieve memories
    agent1_memories = pool.get_agent_memories("agent-1", include_shared=True)
    print(f"  Agent 1 can see {len(agent1_memories)} memories")

    agent2_memories = pool.get_agent_memories("agent-2", include_shared=True)
    print(f"  Agent 2 can see {len(agent2_memories)} memories")

    # Check dependencies
    deps = pool.get_dependencies(entry1)
    print(f"  Entry {entry1} has {len(deps)} dependent entries")

    # Get pool stats
    stats = manager.get_pool_stats("test-pool")
    print(f"  Pool stats: {stats}")

    print("✓ Multi-agent memory working correctly\n")


def test_enum_values():
    """Test enum string values"""
    print("Testing Enum Values...")

    # Test MemoryLayer enum values
    assert MemoryLayer.SESSION.value == "session"
    assert MemoryLayer.TASK.value == "task"
    assert MemoryLayer.LONG_TERM.value == "long_term"
    assert MemoryLayer.GLOBAL.value == "global"
    print("  ✓ MemoryLayer enum values correct")

    # Test CompressionMode enum values
    assert CompressionMode.SPEED.value == "speed"
    assert CompressionMode.BALANCED.value == "balanced"
    assert CompressionMode.QUALITY.value == "quality"
    assert CompressionMode.MAXIMUM.value == "maximum"
    print("  ✓ CompressionMode enum values correct")

    # Test SharingPolicy enum values
    assert SharingPolicy.PRIVATE.value == "private"
    assert SharingPolicy.READ_ONLY.value == "read_only"
    assert SharingPolicy.READ_WRITE.value == "read_write"
    assert SharingPolicy.APPEND_ONLY.value == "append_only"
    print("  ✓ SharingPolicy enum values correct")

    print("✓ All enum values working correctly\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Agent Memory & Layer Features Test")
    print("=" * 60 + "\n")

    try:
        test_enum_values()
        test_memory_layers()
        test_multi_agent_memory()

        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
