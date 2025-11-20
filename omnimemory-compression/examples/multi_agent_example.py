"""
Example: Multi-Agent Memory with Layer Support

Demonstrates how to use the new multi-agent and memory layer features
for building agent platforms like Lovable.
"""

import asyncio
import httpx


# Example 1: Different memory layers for different contexts
async def example_memory_layers():
    """Show how to use different memory layers"""
    print("=" * 60)
    print("Example 1: Memory Layer Support")
    print("=" * 60 + "\n")

    async with httpx.AsyncClient() as client:
        # Session layer: Recent conversation (less compression)
        session_response = await client.post(
            "http://localhost:8001/compress",
            json={
                "context": "User: Can you help me implement a login system?\nAssistant: Sure! I'll help you implement a secure login system with JWT tokens...",
                "memory_layer": "session",  # Recent conversation
                "session_id": "user-123-session",
                "tool_id": "claude-code",
            },
        )
        print("SESSION layer (recent conversation):")
        print(
            f"  Compression ratio: {session_response.json()['compression_ratio']:.2%}"
        )
        print(f"  Quality score: {session_response.json()['quality_score']:.2%}\n")

        # Task layer: Active work (balanced)
        task_response = await client.post(
            "http://localhost:8001/compress",
            json={
                "context": "def login(username, password):\n    # Hash password\n    # Verify credentials\n    # Generate JWT token\n    pass",
                "memory_layer": "task",  # Active work context
                "file_path": "auth.py",
                "session_id": "user-123-session",
                "tool_id": "claude-code",
            },
        )
        print("TASK layer (active work):")
        print(f"  Compression ratio: {task_response.json()['compression_ratio']:.2%}")
        print(f"  Quality score: {task_response.json()['quality_score']:.2%}\n")

        # Long-term layer: Historical context (more compression)
        longterm_response = await client.post(
            "http://localhost:8001/compress",
            json={
                "context": "Summary: Implemented user authentication system with JWT. Challenges: CORS issues, token refresh logic. Solution: Added middleware and refresh endpoint.",
                "memory_layer": "long_term",  # Archived context
                "session_id": "user-123-session",
                "tool_id": "claude-code",
            },
        )
        print("LONG_TERM layer (archived):")
        print(
            f"  Compression ratio: {longterm_response.json()['compression_ratio']:.2%}"
        )
        print(f"  Quality score: {longterm_response.json()['quality_score']:.2%}\n")


# Example 2: Compression modes for different use cases
async def example_compression_modes():
    """Show how to use compression modes"""
    print("=" * 60)
    print("Example 2: Compression Mode Selection")
    print("=" * 60 + "\n")

    async with httpx.AsyncClient() as client:
        context = "This is a long document about machine learning algorithms, neural networks, deep learning, and artificial intelligence. It covers topics like supervised learning, unsupervised learning, reinforcement learning, and more."

        modes = ["speed", "balanced", "quality", "maximum"]

        for mode in modes:
            response = await client.post(
                "http://localhost:8001/compress",
                json={
                    "context": context,
                    "compression_mode": mode,
                    "session_id": "demo-session",
                },
            )
            result = response.json()
            print(f"{mode.upper():12} mode:")
            print(f"  Compression: {result['compression_ratio']:.2%}")
            print(f"  Quality:     {result['quality_score']:.2%}")
            print(
                f"  Tokens:      {result['original_tokens']} → {result['compressed_tokens']}\n"
            )


# Example 3: Multi-agent memory sharing
async def example_multi_agent():
    """Show how to use multi-agent memory sharing"""
    print("=" * 60)
    print("Example 3: Multi-Agent Memory Sharing")
    print("=" * 60 + "\n")

    async with httpx.AsyncClient() as client:
        # Researcher agent adds findings to shared pool
        researcher_response = await client.post(
            "http://localhost:8001/compress",
            json={
                "context": "Research findings: VisionDrop compression achieves 94.4% token reduction while maintaining 91% quality. Key insight: importance-based chunk selection.",
                "memory_layer": "task",
                "agent_id": "researcher-agent-1",
                "shared_pool_id": "lovable-project-alpha",
                "sharing_policy": "read_write",
                "metadata": {
                    "team_id": "lovable-team",
                    "tags": ["research", "compression", "visiondrop"],
                },
            },
        )
        print("Researcher Agent stored memory:")
        print(f"  Pool: lovable-project-alpha")
        print(f"  Quality: {researcher_response.json()['quality_score']:.2%}\n")

        # Developer agent adds implementation to shared pool
        developer_response = await client.post(
            "http://localhost:8001/compress",
            json={
                "context": "Implementation complete: Added MMR selection algorithm to prevent redundancy in multi-agent systems. Uses lambda=0.6 for relevance-diversity balance.",
                "memory_layer": "task",
                "agent_id": "developer-agent-1",
                "shared_pool_id": "lovable-project-alpha",
                "sharing_policy": "read_write",
                "metadata": {
                    "team_id": "lovable-team",
                    "tags": ["implementation", "mmr", "algorithm"],
                    "dependencies": [],  # Could reference researcher's entry
                },
            },
        )
        print("Developer Agent stored memory:")
        print(f"  Pool: lovable-project-alpha")
        print(f"  Quality: {developer_response.json()['quality_score']:.2%}\n")

        # Tester agent adds test results (read-only access)
        tester_response = await client.post(
            "http://localhost:8001/compress",
            json={
                "context": "Test results: All 25 test cases passed. MMR selection reduces redundancy by 37% compared to threshold-based selection. Performance impact: +12ms per compression.",
                "memory_layer": "task",
                "agent_id": "tester-agent-1",
                "shared_pool_id": "lovable-project-alpha",
                "sharing_policy": "read_only",  # Can read others' memories
                "metadata": {
                    "team_id": "lovable-team",
                    "tags": ["testing", "results", "mmr"],
                },
            },
        )
        print("Tester Agent stored memory:")
        print(f"  Pool: lovable-project-alpha")
        print(f"  Quality: {tester_response.json()['quality_score']:.2%}\n")


# Example 4: Advanced - Layer + Mode + Multi-Agent combined
async def example_advanced():
    """Show advanced usage combining all features"""
    print("=" * 60)
    print("Example 4: Advanced Combined Usage")
    print("=" * 60 + "\n")

    async with httpx.AsyncClient() as client:
        # Orchestrator agent coordinating work
        response = await client.post(
            "http://localhost:8001/compress",
            json={
                "context": """
                Project Status Update:
                - Researcher completed VisionDrop analysis
                - Developer implemented MMR algorithm
                - Tester validated with 25 test cases
                - Ready for production deployment
                - Next: Documentation and integration guides
                """,
                "memory_layer": "long_term",  # Archive for future reference
                "compression_mode": "quality",  # High quality for important summary
                "agent_id": "orchestrator-agent",
                "shared_pool_id": "lovable-project-alpha",
                "sharing_policy": "read_only",  # Others can read the summary
                "metadata": {
                    "team_id": "lovable-team",
                    "tags": ["summary", "status", "milestone"],
                    "project": "omnimemory-multiagent",
                    "milestone": "v1.0",
                },
            },
        )

        result = response.json()
        print("Orchestrator Agent - Project Summary:")
        print(f"  Layer: long_term (archived)")
        print(f"  Mode: quality (high fidelity)")
        print(f"  Pool: lovable-project-alpha (shared)")
        print(f"  Compression: {result['compression_ratio']:.2%}")
        print(f"  Quality: {result['quality_score']:.2%}")
        print(f"  Original tokens: {result['original_tokens']}")
        print(f"  Compressed tokens: {result['compressed_tokens']}\n")


# Example 5: Practical use case - Code review workflow
async def example_code_review():
    """Practical example: Multi-agent code review workflow"""
    print("=" * 60)
    print("Example 5: Code Review Workflow")
    print("=" * 60 + "\n")

    async with httpx.AsyncClient() as client:
        pool_id = "code-review-pr-123"

        # Developer submits code
        await client.post(
            "http://localhost:8001/compress",
            json={
                "context": "Added multi-agent memory support with SharedMemoryPool, AgentContext, and dependency tracking. Implemented MMR algorithm for diversity.",
                "memory_layer": "task",
                "agent_id": "dev-alice",
                "shared_pool_id": pool_id,
                "sharing_policy": "read_write",
                "file_path": "agent_memory.py",
                "metadata": {"tags": ["pr-123", "feature", "multi-agent"]},
            },
        )
        print("1. Developer (Alice) submitted code")

        # Reviewer 1 adds feedback
        await client.post(
            "http://localhost:8001/compress",
            json={
                "context": "Code looks good. Suggestion: Add docstrings to SharedMemoryPool methods. Consider adding pool size limits to prevent memory leaks.",
                "memory_layer": "task",
                "agent_id": "reviewer-bob",
                "shared_pool_id": pool_id,
                "sharing_policy": "read_write",
                "metadata": {"tags": ["pr-123", "review", "feedback"]},
            },
        )
        print("2. Reviewer (Bob) added feedback")

        # Reviewer 2 adds feedback
        await client.post(
            "http://localhost:8001/compress",
            json={
                "context": "MMR algorithm implementation looks solid. Minor: Consider caching similarity calculations for performance.",
                "memory_layer": "task",
                "agent_id": "reviewer-carol",
                "shared_pool_id": pool_id,
                "sharing_policy": "read_write",
                "metadata": {"tags": ["pr-123", "review", "performance"]},
            },
        )
        print("3. Reviewer (Carol) added performance notes")

        # Developer responds and updates
        await client.post(
            "http://localhost:8001/compress",
            json={
                "context": "Updated based on feedback: Added comprehensive docstrings, implemented pool size limits (max 10k entries), added LRU cache for MMR similarities.",
                "memory_layer": "task",
                "agent_id": "dev-alice",
                "shared_pool_id": pool_id,
                "sharing_policy": "read_write",
                "metadata": {"tags": ["pr-123", "updated", "addressed-feedback"]},
            },
        )
        print("4. Developer (Alice) addressed feedback")

        # Archive final state
        await client.post(
            "http://localhost:8001/compress",
            json={
                "context": "PR-123 merged: Multi-agent memory support with SharedMemoryPool, MMR algorithm, pool size limits, and comprehensive documentation.",
                "memory_layer": "long_term",  # Archive
                "compression_mode": "quality",  # High quality for records
                "agent_id": "bot-merger",
                "shared_pool_id": pool_id,
                "sharing_policy": "read_only",
                "metadata": {"tags": ["pr-123", "merged", "milestone"]},
            },
        )
        print("5. Bot archived PR to long-term memory\n")

        print("Code review workflow complete!")
        print(
            "All agent interactions stored in shared pool with dependency tracking.\n"
        )


async def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("Multi-Agent Memory & Layer Support Examples")
    print("=" * 60 + "\n")

    print("NOTE: Make sure compression server is running on port 8001")
    print(
        "      Start with: cd omnimemory-compression && python -m src.compression_server\n"
    )

    try:
        await example_memory_layers()
        await example_compression_modes()
        await example_multi_agent()
        await example_advanced()
        await example_code_review()

        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure the compression server is running!")


if __name__ == "__main__":
    asyncio.run(main())
