#!/usr/bin/env python3
"""
LangChain + OmniMemory Integration Examples

This file demonstrates how to use OmniMemory with LangChain agents
for automatic compression and multi-agent memory sharing.
"""


# Example 1: Basic Usage with Compression
def example_basic_usage():
    """Basic LangChain agent with OmniMemory compression."""
    from langchain.agents import initialize_agent, AgentType
    from langchain.tools import Tool
    from langchain.llms import OpenAI
    from langchain_adapter import create_omnimemory

    # Create memory with automatic compression
    memory = create_omnimemory(
        agent_id="research_agent",
        compression_mode="BALANCED",  # 60-70% token savings
        enable_sharing=False,
    )

    # Create a simple tool
    def search_tool(query):
        return f"Search results for: {query}"

    tools = [
        Tool(name="Search", func=search_tool, description="Search for information")
    ]

    # Initialize agent with compressed memory
    llm = OpenAI(temperature=0)
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
    )

    # Use the agent
    response = agent.run("What is quantum computing?")

    # Check compression stats
    stats = memory.get_stats()
    print(f"\nCompression Stats:")
    print(f"  Tokens Saved: {stats['tokens_saved']}")
    print(f"  Compression Ratio: {stats['compression_ratio']:.2f}x")
    print(f"  Messages Processed: {stats['messages_processed']}")


# Example 2: Multi-Agent Memory Sharing
def example_multi_agent_collaboration():
    """Multiple agents sharing compressed memory."""
    from langchain.agents import initialize_agent, AgentType
    from langchain.tools import Tool
    from langchain.llms import OpenAI
    from langchain_adapter import create_shared_memory_pool

    # Create shared memory pool
    shared_pool_id = "project_alpha"

    # Agent 1: Research Agent
    research_memory = create_shared_memory_pool(shared_pool_id)
    research_memory.config.agent_id = "research_001"

    def research_tool(query):
        return f"Research findings: {query}"

    research_agent = initialize_agent(
        [Tool(name="Research", func=research_tool, description="Research information")],
        OpenAI(temperature=0),
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=research_memory,
        verbose=True,
    )

    # Agent 2: Analysis Agent
    analysis_memory = create_shared_memory_pool(shared_pool_id)
    analysis_memory.config.agent_id = "analysis_001"

    def analyze_tool(data):
        return f"Analysis results: {data}"

    analysis_agent = initialize_agent(
        [Tool(name="Analyze", func=analyze_tool, description="Analyze data")],
        OpenAI(temperature=0),
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=analysis_memory,
        verbose=True,
    )

    # Agent 1 does research
    research_result = research_agent.run("Research quantum computing applications")

    # Agent 2 can access Agent 1's compressed memories
    analysis_result = analysis_agent.run(
        "Analyze the research findings on quantum computing"
    )

    print(f"\nShared Memory Pool Statistics:")
    print(
        f"  Research Agent Tokens Saved: {research_memory.get_stats()['tokens_saved']}"
    )
    print(
        f"  Analysis Agent Tokens Saved: {analysis_memory.get_stats()['tokens_saved']}"
    )


# Example 3: Memory Layers with Auto-Promotion
def example_memory_layers():
    """Demonstrate memory layer auto-promotion."""
    from langchain.agents import initialize_agent, AgentType
    from langchain.llms import OpenAI
    from langchain_adapter import OmniMemory, OmniMemoryConfig
    from memory_layers import MemoryLayer

    # Configure memory with auto-promotion
    config = OmniMemoryConfig(
        agent_id="adaptive_agent",
        default_layer=MemoryLayer.SESSION,
        auto_promote=True,
        promotion_threshold=10,  # Promote after 10 messages
        compression_mode="BALANCED",
    )

    memory = OmniMemory(config=config)

    # Create agent
    agent = initialize_agent(
        [],
        OpenAI(temperature=0),
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
    )

    # Simulate conversation over time
    conversations = [
        "Tell me about machine learning",
        "What are neural networks?",
        "Explain backpropagation",
        "What is gradient descent?",
        "How do CNNs work?",
        "What are RNNs?",
        "Explain transformers",
        "What is BERT?",
        "How does GPT work?",
        "What is attention mechanism?",
        "Explain self-attention",  # This will trigger promotion to LONG_TERM
        "What are the latest ML trends?",
    ]

    for i, question in enumerate(conversations):
        print(f"\n--- Conversation {i+1} ---")
        response = agent.run(question)

        # Check current memory layer
        current_layer = memory._get_memory_layer()
        print(f"Current Memory Layer: {current_layer.value}")

        # Check memory distribution
        stats = memory.get_stats()
        if stats["messages_processed"] > 0:
            print(f"Memory Distribution: {stats['memory_distribution']}")


# Example 4: Compression Modes for Different Use Cases
def example_compression_modes():
    """Demonstrate different compression modes."""
    from langchain.llms import OpenAI
    from langchain_adapter import create_omnimemory

    scenarios = [
        {
            "name": "Speed Mode - Fast responses, moderate quality",
            "mode": "SPEED",
            "content": "Process this quickly with acceptable quality loss",
        },
        {
            "name": "Balanced Mode - Best trade-off",
            "mode": "BALANCED",
            "content": "Standard processing with good quality preservation",
        },
        {
            "name": "Quality Mode - High accuracy needed",
            "mode": "QUALITY",
            "content": "Critical information that must be preserved accurately",
        },
        {
            "name": "Maximum Mode - Debugging/Learning",
            "mode": "MAXIMUM",
            "content": "Complex technical details requiring full preservation",
        },
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        print("-" * 50)

        memory = create_omnimemory(
            agent_id=f"test_{scenario['mode'].lower()}",
            compression_mode=scenario["mode"],
        )

        # Simulate compression
        original_text = scenario["content"] * 100  # Make it longer
        compressed, stats = memory._compress_content(original_text)

        print(f"  Original Size: {len(original_text)} chars")
        print(f"  Compressed Size: {len(compressed)} chars")
        print(f"  Compression Ratio: {stats['compression_ratio']:.2f}x")
        print(f"  Quality Score: {stats.get('quality_score', 0):.2%}")


# Example 5: Custom Memory Retrieval
def example_custom_retrieval():
    """Demonstrate custom memory retrieval with MMR."""
    from langchain.agents import initialize_agent, AgentType
    from langchain.llms import OpenAI
    from langchain_adapter import OmniMemory, OmniMemoryConfig
    from memory_layers import MemoryLayer

    # Configure with MMR enabled for diverse retrieval
    config = OmniMemoryConfig(
        agent_id="retrieval_agent",
        enable_mmr=True,  # Enable Maximal Marginal Relevance
        compression_mode="BALANCED",
    )

    memory = OmniMemory(config=config)

    # Create agent
    agent = initialize_agent(
        [],
        OpenAI(temperature=0),
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
    )

    # Add diverse conversations
    topics = [
        "Explain Python decorators",
        "What are Python context managers?",
        "How does Python garbage collection work?",
        "Explain JavaScript closures",
        "What are JavaScript promises?",
        "How does JavaScript event loop work?",
    ]

    for topic in topics:
        agent.run(topic)

    # Retrieve relevant but diverse memories
    query = "programming concepts"
    relevant_memories = memory.get_relevant_memories(
        query=query, k=3, memory_layers=[MemoryLayer.SESSION]
    )

    print(f"\nRetrieved Memories for '{query}':")
    for i, mem in enumerate(relevant_memories, 1):
        print(f"\n{i}. {mem[:100]}...")


# Example 6: Production Setup with Error Handling
def example_production_setup():
    """Production-ready setup with monitoring."""
    from langchain.agents import initialize_agent, AgentType
    from langchain.llms import OpenAI
    from langchain.callbacks import StdOutCallbackHandler
    from langchain_adapter import OmniMemory, OmniMemoryConfig
    import logging

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Production configuration
    config = OmniMemoryConfig(
        agent_id="prod_agent_001",
        compression_mode="BALANCED",
        enable_sharing=True,
        shared_pool_id="production_pool",
        batch_compress=True,  # Batch compression for efficiency
        cache_compressed=True,  # Cache results
        track_savings=True,  # Track cost savings
    )

    try:
        memory = OmniMemory(config=config)

        # Create agent with callbacks
        agent = initialize_agent(
            [],
            OpenAI(temperature=0),
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            callbacks=[StdOutCallbackHandler()],
            verbose=True,
        )

        # Monitor performance
        def monitor_agent_performance():
            stats = memory.get_stats()
            logger.info(f"Performance Metrics:")
            logger.info(f"  Tokens Saved: {stats['tokens_saved']:,}")
            logger.info(f"  Compression Ratio: {stats['compression_ratio']:.2f}x")
            logger.info(f"  Quality Score: {stats.get('average_quality', 0):.2%}")

            # Calculate cost savings (assuming $0.003 per 1K tokens)
            cost_saved = (stats["tokens_saved"] / 1000) * 0.003
            logger.info(f"  Estimated Cost Saved: ${cost_saved:.2f}")

            return stats

        # Run agent tasks
        tasks = [
            "Create a comprehensive plan for building a web application",
            "What are the security considerations?",
            "How should we handle scalability?",
        ]

        for task in tasks:
            try:
                response = agent.run(task)
                logger.info(f"Task completed: {task[:50]}...")
            except Exception as e:
                logger.error(f"Task failed: {e}")

        # Final metrics
        final_stats = monitor_agent_performance()

        # Check if compression is effective
        if final_stats["compression_ratio"] < 2.0:
            logger.warning(
                "Compression ratio below target. Consider adjusting settings."
            )

    except Exception as e:
        logger.error(f"Agent initialization failed: {e}")
        raise


# Main execution
if __name__ == "__main__":
    print("OmniMemory + LangChain Integration Examples")
    print("=" * 50)

    # Uncomment to run examples
    # example_basic_usage()
    # example_multi_agent_collaboration()
    # example_memory_layers()
    # example_compression_modes()
    # example_custom_retrieval()
    # example_production_setup()

    print("\nTo run examples, uncomment the desired function calls in __main__")
    print("\nQuick Start:")
    print("  1. Install dependencies: pip install langchain openai")
    print("  2. Set OPENAI_API_KEY environment variable")
    print("  3. Ensure compression server is running on port 8001")
    print("  4. Run: python langchain_usage.py")
