"""
Example: Using OmniMemory with LlamaIndex

Demonstrates integration with LlamaIndex's query engine.
"""

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from omnimemory_llamaindex import OmniMemoryNodePostprocessor

# Sample documents
documents = [
    Document(
        text="""
        Quantum computing is the use of quantum phenomena such as superposition
        and entanglement to perform computation. Computers that perform quantum
        computations are known as quantum computers. Quantum computers are believed
        to be able to solve certain computational problems, such as integer
        factorization, substantially faster than classical computers.
        """,
        metadata={"source": "quantum_intro.txt"},
    ),
    Document(
        text="""
        Quantum supremacy or quantum advantage is the goal of demonstrating that
        a programmable quantum device can solve a problem that no classical
        computer can solve in any feasible amount of time (irrespective of the
        usefulness of the problem). The term was coined by John Preskill in 2012.
        """,
        metadata={"source": "quantum_supremacy.txt"},
    ),
    Document(
        text="""
        A qubit is a two-state quantum-mechanical system, one of the simplest
        quantum systems displaying the peculiarity of quantum mechanics. Examples
        include the spin of the electron in which the two levels can be taken as
        spin up and spin down; or the polarization of a single photon in which
        the two states can be taken to be the vertical polarization and the
        horizontal polarization.
        """,
        metadata={"source": "qubit.txt"},
    ),
]


def basic_compression_example():
    """Basic node compression example"""
    print("=== Basic Node Compression Example ===\n")

    # Create index
    index = VectorStoreIndex.from_documents(documents)

    # Create OmniMemory postprocessor
    compressor = OmniMemoryNodePostprocessor(
        base_url="http://localhost:8001",
        target_compression=0.5,  # 50% compression
        model_id="gpt-4",
    )

    # Create query engine with compression
    query_engine = index.as_query_engine(
        node_postprocessors=[compressor], similarity_top_k=3
    )

    # Query
    query = "What is quantum computing?"
    response = query_engine.query(query)

    print(f"Query: {query}")
    print(f"\nResponse: {response}")

    # Check compression metadata
    if response.source_nodes:
        node = response.source_nodes[0]
        print(f"\nCompression metadata:")
        print(f"Original tokens: {node.node.metadata.get('original_tokens', 'N/A')}")
        print(
            f"Compressed tokens: {node.node.metadata.get('compressed_tokens', 'N/A')}"
        )
        print(
            f"Compression ratio: {node.node.metadata.get('compression_ratio', 0):.2%}"
        )
        print(f"Quality score: {node.node.metadata.get('quality_score', 0):.2%}\n")


def aggressive_compression_example():
    """Example with aggressive compression"""
    print("=== Aggressive Compression Example ===\n")

    # Create index
    index = VectorStoreIndex.from_documents(documents)

    # Create compressor with aggressive compression
    compressor = OmniMemoryNodePostprocessor(
        base_url="http://localhost:8001",
        target_compression=0.8,  # 80% compression
        model_id="gpt-4",
    )

    # Create query engine
    query_engine = index.as_query_engine(
        node_postprocessors=[compressor], similarity_top_k=3
    )

    # Query
    query = "Explain quantum supremacy"
    response = query_engine.query(query)

    print(f"Query: {query}")
    print(f"\nResponse: {response}")

    if response.source_nodes:
        node = response.source_nodes[0]
        print(f"\nCompressed text:")
        print(node.node.get_content())
        print(
            f"\nTokens: {node.node.metadata.get('original_tokens', 'N/A')} → "
            f"{node.node.metadata.get('compressed_tokens', 'N/A')}\n"
        )


def multi_postprocessor_example():
    """Example combining multiple postprocessors"""
    from llama_index.core.postprocessor import SimilarityPostprocessor

    print("=== Multiple Postprocessors Example ===\n")

    # Create index
    index = VectorStoreIndex.from_documents(documents)

    # Create postprocessors
    similarity_filter = SimilarityPostprocessor(similarity_cutoff=0.7)
    compressor = OmniMemoryNodePostprocessor(
        base_url="http://localhost:8001", target_compression=0.6
    )

    # Create query engine with multiple postprocessors
    query_engine = index.as_query_engine(
        node_postprocessors=[
            similarity_filter,  # Filter by similarity first
            compressor,  # Then compress
        ],
        similarity_top_k=5,
    )

    # Query
    query = "What is a qubit?"
    response = query_engine.query(query)

    print(f"Query: {query}")
    print(f"\nResponse: {response}")
    print(f"\nSource nodes: {len(response.source_nodes)}\n")


async def async_compression_example():
    """Async compression example"""
    print("=== Async Compression Example ===\n")

    # Create index
    index = VectorStoreIndex.from_documents(documents)

    # Create compressor
    compressor = OmniMemoryNodePostprocessor(
        base_url="http://localhost:8001", target_compression=0.5
    )

    # Create query engine
    query_engine = index.as_query_engine(node_postprocessors=[compressor])

    # Async query
    query = "What is quantum computing?"
    response = await query_engine.aquery(query)

    print(f"Query: {query}")
    print(f"\nResponse: {response}\n")


def custom_llm_example():
    """Example with custom LLM"""
    print("=== Custom LLM Example ===\n")

    # Create index with custom LLM
    llm = OpenAI(model="gpt-4-turbo", temperature=0)
    index = VectorStoreIndex.from_documents(documents)

    # Create compressor
    compressor = OmniMemoryNodePostprocessor(
        base_url="http://localhost:8001", target_compression=0.5, model_id="gpt-4-turbo"
    )

    # Create query engine
    query_engine = index.as_query_engine(llm=llm, node_postprocessors=[compressor])

    # Query
    query = "Compare quantum computing with classical computing"
    response = query_engine.query(query)

    print(f"Query: {query}")
    print(f"\nResponse: {response}\n")


if __name__ == "__main__":
    # Note: These examples require OpenAI API key
    # export OPENAI_API_KEY="your-key-here"

    try:
        basic_compression_example()
        aggressive_compression_example()
        multi_postprocessor_example()
        custom_llm_example()

        # Async example
        import asyncio

        asyncio.run(async_compression_example())

    except Exception as e:
        print(f"Error: {e}")
        print(
            "\nNote: Make sure to set OPENAI_API_KEY environment variable and "
            "start OmniMemory service on localhost:8001"
        )
