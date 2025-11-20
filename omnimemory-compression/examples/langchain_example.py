"""
Example: Using OmniMemory with LangChain

Demonstrates integration with LangChain's ContextualCompressionRetriever.
"""

from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from omnimemory_langchain import OmniMemoryDocumentCompressor

# Sample documents
documents = [
    Document(
        page_content="""
        Artificial intelligence (AI) is intelligence demonstrated by machines,
        in contrast to the natural intelligence displayed by humans and animals.
        Leading AI textbooks define the field as the study of "intelligent agents":
        any device that perceives its environment and takes actions that maximize
        its chance of successfully achieving its goals.
        """,
        metadata={"source": "ai_intro.txt"},
    ),
    Document(
        page_content="""
        Machine learning (ML) is a field of inquiry devoted to understanding and
        building methods that 'learn', that is, methods that leverage data to
        improve performance on some set of tasks. It is seen as a part of
        artificial intelligence. Machine learning algorithms build a model based
        on sample data, known as training data.
        """,
        metadata={"source": "ml_intro.txt"},
    ),
    Document(
        page_content="""
        Deep learning is part of a broader family of machine learning methods
        based on artificial neural networks with representation learning. Learning
        can be supervised, semi-supervised or unsupervised. Deep-learning
        architectures such as deep neural networks, deep belief networks,
        recurrent neural networks and convolutional neural networks have been
        applied to fields including computer vision, speech recognition, natural
        language processing.
        """,
        metadata={"source": "dl_intro.txt"},
    ),
]


def basic_compression_example():
    """Basic document compression example"""
    print("=== Basic Compression Example ===\n")

    # Create vector store
    vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Create OmniMemory compressor
    compressor = OmniMemoryDocumentCompressor(
        base_url="http://localhost:8001",
        target_compression=0.5,  # 50% compression
        model_id="gpt-4",
    )

    # Create compression retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    # Retrieve and compress documents
    query = "What is deep learning?"
    compressed_docs = compression_retriever.get_relevant_documents(query)

    print(f"Query: {query}")
    print(f"Number of compressed docs: {len(compressed_docs)}")
    print(f"\nCompressed document:")
    print(compressed_docs[0].page_content)
    print(f"\nMetadata:")
    print(f"Original tokens: {compressed_docs[0].metadata['original_tokens']}")
    print(f"Compressed tokens: {compressed_docs[0].metadata['compressed_tokens']}")
    print(f"Compression ratio: {compressed_docs[0].metadata['compression_ratio']:.2%}")
    print(f"Quality score: {compressed_docs[0].metadata['quality_score']:.2%}\n")


def rag_with_compression_example():
    """RAG pipeline with compression example"""
    print("=== RAG with Compression Example ===\n")

    # Create vector store
    vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Create OmniMemory compressor
    compressor = OmniMemoryDocumentCompressor(
        base_url="http://localhost:8001",
        target_compression=0.7,  # More aggressive compression
    )

    # Create compression retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    # Create QA chain with compression
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4", temperature=0),
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True,
    )

    # Ask question
    query = "Explain the relationship between AI, ML, and deep learning."
    result = qa_chain({"query": query})

    print(f"Query: {query}")
    print(f"\nAnswer: {result['result']}")
    print(f"\nSource documents used: {len(result['source_documents'])}")

    if result["source_documents"]:
        doc = result["source_documents"][0]
        print(f"Compressed tokens: {doc.metadata.get('compressed_tokens', 'N/A')}")
        print(f"Quality score: {doc.metadata.get('quality_score', 'N/A'):.2%}\n")


def prompt_compression_example():
    """Prompt compression example"""
    from omnimemory_langchain import OmniMemoryPromptCompressor

    print("=== Prompt Compression Example ===\n")

    compressor = OmniMemoryPromptCompressor(
        base_url="http://localhost:8001", target_compression=0.5
    )

    # Long context to compress
    long_context = """
    Natural language processing (NLP) is a subfield of linguistics, computer science,
    and artificial intelligence concerned with the interactions between computers and
    human language, in particular how to program computers to process and analyze
    large amounts of natural language data. The result is a computer capable of
    "understanding" the contents of documents, including the contextual nuances of
    the language within them. The technology can then accurately extract information
    and insights contained in the documents as well as categorize and organize the
    documents themselves.
    """

    # Compress context
    compressed = compressor.compress(
        context=long_context, query="What is NLP used for?"
    )

    print(f"Original length: {len(long_context)} chars")
    print(f"Compressed length: {len(compressed)} chars")
    print(f"\nCompressed context:\n{compressed}\n")

    # Use in prompt
    query = "What is NLP used for?"
    prompt = f"Context: {compressed}\n\nQuestion: {query}\n\nAnswer:"
    print(f"Final prompt:\n{prompt}")


if __name__ == "__main__":
    # Note: These examples require OpenAI API key
    # export OPENAI_API_KEY="your-key-here"

    try:
        basic_compression_example()
        # rag_with_compression_example()  # Uncomment if you have OpenAI API key
        prompt_compression_example()
    except Exception as e:
        print(f"Error: {e}")
        print(
            "\nNote: Make sure to set OPENAI_API_KEY environment variable and "
            "start OmniMemory service on localhost:8001"
        )
