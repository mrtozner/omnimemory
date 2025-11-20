"""
BM25 Index Integration with Tri-Index System

Example showing how to integrate BM25 sparse index with FileTriIndex.
"""

from bm25_index import BM25Index
from typing import Dict


def create_tri_index_with_bm25(file_path: str, content: str, language: str = "python"):
    """
    Example: Create Tri-Index artifacts including BM25 tokens.

    This shows how to integrate BM25 into the Tri-Index creation pipeline.

    Args:
        file_path: Path to the file
        content: File content
        language: Programming language

    Returns:
        Dict containing tri-index artifacts including bm25_tokens
    """
    # Initialize BM25 index
    bm25_index = BM25Index(db_path="tri_index_bm25.db")

    # Index the file
    bm25_index.index_file(file_path, content, language=language)

    # Get top-20 tokens for this file (for storing in FileTriIndex)
    bm25_tokens = bm25_index.get_top_tokens(file_path, limit=20)

    # Close index
    bm25_index.close()

    # Return tri-index artifacts
    return {
        "file_path": file_path,
        "bm25_tokens": bm25_tokens,  # Top-20 keywords with TF-IDF scores
        # Other tri-index components would go here:
        # "quantized_vector": ...,
        # "sub_chunk_vectors": ...,
        # "witness_sentences": ...,
        # etc.
    }


def hybrid_search(
    query: str, dense_results: list, bm25_db_path: str = "tri_index_bm25.db"
):
    """
    Example: Hybrid search combining dense (semantic) and sparse (BM25) results.

    Args:
        query: Search query
        dense_results: Results from dense vector search (semantic)
        bm25_db_path: Path to BM25 database

    Returns:
        Combined and re-ranked results
    """
    # Get sparse results from BM25
    bm25_index = BM25Index(db_path=bm25_db_path)
    sparse_results = bm25_index.search(query, limit=60)
    bm25_index.close()

    # Combine results (simple approach: weighted sum)
    # More sophisticated: Reciprocal Rank Fusion (RRF)
    combined_scores = {}

    # Add dense scores (weight 0.7)
    for i, result in enumerate(dense_results):
        file_path = result["file_path"]
        combined_scores[file_path] = result["score"] * 0.7

    # Add sparse scores (weight 0.3)
    for result in sparse_results:
        file_path = result.file_path
        if file_path in combined_scores:
            combined_scores[file_path] += result.score * 0.3
        else:
            combined_scores[file_path] = result.score * 0.3

    # Sort by combined score
    ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    return [{"file_path": fp, "score": score} for fp, score in ranked]


# Example usage
if __name__ == "__main__":
    # Example 1: Create tri-index with BM25
    sample_code = """
import bcrypt
from typing import Optional

class AuthManager:
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        '''Authenticate user and return JWT token'''
        return None
"""

    print("=== Example 1: Create Tri-Index with BM25 ===")
    tri_index = create_tri_index_with_bm25("auth.py", sample_code, language="python")
    print(f"File: {tri_index['file_path']}")
    print(f"Top BM25 tokens:")
    for token, score in list(tri_index["bm25_tokens"].items())[:10]:
        print(f"  {token:20s} | TF-IDF={score:.3f}")

    print("\n=== Example 2: Hybrid Search ===")
    # Simulated dense results
    dense_results = [
        {"file_path": "auth.py", "score": 0.95},
        {"file_path": "user.py", "score": 0.82},
    ]

    # Initialize BM25 for search
    bm25 = BM25Index(db_path="tri_index_bm25.db")
    bm25.index_file("auth.py", sample_code, language="python")
    bm25.index_file(
        "user.py",
        "class User:\n    def __init__(self, name): self.name = name",
        language="python",
    )
    bm25.close()

    # Perform hybrid search
    combined = hybrid_search(
        "authenticate user", dense_results, bm25_db_path="tri_index_bm25.db"
    )
    print(f"Query: 'authenticate user'")
    print(f"Combined results:")
    for result in combined:
        print(f"  {result['file_path']:20s} | score={result['score']:.3f}")

    # Cleanup
    import os

    if os.path.exists("tri_index_bm25.db"):
        os.unlink("tri_index_bm25.db")

    print("\n✅ Integration examples complete!")
