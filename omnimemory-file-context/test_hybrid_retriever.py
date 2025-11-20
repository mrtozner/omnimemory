"""
Test Suite for Hybrid File Retriever

Demonstrates that hybrid retrieval outperforms both:
- Dense-only (semantic search)
- Sparse-only (BM25 keyword search)

Test Scenarios:
1. Semantic queries (where dense wins)
2. Keyword queries (where sparse wins)
3. Mixed queries (where hybrid wins)
4. Structural queries (where fact matching helps)
"""

import pytest
import numpy as np
import logging
import tempfile
import os
from pathlib import Path
from typing import List, Dict

from hybrid_retriever import HybridFileRetriever, HybridSearchResult
from bm25_index import BM25Index

# Try to import Qdrant
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    pytest.skip("Qdrant not available", allow_module_level=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Test data: Sample code files with known content
SAMPLE_FILES = {
    "auth/jwt_handler.py": """
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional

class JWTHandler:
    '''Handles JWT token generation and validation for authentication'''

    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    def generate_token(self, user_id: str, expires_in: int = 3600) -> str:
        '''Generate JWT token for authenticated user'''
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')

    def validate_token(self, token: str) -> Optional[str]:
        '''Validate JWT token and return user_id'''
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload.get('user_id')
        except jwt.ExpiredSignatureError:
            return None
""",
    "auth/password_manager.py": """
import bcrypt
from typing import Tuple

class PasswordManager:
    '''Manages password hashing and verification using bcrypt'''

    def hash_password(self, password: str) -> bytes:
        '''Hash a password using bcrypt'''
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt)

    def verify_password(self, password: str, hashed: bytes) -> bool:
        '''Verify a password against its hash'''
        return bcrypt.checkpw(password.encode('utf-8'), hashed)

    def change_password(self, old_password: str, new_password: str,
                       current_hash: bytes) -> Tuple[bool, bytes]:
        '''Change user password after verification'''
        if not self.verify_password(old_password, current_hash):
            return False, current_hash
        return True, self.hash_password(new_password)
""",
    "auth/user_service.py": """
from typing import Optional
from dataclasses import dataclass

@dataclass
class User:
    id: str
    username: str
    email: str
    password_hash: bytes

class UserService:
    '''Service for managing user accounts and authentication'''

    def __init__(self, db_connection):
        self.db = db_connection

    def create_user(self, username: str, email: str, password_hash: bytes) -> User:
        '''Create a new user account'''
        # Implementation details...
        pass

    def authenticate(self, username: str, password: str) -> Optional[User]:
        '''Authenticate user with username and password'''
        # Implementation details...
        pass

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        '''Retrieve user by ID'''
        # Implementation details...
        pass
""",
    "database/connection_pool.py": """
import psycopg2
from psycopg2 import pool
from typing import Optional

class DatabaseConnectionPool:
    '''Manages PostgreSQL database connection pool'''

    def __init__(self, host: str, port: int, database: str,
                 user: str, password: str, min_conn: int = 5, max_conn: int = 20):
        self.connection_pool = psycopg2.pool.SimpleConnectionPool(
            min_conn, max_conn,
            host=host, port=port, database=database,
            user=user, password=password
        )

    def get_connection(self):
        '''Get a connection from the pool'''
        return self.connection_pool.getconn()

    def release_connection(self, connection):
        '''Return a connection to the pool'''
        self.connection_pool.putconn(connection)

    def close_all(self):
        '''Close all connections in the pool'''
        self.connection_pool.closeall()
""",
    "api/auth_routes.py": """
from flask import Flask, request, jsonify
from auth.jwt_handler import JWTHandler
from auth.password_manager import PasswordManager
from auth.user_service import UserService

app = Flask(__name__)
jwt_handler = JWTHandler(secret_key='secret')
password_manager = PasswordManager()
user_service = UserService(db_connection=None)

@app.route('/api/login', methods=['POST'])
def login():
    '''Login endpoint - authenticate user and return JWT token'''
    data = request.json
    username = data.get('username')
    password = data.get('password')

    user = user_service.authenticate(username, password)
    if user:
        token = jwt_handler.generate_token(user.id)
        return jsonify({'token': token})
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/register', methods=['POST'])
def register():
    '''Registration endpoint - create new user account'''
    data = request.json
    password_hash = password_manager.hash_password(data['password'])
    user = user_service.create_user(
        data['username'], data['email'], password_hash
    )
    return jsonify({'user_id': user.id})
""",
}

# Ground truth: Relevant files for each query
GROUND_TRUTH = {
    "JWT token generation": [
        "auth/jwt_handler.py",  # Most relevant
        "api/auth_routes.py",  # Uses JWT
    ],
    "password hashing bcrypt": [
        "auth/password_manager.py",  # Most relevant
        "api/auth_routes.py",  # Uses password manager
    ],
    "user authentication": [
        "auth/user_service.py",  # Most relevant
        "api/auth_routes.py",  # Authentication endpoints
        "auth/jwt_handler.py",  # Part of auth flow
    ],
    "database connection pool": [
        "database/connection_pool.py",  # Most relevant
    ],
    "authenticate user credentials": [
        "auth/user_service.py",  # authenticate method
        "api/auth_routes.py",  # login endpoint
        "auth/password_manager.py",  # verify_password
    ],
}


class TestDataGenerator:
    """Helper to generate test embeddings and populate indexes"""

    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.bm25_db_path = os.path.join(temp_dir, "test_bm25.db")
        self.qdrant_collection = "test_hybrid_retrieval"

    def generate_embedding(self, text: str, dimension: int = 768) -> np.ndarray:
        """
        Generate deterministic embedding from text using simple hash-based approach.

        In production, would use actual embedding model.
        """
        # Simple hash-based embedding for testing
        # Use text hash to seed random generator for determinism
        hash_val = hash(text) % (2**32)
        np.random.seed(hash_val)

        # Generate base embedding
        embedding = np.random.randn(dimension).astype(np.float32)

        # Add content-based features for similarity
        # Extract key terms and weight embedding accordingly
        tokens = text.lower().split()
        for token in tokens[:100]:  # First 100 tokens
            token_hash = hash(token) % dimension
            embedding[token_hash] += 0.1

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def setup_indexes(self) -> tuple:
        """
        Setup BM25 and Qdrant indexes with test data.

        Returns:
            (bm25_index, qdrant_client, embeddings_map)
        """
        # Initialize BM25 index
        bm25_index = BM25Index(db_path=self.bm25_db_path)

        # Index all files in BM25
        for file_path, content in SAMPLE_FILES.items():
            bm25_index.index_file(file_path, content, language="python")

        logger.info(f"✓ Indexed {len(SAMPLE_FILES)} files in BM25")

        # Initialize Qdrant
        qdrant_client = QdrantClient(url="http://localhost:6333")

        # Create collection
        try:
            qdrant_client.delete_collection(self.qdrant_collection)
        except:
            pass

        qdrant_client.create_collection(
            collection_name=self.qdrant_collection,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

        # Generate embeddings and populate Qdrant
        embeddings_map = {}
        points = []

        for idx, (file_path, content) in enumerate(SAMPLE_FILES.items()):
            embedding = self.generate_embedding(content)
            embeddings_map[file_path] = embedding

            # Create mock facts (structural information)
            facts = self._extract_mock_facts(content, file_path)

            # Create mock witnesses (important lines)
            witnesses = self._extract_mock_witnesses(content)

            point = PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={
                    "file_path": file_path,
                    "facts": facts,
                    "witnesses": witnesses,
                    "last_modified": "2024-01-15T10:00:00",
                },
            )
            points.append(point)

        qdrant_client.upsert(
            collection_name=self.qdrant_collection,
            points=points,
        )

        logger.info(f"✓ Indexed {len(points)} files in Qdrant")

        return bm25_index, qdrant_client, embeddings_map

    def _extract_mock_facts(self, content: str, file_path: str) -> List[Dict]:
        """Extract structural facts from content"""
        facts = []

        # Extract imports
        for line in content.split("\n"):
            if line.strip().startswith("import ") or line.strip().startswith("from "):
                parts = line.split()
                if "import" in parts:
                    idx = parts.index("import")
                    if idx + 1 < len(parts):
                        module = parts[idx + 1].rstrip(",")
                        facts.append(
                            {
                                "predicate": "imports",
                                "object": f"module:{module}",
                                "confidence": 1.0,
                            }
                        )

        # Extract classes
        for line in content.split("\n"):
            if line.strip().startswith("class "):
                class_name = line.split("class ")[1].split("(")[0].split(":")[0].strip()
                facts.append(
                    {
                        "predicate": "defines_class",
                        "object": f"class:{class_name}",
                        "confidence": 1.0,
                    }
                )

        # Extract functions
        for line in content.split("\n"):
            if line.strip().startswith("def "):
                func_name = line.split("def ")[1].split("(")[0].strip()
                facts.append(
                    {
                        "predicate": "defines_function",
                        "object": f"function:{func_name}",
                        "confidence": 1.0,
                    }
                )

        return facts

    def _extract_mock_witnesses(self, content: str) -> List[str]:
        """Extract important lines as witnesses"""
        witnesses = []

        # Extract docstrings and comments
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("'''") or stripped.startswith('"""'):
                docstring = stripped.strip("'\"")
                if len(docstring) > 10:
                    witnesses.append(docstring)

        return witnesses[:5]  # Max 5 witnesses


def calculate_recall_at_k(
    results: List[HybridSearchResult], ground_truth: List[str], k: int = 5
) -> float:
    """Calculate Recall@K metric"""
    if not ground_truth:
        return 0.0

    retrieved_files = {r.file_path for r in results[:k]}
    relevant_files = set(ground_truth)

    hits = len(retrieved_files & relevant_files)
    return hits / len(relevant_files)


def calculate_mrr(results: List[HybridSearchResult], ground_truth: List[str]) -> float:
    """Calculate Mean Reciprocal Rank"""
    for rank, result in enumerate(results, 1):
        if result.file_path in ground_truth:
            return 1.0 / rank
    return 0.0


@pytest.fixture(scope="module")
def test_setup():
    """Setup test environment with indexes"""
    with tempfile.TemporaryDirectory() as temp_dir:
        generator = TestDataGenerator(temp_dir)
        bm25_index, qdrant_client, embeddings_map = generator.setup_indexes()

        # Create retriever
        retriever = HybridFileRetriever(
            bm25_db_path=generator.bm25_db_path,
            qdrant_host="localhost",
            qdrant_port=6333,
            qdrant_collection=generator.qdrant_collection,
        )

        yield {
            "retriever": retriever,
            "bm25_index": bm25_index,
            "qdrant_client": qdrant_client,
            "embeddings_map": embeddings_map,
            "generator": generator,
        }

        # Cleanup
        retriever.close()
        try:
            qdrant_client.delete_collection(generator.qdrant_collection)
        except:
            pass


def test_hybrid_vs_dense_only(test_setup):
    """Test that hybrid search outperforms dense-only search"""
    retriever = test_setup["retriever"]
    generator = test_setup["generator"]

    query = "JWT token generation"
    query_embedding = generator.generate_embedding(query)
    ground_truth = GROUND_TRUTH[query]

    # Dense-only search
    dense_results = retriever.dense_search(query_embedding, limit=5)
    dense_files = [r[0] for r in dense_results]
    dense_recall = calculate_recall_at_k(
        [HybridSearchResult(file_path=f, final_score=s) for f, s, _ in dense_results],
        ground_truth,
        k=5,
    )

    # Hybrid search
    hybrid_results = retriever.search_files(
        query=query,
        query_embedding=query_embedding,
        limit=5,
    )
    hybrid_recall = calculate_recall_at_k(hybrid_results, ground_truth, k=5)

    logger.info(f"\nQuery: '{query}'")
    logger.info(f"Dense-only Recall@5: {dense_recall:.2%}")
    logger.info(f"Hybrid Recall@5: {hybrid_recall:.2%}")
    logger.info(
        f"Improvement: {(hybrid_recall - dense_recall) / (dense_recall + 1e-8):.1%}"
    )

    # Hybrid should be at least as good as dense-only
    assert (
        hybrid_recall >= dense_recall
    ), f"Hybrid recall ({hybrid_recall:.2%}) should >= dense recall ({dense_recall:.2%})"


def test_hybrid_vs_sparse_only(test_setup):
    """Test that hybrid search outperforms sparse-only search"""
    retriever = test_setup["retriever"]
    generator = test_setup["generator"]

    query = "password hashing bcrypt"
    query_embedding = generator.generate_embedding(query)
    ground_truth = GROUND_TRUTH[query]

    # Sparse-only search
    sparse_results = retriever.sparse_search(query, limit=5)
    sparse_recall = calculate_recall_at_k(
        [
            HybridSearchResult(file_path=r.file_path, final_score=r.score)
            for r in sparse_results
        ],
        ground_truth,
        k=5,
    )

    # Hybrid search
    hybrid_results = retriever.search_files(
        query=query,
        query_embedding=query_embedding,
        limit=5,
    )
    hybrid_recall = calculate_recall_at_k(hybrid_results, ground_truth, k=5)

    logger.info(f"\nQuery: '{query}'")
    logger.info(f"Sparse-only Recall@5: {sparse_recall:.2%}")
    logger.info(f"Hybrid Recall@5: {hybrid_recall:.2%}")
    logger.info(
        f"Improvement: {(hybrid_recall - sparse_recall) / (sparse_recall + 1e-8):.1%}"
    )

    # Hybrid should be at least as good as sparse-only
    assert (
        hybrid_recall >= sparse_recall
    ), f"Hybrid recall ({hybrid_recall:.2%}) should >= sparse recall ({sparse_recall:.2%})"


def test_hybrid_on_mixed_queries(test_setup):
    """Test hybrid search on queries that need both semantic and keyword matching"""
    retriever = test_setup["retriever"]
    generator = test_setup["generator"]

    query = "authenticate user credentials"
    query_embedding = generator.generate_embedding(query)
    ground_truth = GROUND_TRUTH[query]

    # Dense-only
    dense_results = retriever.dense_search(query_embedding, limit=5)
    dense_recall = calculate_recall_at_k(
        [HybridSearchResult(file_path=f, final_score=s) for f, s, _ in dense_results],
        ground_truth,
        k=5,
    )

    # Sparse-only
    sparse_results = retriever.sparse_search(query, limit=5)
    sparse_recall = calculate_recall_at_k(
        [
            HybridSearchResult(file_path=r.file_path, final_score=r.score)
            for r in sparse_results
        ],
        ground_truth,
        k=5,
    )

    # Hybrid
    hybrid_results = retriever.search_files(
        query=query,
        query_embedding=query_embedding,
        limit=5,
    )
    hybrid_recall = calculate_recall_at_k(hybrid_results, ground_truth, k=5)

    logger.info(f"\nQuery: '{query}'")
    logger.info(f"Dense-only Recall@5: {dense_recall:.2%}")
    logger.info(f"Sparse-only Recall@5: {sparse_recall:.2%}")
    logger.info(f"Hybrid Recall@5: {hybrid_recall:.2%}")

    # Hybrid should beat both individual methods
    assert (
        hybrid_recall >= dense_recall
    ), f"Hybrid ({hybrid_recall:.2%}) should >= dense ({dense_recall:.2%})"
    assert (
        hybrid_recall >= sparse_recall
    ), f"Hybrid ({hybrid_recall:.2%}) should >= sparse ({sparse_recall:.2%})"


def test_fact_matching_improves_results(test_setup):
    """Test that structural fact matching improves retrieval"""
    retriever = test_setup["retriever"]
    generator = test_setup["generator"]

    # Query with clear structural hints
    query = "JWTHandler class for token generation"
    query_embedding = generator.generate_embedding(query)

    results = retriever.search_files(
        query=query,
        query_embedding=query_embedding,
        limit=5,
    )

    # Check that fact matching contributed to scores
    has_fact_scores = any(r.fact_score > 0 for r in results)
    assert has_fact_scores, "Fact matching should contribute to at least some results"

    # Check that the correct file (with JWTHandler class) ranks high
    top_file = results[0].file_path
    logger.info(f"\nQuery: '{query}'")
    logger.info(f"Top result: {top_file}")
    logger.info(f"Fact score: {results[0].fact_score:.3f}")

    # The file with JWTHandler should be in top results
    jwt_handler_found = any("jwt_handler" in r.file_path for r in results[:3])
    assert jwt_handler_found, "File with JWTHandler class should be in top-3 results"


def test_witness_reranking(test_setup):
    """Test that witness reranking adjusts scores"""
    retriever = test_setup["retriever"]
    generator = test_setup["generator"]

    query = "handles JWT token generation and validation"
    query_embedding = generator.generate_embedding(query)

    # Search without witness reranking
    results_no_rerank = retriever.search_files(
        query=query,
        query_embedding=query_embedding,
        limit=5,
        enable_witness_rerank=False,
    )

    # Search with witness reranking
    results_with_rerank = retriever.search_files(
        query=query,
        query_embedding=query_embedding,
        limit=5,
        enable_witness_rerank=True,
    )

    logger.info(f"\nQuery: '{query}'")
    logger.info("Without reranking:")
    for r in results_no_rerank[:3]:
        logger.info(f"  {r.file_path}: {r.final_score:.3f}")

    logger.info("With reranking:")
    for r in results_with_rerank[:3]:
        logger.info(f"  {r.file_path}: {r.final_score:.3f}")

    # Scores should be different (reranking has effect)
    scores_changed = any(
        abs(r1.final_score - r2.final_score) > 0.01
        for r1, r2 in zip(results_no_rerank, results_with_rerank)
    )
    assert scores_changed, "Witness reranking should affect scores"


def test_comprehensive_metrics(test_setup):
    """Comprehensive evaluation across all test queries"""
    retriever = test_setup["retriever"]
    generator = test_setup["generator"]

    results_summary = {
        "dense_only": {"recalls": [], "mrrs": []},
        "sparse_only": {"recalls": [], "mrrs": []},
        "hybrid": {"recalls": [], "mrrs": []},
    }

    for query, ground_truth in GROUND_TRUTH.items():
        query_embedding = generator.generate_embedding(query)

        # Dense-only
        dense_results = retriever.dense_search(query_embedding, limit=5)
        dense_results_wrapped = [
            HybridSearchResult(file_path=f, final_score=s) for f, s, _ in dense_results
        ]
        dense_recall = calculate_recall_at_k(dense_results_wrapped, ground_truth, k=5)
        dense_mrr = calculate_mrr(dense_results_wrapped, ground_truth)

        # Sparse-only
        sparse_results = retriever.sparse_search(query, limit=5)
        sparse_results_wrapped = [
            HybridSearchResult(file_path=r.file_path, final_score=r.score)
            for r in sparse_results
        ]
        sparse_recall = calculate_recall_at_k(sparse_results_wrapped, ground_truth, k=5)
        sparse_mrr = calculate_mrr(sparse_results_wrapped, ground_truth)

        # Hybrid
        hybrid_results = retriever.search_files(
            query=query,
            query_embedding=query_embedding,
            limit=5,
        )
        hybrid_recall = calculate_recall_at_k(hybrid_results, ground_truth, k=5)
        hybrid_mrr = calculate_mrr(hybrid_results, ground_truth)

        results_summary["dense_only"]["recalls"].append(dense_recall)
        results_summary["dense_only"]["mrrs"].append(dense_mrr)
        results_summary["sparse_only"]["recalls"].append(sparse_recall)
        results_summary["sparse_only"]["mrrs"].append(sparse_mrr)
        results_summary["hybrid"]["recalls"].append(hybrid_recall)
        results_summary["hybrid"]["mrrs"].append(hybrid_mrr)

    # Calculate averages
    avg_dense_recall = np.mean(results_summary["dense_only"]["recalls"])
    avg_sparse_recall = np.mean(results_summary["sparse_only"]["recalls"])
    avg_hybrid_recall = np.mean(results_summary["hybrid"]["recalls"])

    avg_dense_mrr = np.mean(results_summary["dense_only"]["mrrs"])
    avg_sparse_mrr = np.mean(results_summary["sparse_only"]["mrrs"])
    avg_hybrid_mrr = np.mean(results_summary["hybrid"]["mrrs"])

    logger.info("\n" + "=" * 80)
    logger.info("COMPREHENSIVE EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"\nAverage Recall@5:")
    logger.info(f"  Dense-only:  {avg_dense_recall:.2%}")
    logger.info(f"  Sparse-only: {avg_sparse_recall:.2%}")
    logger.info(f"  Hybrid:      {avg_hybrid_recall:.2%}")
    logger.info(f"\nAverage MRR:")
    logger.info(f"  Dense-only:  {avg_dense_mrr:.3f}")
    logger.info(f"  Sparse-only: {avg_sparse_mrr:.3f}")
    logger.info(f"  Hybrid:      {avg_hybrid_mrr:.3f}")
    logger.info("=" * 80)

    # Hybrid should beat or match both individual methods on average
    assert avg_hybrid_recall >= max(
        avg_dense_recall, avg_sparse_recall
    ), "Hybrid should have best average recall"

    # Print relative improvements
    dense_improvement = (
        (avg_hybrid_recall - avg_dense_recall) / (avg_dense_recall + 1e-8) * 100
    )
    sparse_improvement = (
        (avg_hybrid_recall - avg_sparse_recall) / (avg_sparse_recall + 1e-8) * 100
    )

    logger.info(f"\nHybrid Improvements:")
    logger.info(f"  vs Dense:  +{dense_improvement:.1f}%")
    logger.info(f"  vs Sparse: +{sparse_improvement:.1f}%")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
