"""
BM25 Sparse Index for Exact Match Search

Implements BM25 (Best Match 25) algorithm for sparse lexical indexing of code files.
Extracts identifiers, imports, and keywords for fast exact-match search in Tri-Index system.

BM25 Formula:
    score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D| / avgdl))

    Where:
    - D = document (file)
    - Q = query
    - qi = query term
    - f(qi,D) = frequency of qi in D
    - |D| = length of D
    - avgdl = average document length in corpus
    - k1 = term frequency saturation parameter (default 1.5)
    - b = length normalization parameter (default 0.75)
    - IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5))
    - N = total number of documents
    - df(qi) = number of documents containing qi
"""

import ast
import re
import math
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BM25SearchResult:
    """Result from BM25 search"""

    file_path: str
    score: float
    matched_tokens: Dict[str, float]  # token -> contribution to score


class BM25Index:
    """
    BM25 sparse index for exact match search on code files.

    Features:
    - Smart code tokenization (classes, functions, imports, variables)
    - Weighted tokens (classes/functions > variables > comments)
    - Persistent SQLite storage
    - Incremental updates
    - BM25 ranking algorithm

    Storage Schema:
    - tokens: (token TEXT PRIMARY KEY, doc_freq INTEGER)
    - posting_lists: (token TEXT, file_path TEXT, term_freq INTEGER, weight REAL)
    - corpus_stats: (key TEXT PRIMARY KEY, value REAL)
    - file_metadata: (file_path TEXT PRIMARY KEY, doc_length INTEGER, last_indexed TIMESTAMP)
    """

    # Token weights for different code elements
    WEIGHTS = {
        "class": 5.0,  # Class names are very important
        "function": 4.0,  # Function/method names are important
        "import": 3.0,  # Import names are important
        "variable": 1.5,  # Variable names are somewhat important
        "keyword": 1.0,  # Keywords are baseline
        "comment": 0.5,  # Comments are less important
    }

    # BM25 parameters
    K1 = 1.5  # Term frequency saturation
    B = 0.75  # Length normalization

    def __init__(self, db_path: str = None):
        """
        Initialize BM25 index with SQLite storage.

        Args:
            db_path: Path to SQLite database file. If None, uses in-memory database.
        """
        if db_path is None:
            db_path = ":memory:"

        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

        # Cache for statistics (avoid repeated DB queries)
        self._stats_cache = {}
        self._refresh_stats_cache()

    def _init_schema(self):
        """Initialize SQLite schema for BM25 index."""
        cursor = self.conn.cursor()

        # Tokens table: stores global token statistics
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tokens (
                token TEXT PRIMARY KEY,
                doc_freq INTEGER DEFAULT 0
            )
        """
        )

        # Posting lists: token -> file mappings with frequencies
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS posting_lists (
                token TEXT,
                file_path TEXT,
                term_freq INTEGER,
                weight REAL DEFAULT 1.0,
                PRIMARY KEY (token, file_path),
                FOREIGN KEY (token) REFERENCES tokens(token)
            )
        """
        )

        # Corpus statistics
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS corpus_stats (
                key TEXT PRIMARY KEY,
                value REAL
            )
        """
        )

        # File metadata
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS file_metadata (
                file_path TEXT PRIMARY KEY,
                doc_length INTEGER,
                last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create indexes for fast lookups
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_posting_token
            ON posting_lists(token)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_posting_file
            ON posting_lists(file_path)
        """
        )

        self.conn.commit()
        logger.info(f"BM25 index initialized at {self.db_path}")

    def _refresh_stats_cache(self):
        """Refresh cached corpus statistics."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT key, value FROM corpus_stats")
        self._stats_cache = dict(cursor.fetchall())

        # Initialize defaults if not present
        if "num_docs" not in self._stats_cache:
            self._stats_cache["num_docs"] = 0.0
        if "avg_doc_length" not in self._stats_cache:
            self._stats_cache["avg_doc_length"] = 0.0

    def tokenize_code(
        self, content: str, language: str = "python"
    ) -> Dict[str, Tuple[int, float]]:
        """
        Smart tokenization for code files.

        Extracts and weights tokens based on their role in the code:
        - Class names (weight 5.0)
        - Function/method names (weight 4.0)
        - Import module names (weight 3.0)
        - Variable names (weight 1.5)
        - Comments (weight 0.5)

        Args:
            content: Source code content
            language: Programming language (default: "python")

        Returns:
            Dict mapping token -> (frequency, weight)
        """
        tokens = defaultdict(lambda: [0, 0.0])  # [count, max_weight]

        if language == "python":
            tokens_dict = self._tokenize_python(content)
        elif language in ["javascript", "typescript", "jsx", "tsx"]:
            tokens_dict = self._tokenize_javascript(content)
        else:
            # Fallback: generic tokenization
            tokens_dict = self._tokenize_generic(content)

        return tokens_dict

    def _tokenize_python(self, content: str) -> Dict[str, Tuple[int, float]]:
        """Tokenize Python code using AST."""
        tokens = defaultdict(lambda: [0, 0.0])

        try:
            tree = ast.parse(content)

            # Extract class names
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    token = node.name.lower()  # Normalize to lowercase
                    tokens[token][0] += 1
                    tokens[token][1] = max(tokens[token][1], self.WEIGHTS["class"])

                # Extract function/method names
                elif isinstance(node, ast.FunctionDef) or isinstance(
                    node, ast.AsyncFunctionDef
                ):
                    token = node.name.lower()  # Normalize to lowercase
                    tokens[token][0] += 1
                    tokens[token][1] = max(tokens[token][1], self.WEIGHTS["function"])

                # Extract import names
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        token = alias.name.split(".")[
                            0
                        ].lower()  # Normalize to lowercase
                        tokens[token][0] += 1
                        tokens[token][1] = max(tokens[token][1], self.WEIGHTS["import"])

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        token = node.module.split(".")[
                            0
                        ].lower()  # Normalize to lowercase
                        tokens[token][0] += 1
                        tokens[token][1] = max(tokens[token][1], self.WEIGHTS["import"])

                # Extract variable names (assignments)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    token = node.id.lower()  # Normalize to lowercase
                    # Skip private variables (starts with _)
                    if not token.startswith("_"):
                        tokens[token][0] += 1
                        tokens[token][1] = max(
                            tokens[token][1], self.WEIGHTS["variable"]
                        )

        except SyntaxError as e:
            logger.warning(f"Failed to parse Python code: {e}")
            # Fallback to regex-based extraction
            return self._tokenize_generic(content)

        # Extract comments
        comment_tokens = self._extract_comments(content, "#")
        for token, count in comment_tokens.items():
            tokens[token][0] += count
            tokens[token][1] = max(tokens[token][1], self.WEIGHTS["comment"])

        # Convert to tuple format
        return {token: (counts[0], counts[1]) for token, counts in tokens.items()}

    def _tokenize_javascript(self, content: str) -> Dict[str, Tuple[int, float]]:
        """Tokenize JavaScript/TypeScript code using regex patterns."""
        tokens = defaultdict(lambda: [0, 0.0])

        # Class declarations: class ClassName
        for match in re.finditer(r"\bclass\s+([A-Z][a-zA-Z0-9_]*)", content):
            token = match.group(1).lower()  # Normalize to lowercase
            tokens[token][0] += 1
            tokens[token][1] = max(tokens[token][1], self.WEIGHTS["class"])

        # Function declarations: function functionName, const functionName =
        for match in re.finditer(r"\bfunction\s+([a-zA-Z_][a-zA-Z0-9_]*)", content):
            token = match.group(1).lower()  # Normalize to lowercase
            tokens[token][0] += 1
            tokens[token][1] = max(tokens[token][1], self.WEIGHTS["function"])

        for match in re.finditer(
            r"\bconst\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s+)?(?:function|\()",
            content,
        ):
            token = match.group(1).lower()  # Normalize to lowercase
            tokens[token][0] += 1
            tokens[token][1] = max(tokens[token][1], self.WEIGHTS["function"])

        # Class methods: methodName(args) { or async methodName(args) {
        for match in re.finditer(
            r"(?:async\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{",
            content,
        ):
            token = match.group(1).lower()  # Normalize to lowercase
            # Skip constructor, if, while, for, switch (keywords)
            if token not in {
                "constructor",
                "if",
                "while",
                "for",
                "switch",
                "catch",
                "function",
            }:
                tokens[token][0] += 1
                tokens[token][1] = max(tokens[token][1], self.WEIGHTS["function"])

        # Arrow function properties: methodName = (args) =>
        for match in re.finditer(
            r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>",
            content,
        ):
            token = match.group(1).lower()  # Normalize to lowercase
            tokens[token][0] += 1
            tokens[token][1] = max(tokens[token][1], self.WEIGHTS["function"])

        # Import statements: import ... from 'module'
        for match in re.finditer(r"import\s+.*?from\s+['\"]([^'\"]+)['\"]", content):
            module = match.group(1).split("/")[0].lower()  # Normalize to lowercase
            tokens[module][0] += 1
            tokens[module][1] = max(tokens[module][1], self.WEIGHTS["import"])

        # Variable declarations: const, let, var
        for match in re.finditer(
            r"\b(?:const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)", content
        ):
            token = match.group(1).lower()  # Normalize to lowercase
            if not token.startswith("_"):
                tokens[token][0] += 1
                tokens[token][1] = max(tokens[token][1], self.WEIGHTS["variable"])

        # Extract comments
        comment_tokens = self._extract_comments(content, "//")
        for token, count in comment_tokens.items():
            tokens[token][0] += count
            tokens[token][1] = max(tokens[token][1], self.WEIGHTS["comment"])

        return {token: (counts[0], counts[1]) for token, counts in tokens.items()}

    def _tokenize_generic(self, content: str) -> Dict[str, Tuple[int, float]]:
        """Generic tokenization for any text using word boundaries."""
        tokens = defaultdict(lambda: [0, 0.0])

        # Extract words (alphanumeric + underscore)
        for match in re.finditer(r"\b([a-zA-Z_][a-zA-Z0-9_]{2,})\b", content):
            token = match.group(1).lower()
            # Skip common stop words
            if token not in {
                "the",
                "and",
                "for",
                "are",
                "but",
                "not",
                "you",
                "this",
                "that",
                "with",
                "from",
            }:
                tokens[token][0] += 1
                tokens[token][1] = max(tokens[token][1], self.WEIGHTS["keyword"])

        return {token: (counts[0], counts[1]) for token, counts in tokens.items()}

    def _extract_comments(self, content: str, comment_char: str) -> Dict[str, int]:
        """Extract tokens from comments."""
        tokens = Counter()

        # Find all comment lines
        for line in content.split("\n"):
            if comment_char in line:
                comment = line.split(comment_char, 1)[1]
                # Extract words from comment
                words = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]{2,})\b", comment)
                tokens.update(word.lower() for word in words)

        return dict(tokens)

    def index_file(self, file_path: str, content: str, language: str = "python"):
        """
        Index a file for BM25 search.

        Extracts tokens, calculates weights, and stores in index.

        Args:
            file_path: Path to the file
            content: File content
            language: Programming language
        """
        # Remove existing index for this file
        self.remove_file(file_path)

        # Tokenize the content
        tokens = self.tokenize_code(content, language)

        if not tokens:
            logger.warning(f"No tokens extracted from {file_path}")
            return

        # Calculate document length (sum of term frequencies)
        doc_length = sum(freq for freq, _ in tokens.values())

        cursor = self.conn.cursor()

        # Add file metadata
        cursor.execute(
            """
            INSERT INTO file_metadata (file_path, doc_length)
            VALUES (?, ?)
        """,
            (file_path, doc_length),
        )

        # Add tokens and posting lists
        for token, (term_freq, weight) in tokens.items():
            # Insert or ignore token in tokens table
            cursor.execute(
                """
                INSERT OR IGNORE INTO tokens (token, doc_freq)
                VALUES (?, 0)
            """,
                (token,),
            )

            # Update document frequency
            cursor.execute(
                """
                UPDATE tokens
                SET doc_freq = doc_freq + 1
                WHERE token = ?
            """,
                (token,),
            )

            # Add to posting list
            cursor.execute(
                """
                INSERT INTO posting_lists (token, file_path, term_freq, weight)
                VALUES (?, ?, ?, ?)
            """,
                (token, file_path, term_freq, weight),
            )

        self.conn.commit()

        # Update corpus statistics
        self.update_statistics()

        logger.debug(
            f"Indexed {file_path}: {len(tokens)} tokens, doc_length={doc_length}"
        )

    def remove_file(self, file_path: str):
        """
        Remove a file from the index.

        Args:
            file_path: Path to the file to remove
        """
        cursor = self.conn.cursor()

        # Get tokens for this file
        cursor.execute(
            """
            SELECT token FROM posting_lists WHERE file_path = ?
        """,
            (file_path,),
        )
        tokens = [row[0] for row in cursor.fetchall()]

        # Remove from posting lists
        cursor.execute(
            """
            DELETE FROM posting_lists WHERE file_path = ?
        """,
            (file_path,),
        )

        # Update document frequencies
        for token in tokens:
            cursor.execute(
                """
                UPDATE tokens
                SET doc_freq = doc_freq - 1
                WHERE token = ?
            """,
                (token,),
            )

        # Remove tokens with zero doc_freq
        cursor.execute(
            """
            DELETE FROM tokens WHERE doc_freq <= 0
        """
        )

        # Remove file metadata
        cursor.execute(
            """
            DELETE FROM file_metadata WHERE file_path = ?
        """,
            (file_path,),
        )

        self.conn.commit()

        # Update corpus statistics
        self.update_statistics()

        logger.debug(f"Removed {file_path} from index")

    def update_statistics(self):
        """Update corpus statistics (num_docs, avg_doc_length)."""
        cursor = self.conn.cursor()

        # Calculate number of documents
        cursor.execute("SELECT COUNT(*) FROM file_metadata")
        num_docs = cursor.fetchone()[0]

        # Calculate average document length
        cursor.execute("SELECT AVG(doc_length) FROM file_metadata")
        result = cursor.fetchone()[0]
        avg_doc_length = result if result else 0.0

        # Update stats table
        cursor.execute(
            """
            INSERT OR REPLACE INTO corpus_stats (key, value)
            VALUES ('num_docs', ?), ('avg_doc_length', ?)
        """,
            (float(num_docs), float(avg_doc_length)),
        )

        self.conn.commit()

        # Refresh cache
        self._refresh_stats_cache()

        logger.debug(
            f"Updated stats: num_docs={num_docs}, avg_doc_length={avg_doc_length:.2f}"
        )

    def calculate_idf(self, token: str) -> float:
        """
        Calculate IDF (Inverse Document Frequency) for a token.

        IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5))

        Args:
            token: Token to calculate IDF for

        Returns:
            IDF score
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT doc_freq FROM tokens WHERE token = ?", (token,))
        result = cursor.fetchone()

        if not result:
            return 0.0

        doc_freq = result[0]
        num_docs = self._stats_cache.get("num_docs", 0.0)

        if num_docs == 0:
            return 0.0

        idf = math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
        return max(0.0, idf)

    def search(self, query: str, limit: int = 60) -> List[BM25SearchResult]:
        """
        Search for files using BM25 ranking.

        Args:
            query: Search query (space-separated tokens)
            limit: Maximum number of results to return

        Returns:
            List of BM25SearchResult objects, sorted by score (descending)
        """
        # Tokenize query
        query_tokens = self._tokenize_query(query)

        if not query_tokens:
            logger.warning("No valid query tokens")
            return []

        # Get statistics
        num_docs = self._stats_cache.get("num_docs", 0.0)
        avg_doc_length = self._stats_cache.get("avg_doc_length", 1.0)

        if num_docs == 0:
            logger.warning("Index is empty")
            return []

        # Calculate BM25 scores for each file
        file_scores = defaultdict(lambda: {"score": 0.0, "tokens": {}})

        cursor = self.conn.cursor()

        for token in query_tokens:
            idf = self.calculate_idf(token)

            if idf == 0:
                continue

            # Get posting list for this token
            cursor.execute(
                """
                SELECT pl.file_path, pl.term_freq, pl.weight, fm.doc_length
                FROM posting_lists pl
                JOIN file_metadata fm ON pl.file_path = fm.file_path
                WHERE pl.token = ?
            """,
                (token,),
            )

            for file_path, term_freq, weight, doc_length in cursor.fetchall():
                # BM25 score formula
                # score = IDF * (f * (k1 + 1)) / (f + k1 * (1 - b + b * |D| / avgdl))

                # Apply weight to term frequency
                weighted_freq = term_freq * weight

                # Length normalization
                norm = 1 - self.B + self.B * (doc_length / avg_doc_length)

                # BM25 score for this token
                token_score = (
                    idf
                    * (weighted_freq * (self.K1 + 1))
                    / (weighted_freq + self.K1 * norm)
                )

                file_scores[file_path]["score"] += token_score
                file_scores[file_path]["tokens"][token] = token_score

        # Convert to results and sort
        results = [
            BM25SearchResult(
                file_path=file_path, score=data["score"], matched_tokens=data["tokens"]
            )
            for file_path, data in file_scores.items()
        ]

        results.sort(key=lambda x: x.score, reverse=True)

        return results[:limit]

    def _tokenize_query(self, query: str) -> List[str]:
        """
        Tokenize search query.

        Extracts words and normalizes to lowercase.

        Args:
            query: Search query

        Returns:
            List of query tokens
        """
        # Extract words (alphanumeric + underscore)
        tokens = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]+)\b", query.lower())

        # Remove stop words
        stop_words = {
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "this",
            "that",
            "with",
            "from",
        }
        tokens = [t for t in tokens if t not in stop_words]

        return tokens

    def get_top_tokens(self, file_path: str, limit: int = 20) -> Dict[str, float]:
        """
        Get top tokens for a file (for storing in FileTriIndex).

        Returns tokens sorted by TF-IDF score.

        Args:
            file_path: Path to the file
            limit: Maximum number of tokens to return

        Returns:
            Dict mapping token -> TF-IDF score
        """
        cursor = self.conn.cursor()

        # Get all tokens for this file
        cursor.execute(
            """
            SELECT token, term_freq, weight
            FROM posting_lists
            WHERE file_path = ?
        """,
            (file_path,),
        )

        results = cursor.fetchall()

        if not results:
            return {}

        # Calculate TF-IDF for each token
        token_scores = {}
        for token, term_freq, weight in results:
            idf = self.calculate_idf(token)
            tf = term_freq * weight  # Apply weight
            tfidf = tf * idf
            token_scores[token] = tfidf

        # Sort by score and return top-N
        sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)

        return dict(sorted_tokens[:limit])

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("BM25 index closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
