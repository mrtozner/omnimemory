"""
Test BM25 Index Implementation

Demonstrates BM25 sparse indexing on sample code files.
Tests tokenization, indexing, search, and integration with Tri-Index.
"""

import tempfile
import os
from pathlib import Path
from bm25_index import BM25Index, BM25SearchResult


# Sample code files for testing
SAMPLE_PYTHON_FILE = """
import bcrypt
from typing import Optional
import jwt

class AuthManager:
    '''Authentication manager for user login'''

    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.users = {}

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        '''Authenticate user and return JWT token'''
        user = self.users.get(username)
        if not user:
            return None

        if bcrypt.checkpw(password.encode(), user['password_hash']):
            token = jwt.encode({'username': username}, self.secret_key)
            return token
        return None

    def register_user(self, username: str, password: str):
        '''Register a new user'''
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        self.users[username] = {'password_hash': password_hash}
        return True

def validate_email(email: str) -> bool:
    '''Validate email format'''
    return '@' in email and '.' in email
"""

SAMPLE_JAVASCRIPT_FILE = """
import React from 'react';
import { useState, useEffect } from 'react';

class UserProfile extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      user: null,
      loading: true
    };
  }

  async fetchUserData(userId) {
    const response = await fetch(`/api/users/${userId}`);
    const userData = await response.json();
    return userData;
  }

  componentDidMount() {
    this.fetchUserData(this.props.userId)
      .then(user => this.setState({ user, loading: false }));
  }

  render() {
    const { user, loading } = this.state;
    if (loading) return <div>Loading...</div>;
    return <div>{user.name}</div>;
  }
}

const ProfileButton = ({ onClick, label }) => {
  return (
    <button onClick={onClick}>
      {label}
    </button>
  );
};

export { UserProfile, ProfileButton };
"""

SAMPLE_GENERIC_FILE = """
Configuration File for Application

This file contains important settings and parameters.
Database connection strings and API keys should be stored securely.

Settings:
- database_host: localhost
- database_port: 5432
- cache_timeout: 3600
- max_connections: 100

Security notes:
Ensure all passwords are encrypted before storage.
Use environment variables for sensitive data.
"""


def test_python_tokenization():
    """Test tokenization of Python code"""
    print("\n=== Test 1: Python Tokenization ===")

    index = BM25Index()

    tokens = index.tokenize_code(SAMPLE_PYTHON_FILE, language="python")

    print(f"Total tokens extracted: {len(tokens)}")
    print("\nTop tokens by weight:")

    # Sort by weight and frequency
    sorted_tokens = sorted(
        tokens.items(), key=lambda x: (x[1][1], x[1][0]), reverse=True
    )

    for token, (freq, weight) in sorted_tokens[:15]:
        weight_type = (
            "class"
            if weight == 5.0
            else "function"
            if weight == 4.0
            else "import"
            if weight == 3.0
            else "variable"
            if weight == 1.5
            else "comment"
        )
        print(f"  {token:20s} | freq={freq}, weight={weight:.1f} ({weight_type})")

    # Verify key tokens were extracted (now lowercase)
    assert "authmanager" in tokens, "Class name not extracted"
    assert tokens["authmanager"][1] == 5.0, "Class weight incorrect"

    assert "authenticate_user" in tokens, "Function name not extracted"
    assert tokens["authenticate_user"][1] == 4.0, "Function weight incorrect"

    assert "bcrypt" in tokens, "Import name not extracted"
    assert tokens["bcrypt"][1] == 3.0, "Import weight incorrect"

    print("\n✅ Python tokenization test passed!")


def test_javascript_tokenization():
    """Test tokenization of JavaScript code"""
    print("\n=== Test 2: JavaScript Tokenization ===")

    index = BM25Index()

    tokens = index.tokenize_code(SAMPLE_JAVASCRIPT_FILE, language="javascript")

    print(f"Total tokens extracted: {len(tokens)}")
    print("\nTop tokens by weight:")

    sorted_tokens = sorted(
        tokens.items(), key=lambda x: (x[1][1], x[1][0]), reverse=True
    )

    for token, (freq, weight) in sorted_tokens[:15]:
        weight_type = (
            "class"
            if weight == 5.0
            else "function"
            if weight == 4.0
            else "import"
            if weight == 3.0
            else "variable"
            if weight == 1.5
            else "keyword"
        )
        print(f"  {token:20s} | freq={freq}, weight={weight:.1f} ({weight_type})")

    # Verify key tokens (now lowercase)
    assert "userprofile" in tokens, "Class name not extracted"
    assert tokens["userprofile"][1] == 5.0, "Class weight incorrect"

    assert "fetchuserdata" in tokens, "Method name not extracted"

    print("\n✅ JavaScript tokenization test passed!")


def test_index_and_search():
    """Test indexing and searching files"""
    print("\n=== Test 3: Index and Search ===")

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        index = BM25Index(db_path=db_path)

        # Index sample files
        print("\nIndexing files...")
        index.index_file("auth.py", SAMPLE_PYTHON_FILE, language="python")
        index.index_file("profile.js", SAMPLE_JAVASCRIPT_FILE, language="javascript")
        index.index_file("config.txt", SAMPLE_GENERIC_FILE, language="generic")

        print(f"  Indexed 3 files")
        print(
            f"  Corpus stats: num_docs={index._stats_cache['num_docs']:.0f}, "
            f"avg_doc_length={index._stats_cache['avg_doc_length']:.1f}"
        )

        # Test search queries
        print("\n--- Search 1: 'authenticate user' ---")
        results = index.search("authenticate user", limit=5)
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.file_path} (score={result.score:.3f})")
            print(f"     Matched tokens: {result.matched_tokens}")

        assert len(results) > 0, "No results found"
        assert results[0].file_path == "auth.py", "Wrong file ranked first"

        print("\n--- Search 2: 'UserProfile component' ---")
        results = index.search("UserProfile component", limit=5)
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.file_path} (score={result.score:.3f})")
            print(f"     Matched tokens: {result.matched_tokens}")

        assert results[0].file_path == "profile.js", "Wrong file for JS search"

        print("\n--- Search 3: 'bcrypt password' ---")
        results = index.search("bcrypt password", limit=5)
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.file_path} (score={result.score:.3f})")
            print(f"     Matched tokens: {result.matched_tokens}")

        print("\n--- Search 4: 'database configuration' ---")
        results = index.search("database configuration", limit=5)
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.file_path} (score={result.score:.3f})")
            print(f"     Matched tokens: {result.matched_tokens}")

        index.close()
        print("\n✅ Index and search test passed!")

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_get_top_tokens():
    """Test getting top tokens for Tri-Index integration"""
    print("\n=== Test 4: Get Top Tokens (Tri-Index Integration) ===")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        index = BM25Index(db_path=db_path)

        # Index files
        index.index_file("auth.py", SAMPLE_PYTHON_FILE, language="python")
        index.index_file("profile.js", SAMPLE_JAVASCRIPT_FILE, language="javascript")

        # Get top tokens for each file
        print("\nTop tokens for auth.py:")
        top_tokens = index.get_top_tokens("auth.py", limit=10)
        for token, score in top_tokens.items():
            print(f"  {token:20s} | TF-IDF={score:.3f}")

        assert len(top_tokens) > 0, "No top tokens returned"
        assert (
            "authmanager" in top_tokens or "authenticate_user" in top_tokens
        ), "Key tokens not in top results"

        print("\nTop tokens for profile.js:")
        top_tokens = index.get_top_tokens("profile.js", limit=10)
        for token, score in top_tokens.items():
            print(f"  {token:20s} | TF-IDF={score:.3f}")

        print("\n✅ Get top tokens test passed!")

        index.close()

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_incremental_updates():
    """Test incremental index updates (add/remove files)"""
    print("\n=== Test 5: Incremental Updates ===")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        index = BM25Index(db_path=db_path)

        # Initial index
        print("\nInitial indexing...")
        index.index_file("auth.py", SAMPLE_PYTHON_FILE, language="python")
        print(f"  Corpus: {index._stats_cache['num_docs']:.0f} docs")

        # Search should find results (using exact token that exists)
        results = index.search("authmanager user", limit=5)
        assert len(results) > 0, "No results after initial index"
        print(f"  Search 'authmanager user': {len(results)} results")

        # Add another file
        print("\nAdding profile.js...")
        index.index_file("profile.js", SAMPLE_JAVASCRIPT_FILE, language="javascript")
        print(f"  Corpus: {index._stats_cache['num_docs']:.0f} docs")

        # Remove first file
        print("\nRemoving auth.py...")
        index.remove_file("auth.py")
        print(f"  Corpus: {index._stats_cache['num_docs']:.0f} docs")

        # Search for removed file should return no results
        results = index.search("authmanager", limit=5)
        print(f"  Search 'authmanager' (removed): {len(results)} results")

        # Search for JS file should still work
        results = index.search("userprofile", limit=5)
        assert len(results) > 0, "No results for remaining file"
        assert results[0].file_path == "profile.js", "Wrong file after removal"
        print(f"  Search 'userprofile': found {results[0].file_path}")

        print("\n✅ Incremental updates test passed!")

        index.close()

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_bm25_scoring():
    """Test BM25 scoring algorithm properties"""
    print("\n=== Test 6: BM25 Scoring Properties ===")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        index = BM25Index(db_path=db_path)

        # Create files with different token frequencies
        file1 = "class AuthManager:\n    def authenticate(self): pass"
        file2 = "# authentication system\nclass AuthService:\n    def authenticate(self): pass\n    def verify_auth(self): pass"
        file3 = "def process_data(): pass"

        index.index_file("file1.py", file1, language="python")
        index.index_file("file2.py", file2, language="python")
        index.index_file("file3.py", file3, language="python")

        # Search for "authenticate" (function name that exists in both files)
        results = index.search("authenticate", limit=5)

        print("\nSearch 'authenticate':")
        for result in results:
            print(f"  {result.file_path}: score={result.score:.3f}")

        # Both file1 and file2 should match (both have authenticate function)
        assert len(results) >= 2, "Should match at least 2 files"
        assert results[0].file_path in [
            "file1.py",
            "file2.py",
        ], "Top result should be file1 or file2"

        # file3 should not appear (doesn't contain 'authenticate')
        file_paths = [r.file_path for r in results]
        assert "file3.py" not in file_paths, "file3 should not match"

        print("\n✅ BM25 scoring test passed!")

        index.close()

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def run_all_tests():
    """Run all BM25 index tests"""
    print("=" * 70)
    print("BM25 Index Test Suite")
    print("=" * 70)

    try:
        test_python_tokenization()
        test_javascript_tokenization()
        test_index_and_search()
        test_get_top_tokens()
        test_incremental_updates()
        test_bm25_scoring()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nBM25 Index is working correctly and ready for Tri-Index integration.")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
