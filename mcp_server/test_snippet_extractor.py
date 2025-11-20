"""
Unit tests for SOTA snippet extractor

Tests cover:
- Query-aware relevance scoring
- Code block detection and preservation
- Sentence boundary detection
- Multi-segment extraction
- Smart truncation without query
"""

import pytest
from snippet_extractor import SnippetExtractor, extract_snippet


class TestSnippetExtractor:
    """Test suite for SnippetExtractor"""

    def test_short_content_returned_as_is(self):
        """Short content should be returned without modification"""
        content = "This is a short piece of text."
        result = extract_snippet(content, query="")
        assert result == content

    def test_simple_truncation_without_query(self):
        """Long content without query should use smart truncation"""
        content = "This is a very long piece of text. " * 50
        result = extract_snippet(content, query="", max_length=100)

        assert len(result) <= 103  # 100 + "..."
        assert result.endswith("...")
        assert "This is a very long piece of text." in result

    def test_query_aware_extraction(self):
        """Should extract portion containing query terms"""
        content = (
            "The beginning has nothing relevant. " * 5
            + "Authentication is handled by the auth module using JWT tokens. "
            + "The end has nothing relevant. " * 5
        )

        result = extract_snippet(content, query="authentication JWT", max_length=200)

        # Should contain the relevant portion
        assert "authentication" in result.lower()
        assert "jwt" in result.lower()
        # Should NOT start with irrelevant beginning
        assert not result.startswith("The beginning")

    def test_code_block_extraction(self):
        """Should preserve code block boundaries"""
        content = """
Irrelevant text here.

def authenticate(username, password):
    '''Authenticate user with credentials'''
    if not username or not password:
        return False
    return verify_credentials(username, password)

More irrelevant text.

def other_function():
    pass
"""

        result = extract_snippet(content, query="authenticate", max_length=300)

        # Should contain the authenticate function
        assert "def authenticate" in result
        assert "username" in result
        assert "password" in result
        # Should NOT include other_function if space limited
        assert "other_function" not in result or len(result) <= 300

    def test_multi_segment_extraction(self):
        """Should extract multiple relevant segments with ellipsis"""
        content = (
            """
First relevant section about authentication using JWT tokens.

"""
            + "Irrelevant content. " * 20
            + """

Second relevant section about token validation and authentication flow.
"""
        )

        extractor = SnippetExtractor(max_length=200)
        result = extractor.extract(content, query="authentication token")

        # Should contain parts from both sections
        assert "authentication" in result.lower()
        assert "token" in result.lower()
        # Should have ellipsis separating segments
        if "First" in result and "Second" in result:
            assert "..." in result

    def test_sentence_boundary_preservation(self):
        """Should respect sentence boundaries when truncating"""
        content = (
            "This is the first sentence. "
            "This is the second sentence. "
            "This is the third sentence. "
            "This is the fourth sentence."
        )

        result = extract_snippet(content, query="", max_length=60)

        # Should end at sentence boundary, not mid-sentence
        if "..." in result:
            text_before_ellipsis = result.replace("...", "").strip()
            assert text_before_ellipsis.endswith(".")

    def test_code_block_bracket_matching(self):
        """Should match brackets to find complete code blocks"""
        content = '''
def process_data(items):
    """Process items with nested structures"""
    results = []
    for item in items:
        if item.get("valid"):
            results.append({
                "id": item["id"],
                "data": item["data"]
            })
    return results

def other_function():
    pass
'''

        result = extract_snippet(content, query="process_data", max_length=400)

        # Should contain complete function with all brackets matched
        assert "def process_data" in result
        assert "return results" in result
        # Verify brackets are balanced
        assert result.count("{") == result.count("}")
        assert result.count("[") == result.count("]")
        assert result.count("(") == result.count(")")

    def test_relevance_scoring_with_multiple_terms(self):
        """Should score segments based on multiple query terms"""
        content = """
Section A: Basic user authentication.

Section B: Advanced authentication with JWT tokens and session management.

Section C: Database configuration settings.
"""

        extractor = SnippetExtractor(max_length=200)
        result = extractor.extract(content, query="authentication JWT session")

        # Section B should be preferred (has all 3 terms)
        assert "Section B" in result or "Advanced" in result
        assert "jwt" in result.lower()

    def test_stop_word_removal(self):
        """Should ignore stop words in query"""
        content = (
            "Irrelevant beginning. " * 3
            + "The authentication system is important. "
            + "Irrelevant end. " * 3
        )

        result = extract_snippet(
            content, query="the authentication is important", max_length=200
        )

        # Should find content based on "authentication important" (stop words removed)
        assert "authentication" in result.lower()
        assert "important" in result.lower()

    def test_empty_content(self):
        """Should handle empty content gracefully"""
        result = extract_snippet("", query="test")
        assert result == ""

    def test_custom_max_length(self):
        """Should respect custom max_length parameter"""
        content = "Word " * 100

        result50 = extract_snippet(content, query="", max_length=50)
        result100 = extract_snippet(content, query="", max_length=100)
        result200 = extract_snippet(content, query="", max_length=200)

        assert len(result50) <= 53  # 50 + "..."
        assert len(result100) <= 103
        assert len(result200) <= 203

    def test_code_bonus_scoring(self):
        """Code blocks should get bonus in relevance scoring"""
        content = """
Regular text mentioning authentication here.

def authenticate_user(username):
    '''Code block also mentioning authentication'''
    return True
"""

        extractor = SnippetExtractor(max_length=150)
        result = extractor.extract(content, query="authentication")

        # Code block should be preferred due to bonus
        assert "def authenticate_user" in result

    def test_position_bonus_scoring(self):
        """Earlier positions should get bonus in scoring"""
        content = (
            "Early mention of the search term. " * 2
            + "Irrelevant content. " * 20
            + "Late mention of the search term. " * 2
        )

        extractor = SnippetExtractor(max_length=100)
        result = extractor.extract(content, query="search term")

        # Earlier mention should be preferred
        assert "Early mention" in result

    def test_paragraph_segmentation(self):
        """Should segment by paragraphs"""
        content = """
First paragraph with some content about databases.

Second paragraph with content about authentication.

Third paragraph with content about caching.
"""

        extractor = SnippetExtractor(max_length=200)
        result = extractor.extract(content, query="authentication")

        # Should extract the relevant paragraph
        assert "authentication" in result.lower()
        assert "Second paragraph" in result or "authentication" in result.lower()

    def test_no_overlap_in_segments(self):
        """Selected segments should not overlap"""
        content = "A" * 100 + "RELEVANT" + "B" * 100 + "RELEVANT" + "C" * 100

        extractor = SnippetExtractor(max_length=500)
        segments = extractor._segment_content(content)
        scored = extractor._score_segments(segments, ["relevant"])
        selected = extractor._select_segments(scored)

        # Verify no overlaps
        for i, seg1 in enumerate(selected):
            for seg2 in selected[i + 1 :]:
                assert not (
                    seg1.start_pos <= seg2.start_pos < seg1.end_pos
                    or seg1.start_pos < seg2.end_pos <= seg1.end_pos
                )

    def test_typescript_code_extraction(self):
        """Should handle TypeScript/JavaScript code"""
        content = """
export const authenticateUser = async (credentials: UserCredentials) => {
    const token = await generateToken(credentials);
    return {
        success: true,
        token
    };
};

const otherFunction = () => {};
"""

        result = extract_snippet(content, query="authenticate", max_length=300)

        # Should extract the relevant function
        assert "authenticateUser" in result
        assert "credentials" in result

    def test_mixed_content(self):
        """Should handle mixed text and code content"""
        content = """
This is documentation about authentication.

The authentication flow works as follows:

```python
def authenticate(user):
    return verify_token(user.token)
```

And here is more explanation.
"""

        result = extract_snippet(content, query="authentication", max_length=200)

        # Should include relevant portions
        assert "authentication" in result.lower()


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_very_long_query(self):
        """Should handle very long queries"""
        content = "Simple content about authentication."
        long_query = "authentication " * 100

        result = extract_snippet(content, query=long_query, max_length=100)
        assert "authentication" in result.lower()

    def test_special_characters_in_query(self):
        """Should handle special characters in query"""
        content = "Content about user.authenticate() method."

        result = extract_snippet(content, query="user.authenticate()", max_length=100)
        assert "authenticate" in result.lower()

    def test_unicode_content(self):
        """Should handle Unicode content"""
        content = "Content with émojis 🔒 and spëcial çharacters."

        result = extract_snippet(content, query="émojis", max_length=100)
        assert "émojis" in result or "mojis" in result

    def test_very_long_single_line(self):
        """Should handle content with very long lines"""
        content = "A" * 10000  # Single 10K character line

        result = extract_snippet(content, query="", max_length=200)
        assert len(result) <= 203
        assert result.endswith("...")

    def test_no_sentence_boundaries(self):
        """Should handle content with no sentence boundaries"""
        content = "word " * 1000  # No periods or boundaries

        result = extract_snippet(content, query="", max_length=100)
        assert len(result) <= 103


def test_integration_with_real_code():
    """Integration test with real-world code snippet"""
    real_code = '''
class QdrantVectorStore:
    """
    Qdrant-based vector store for semantic search

    Compatible with RealFAISSIndex interface:
    - add_document(content, importance)
    - search(query, k)
    """

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.collection_name = "omnimemory_embeddings"
        self.client = QdrantClient(url="http://localhost:6333")

    async def search(self, query: str, k: int = 5):
        """Search for similar documents"""
        query_embedding = await self._get_embedding(query)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k
        )
        return results
'''

    result = extract_snippet(real_code, query="search vector semantic", max_length=300)

    # Should extract relevant parts
    assert "search" in result.lower()
    assert "QdrantVectorStore" in result or "semantic" in result.lower()
    assert len(result) <= 303


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
