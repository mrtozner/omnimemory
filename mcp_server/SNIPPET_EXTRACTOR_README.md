# SOTA Snippet Extractor

**State-of-the-art snippet extraction for search results with query-aware relevance scoring**

## Overview

Replaces simple truncation (`content[:200] + "..."`) with intelligent, context-aware snippet extraction using modern NLP and code parsing techniques.

## Features

### 1. Query-Aware Relevance Scoring (BM25-inspired)
- Extracts portions containing query terms, not just first 200 characters
- Uses BM25-inspired algorithm for term frequency and document frequency scoring
- Removes stop words for better matching
- Supports multi-term queries

**Example:**
```python
content = "Irrelevant intro... Authentication uses JWT tokens... Irrelevant end..."
snippet = extract_snippet(content, query="authentication JWT", max_length=200)
# Result: "Authentication uses JWT tokens..." (skips irrelevant parts)
```

### 2. Code Block Detection & Preservation
- Detects function and class definitions (Python, JavaScript, TypeScript, Go)
- Matches brackets to find complete code blocks
- Preserves syntax structure (doesn't cut mid-function)

**Example:**
```python
content = '''
def calculate(): pass

def authenticate(user, password):
    """Authenticate user"""
    token = generate_jwt(user)
    return token

def send_email(): pass
'''

snippet = extract_snippet(content, query="authenticate", max_length=300)
# Result: Complete authenticate() function, not partial
```

### 3. Multi-Segment Extraction
- Extracts multiple relevant portions from different parts of the document
- Combines segments with `" ... "` separators
- Prioritizes highest-scored segments within length limit

**Example:**
```python
content = "Intro about auth... [500 chars of irrelevant]... JWT tokens... [500 chars]... token expiration..."
snippet = extract_snippet(content, query="auth JWT token expiration", max_length=250)
# Result: "Intro about auth ... JWT tokens ... token expiration"
```

### 4. Smart Boundary Detection
- Respects sentence boundaries (doesn't cut mid-sentence)
- Detects paragraph breaks
- Finds code block boundaries
- Prevents cutting inside brackets/parentheses

### 5. Bonus Scoring
- **Code blocks**: 1.3x multiplier (code often more relevant than docs)
- **Position**: Earlier content gets slight bonus (0.8-1.0x)
- **Length normalization**: Prevents bias toward long/short segments

## Architecture

```
Content Input
    ↓
Segmentation (by paragraphs → sentences → chunks)
    ↓
Query Term Extraction (remove stop words)
    ↓
Segment Scoring (BM25 + bonuses)
    ↓
Segment Selection (highest scored, non-overlapping)
    ↓
Snippet Building (combine with ellipsis)
    ↓
Final Snippet
```

## Usage

### Basic Usage

```python
from snippet_extractor import extract_snippet

# Query-aware extraction
snippet = extract_snippet(
    content=long_document,
    query="authentication JWT",
    max_length=300
)

# Smart truncation (no query)
snippet = extract_snippet(
    content=long_document,
    max_length=300
)
```

### Advanced Usage

```python
from snippet_extractor import SnippetExtractor

# Custom configuration
extractor = SnippetExtractor(
    max_length=500,        # Maximum snippet length
    context_chars=100,     # Context around matches
    min_segment_length=40  # Minimum segment size
)

snippet = extractor.extract(content, query)
```

## Integration with Qdrant Vector Store

**Before:**
```python
# Simple truncation (qdrant_vector_store.py:203)
snippet = content[:200] + "..." if len(content) > 200 else content
```

**After:**
```python
# SOTA snippet extraction with query-aware relevance scoring
snippet = extract_snippet(
    content=content,
    query=query,  # Use search query for relevance
    max_length=300
)
```

## Performance Benefits

| Metric | Old (Simple Truncation) | New (SOTA Extraction) |
|--------|------------------------|----------------------|
| **Relevance** | Low (first 200 chars) | High (query-aware) |
| **Code handling** | Poor (cuts mid-function) | Excellent (preserves blocks) |
| **Multi-segment** | No | Yes (with ellipsis) |
| **Boundary respect** | No | Yes (sentences, brackets) |
| **Token efficiency** | Same | Better (shows only relevant) |

## Examples

### Example 1: Query-Aware vs Simple Truncation

**Content (1000 chars):**
```
[200 chars of intro]
[300 chars about authentication and JWT]
[500 chars of conclusion]
```

**Simple truncation:**
```
Result: [200 chars of intro]...
Contains query terms: No
```

**SOTA extraction with query="authentication JWT":**
```
Result: [300 chars about authentication and JWT]
Contains query terms: Yes
Relevance: High
```

### Example 2: Code Block Preservation

**Content:**
```python
def helper(): pass

def authenticate_user(username, password):
    """Main authentication function"""
    if not valid(username, password):
        return None
    token = generate_jwt(username)
    return token

def other_func(): pass
```

**Query:** "authenticate"

**Simple truncation (cuts at 200 chars):**
```python
def helper(): pass

def authenticate_user(username, password):
    """Main authentication function"""
    if not valid(username, password):
        ret...
```
❌ Incomplete function, invalid syntax

**SOTA extraction:**
```python
def authenticate_user(username, password):
    """Main authentication function"""
    if not valid(username, password):
        return None
    token = generate_jwt(username)
    return token
```
✅ Complete function, valid syntax

## Algorithm Details

### BM25-Inspired Scoring

For each segment and query term:

```python
# Term frequency in segment
tf = segment.count(term)

# Inverse document frequency
idf = log(1 + (total_segments - segments_with_term + 0.5) / (segments_with_term + 0.5))

# Length normalization
k1 = 1.5  # Term frequency saturation
b = 0.75  # Length normalization factor
length_norm = 1 - b + b * (segment_length / avg_segment_length)

# BM25 formula
score = idf * (tf * (k1 + 1)) / (tf + k1 * length_norm)
```

### Segment Selection

1. Sort segments by score (descending)
2. Select highest scored segment
3. Add next highest scored segment if:
   - No overlap with selected segments
   - Fits within max_length (with " ... " separators)
   - Has score > 0 (contains query terms)
4. Sort selected segments by position (natural reading order)
5. Build final snippet with ellipsis between gaps

## Testing

Comprehensive test suite with 23 tests covering:

- ✅ Query-aware extraction
- ✅ Code block detection
- ✅ Multi-segment extraction
- ✅ Boundary preservation
- ✅ Relevance scoring
- ✅ Stop word removal
- ✅ Edge cases (Unicode, long queries, etc.)

**Run tests:**
```bash
python3 -m pytest test_snippet_extractor.py -v
```

**Test results:**
```
23 passed in 0.02s ✅
```

## Migration Guide

### For Qdrant Vector Store

**File:** `mcp_server/qdrant_vector_store.py`

1. Import the extractor:
```python
from snippet_extractor import extract_snippet
```

2. Replace line 203:
```python
# OLD
snippet = content[:200] + "..." if len(content) > 200 else content

# NEW
snippet = extract_snippet(
    content=content,
    query=query,
    max_length=300
)
```

### For Other Use Cases

```python
from snippet_extractor import extract_snippet

# Simple usage
snippet = extract_snippet(long_content, "search query")

# Custom length
snippet = extract_snippet(long_content, "search query", max_length=500)

# No query (smart truncation)
snippet = extract_snippet(long_content, max_length=300)
```

## Configuration Options

### SnippetExtractor Parameters

- `max_length` (int): Maximum snippet length in characters (default: 300)
- `context_chars` (int): Characters of context around matches (default: 50)
- `min_segment_length` (int): Minimum segment length to consider (default: 30)

### Tunable Constants

In `snippet_extractor.py`:

```python
# BM25 parameters
k1 = 1.5  # Term frequency saturation
b = 0.75  # Length normalization

# Scoring bonuses
code_bonus = 1.3        # Multiplier for code blocks
position_penalty = 0.2  # Max penalty for later position

# Segmentation thresholds
large_paragraph_threshold = 200  # Split paragraphs > 200 chars by sentences
separator_gap = 10               # Minimum gap to show " ... "
```

## Limitations & Future Work

### Current Limitations

1. **Language support**: Code detection optimized for Python, JavaScript, TypeScript, Go
   - *Future*: Add support for Java, C++, Rust, etc.

2. **Embedding-aware**: Not yet aware of embedding models
   - *Future*: Optimize snippet boundaries for embedding alignment

3. **Query highlighting**: Doesn't highlight query terms in output
   - *Future*: Add optional HTML/markdown highlighting

4. **Caching**: No caching of segmentation/scoring
   - *Future*: Cache segments for frequently accessed documents

### Planned Enhancements

- [ ] Query term highlighting in snippets
- [ ] Embedding-aware boundary detection
- [ ] Additional language support
- [ ] Configurable separator (" ... " → custom)
- [ ] HTML/markdown output formatting
- [ ] Performance benchmarks

## Credits

**Implementation:** SOTA snippet extraction system
**Algorithm:** BM25-inspired scoring with code-aware segmentation
**Testing:** Comprehensive test suite with 23 test cases
**Integration:** Qdrant Vector Store (mcp_server/qdrant_vector_store.py)

---

**Version:** 1.0.0
**Last Updated:** 2025-01-16
**Status:** ✅ Production Ready (all tests passing)
