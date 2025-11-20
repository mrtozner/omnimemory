# Smart Compression - Implementation Guide

## Status: 50% Complete - Ready for Final Integration

### ✅ What's Done

1. **Production code parser** (`src/code_parser.py`) - 10+ languages, 811 lines
2. **Data structures** added to `src/visiondrop.py`:
   - `ContentType` enum
   - `ChunkPriority` enum
   - Enhanced `CompressedContext` with new metrics
3. **Code parser imported** and initialized in `VisionDropCompressor.__init__()`

### 🚧 What Needs to be Done

## Step 1: Modify `compress()` method signature

**File**: `src/visiondrop.py:212`

**Change**:
```python
async def compress(
    self,
    context: str,
    query: Optional[str] = None,
    model_id: Optional[str] = None,
    file_path: str = "",  # ADD THIS PARAMETER
) -> CompressedContext:
```

## Step 2: Add smart priority detection at start of `compress()`

**File**: `src/visiondrop.py` - Add after line ~226 (after model_id assignment)

**Add this code**:
```python
# Step 0: Smart content-aware compression
# Parse code to identify structural elements (MUST_KEEP vs COMPRESSIBLE)
content_type = self._detect_content_type(context, file_path)
critical_elements_lines = set()

if content_type == ContentType.CODE:
    # Use production-grade parser for code
    code_elements = self.code_parser.parse(context, file_path)

    # Extract line numbers of MUST_KEEP elements
    for elem in code_elements:
        if elem.priority == "must_keep":
            for line_num in range(elem.line_start, elem.line_end + 1):
                critical_elements_lines.add(line_num)

    logger.info(f"Detected {len(code_elements)} code elements, "
                f"{len(critical_elements_lines)} critical lines to preserve")
else:
    # For non-code, use simple heuristics
    critical_elements = self._extract_critical_elements(context, content_type)
    logger.info(f"Detected {content_type.value}, {len(critical_elements)} critical elements")
```

## Step 3: Mark chunk priorities during chunking

**File**: `src/visiondrop.py` - Add after Step 1 (smart chunking)

**After the line** `chunks = self._smart_chunking(context)` **add**:

```python
# Step 1.5: Mark chunk priorities based on code parsing
chunk_priorities = []
lines = context.split('\n')

for chunk in chunks:
    # Determine if chunk contains critical code elements
    has_critical = False

    if content_type == ContentType.CODE and critical_elements_lines:
        # Check if chunk contains any critical line numbers
        chunk_start_line = 0
        for i, line in enumerate(lines):
            if chunk in line:
                chunk_start_line = i
                break

        # Check if chunk overlaps with critical lines
        chunk_line_count = chunk.count('\n') + 1
        chunk_lines = set(range(chunk_start_line, chunk_start_line + chunk_line_count))
        has_critical = bool(chunk_lines & critical_elements_lines)

    priority = ChunkPriority.MUST_KEEP if has_critical else ChunkPriority.COMPRESSIBLE
    chunk_priorities.append(priority)

logger.info(f"Chunk priorities: {sum(1 for p in chunk_priorities if p == ChunkPriority.MUST_KEEP)} MUST_KEEP, "
            f"{sum(1 for p in chunk_priorities if p == ChunkPriority.COMPRESSIBLE)} COMPRESSIBLE")
```

## Step 4: Modify threshold calculation for priority-aware compression

**File**: `src/visiondrop.py` - Replace the existing threshold calculation

**Find this line** (around line 283):
```python
# Step 5: Token-aware adaptive thresholding
threshold = self._adaptive_threshold_token_aware(
```

**Replace the entire threshold section with**:

```python
# Step 5: MODIFIED - Priority-aware threshold calculation
critical_indices = [i for i, p in enumerate(chunk_priorities) if p == ChunkPriority.MUST_KEEP]
compressible_indices = [i for i, p in enumerate(chunk_priorities) if p == ChunkPriority.COMPRESSIBLE]

# Calculate tokens in critical vs compressible chunks
critical_tokens = sum(chunk_token_counts[i] for i in critical_indices) if critical_indices else 0
compressible_tokens = sum(chunk_token_counts[i] for i in compressible_indices) if compressible_indices else 0

logger.info(f"Token distribution: {critical_tokens} critical, {compressible_tokens} compressible")

# Calculate budget for compressible chunks
target_tokens = int(original_tokens * (1 - self.target_compression))
tokens_available_for_compressible = target_tokens - critical_tokens

if tokens_available_for_compressible <= 0 or not compressible_indices:
    # Critical chunks exceed budget OR no compressible chunks - keep only critical
    logger.warning(f"Critical chunks ({critical_tokens} tokens) exceed budget ({target_tokens} tokens). "
                   f"Compression ratio will be lower than target.")
    retained_indices = critical_indices
else:
    # Select best compressible chunks to fill remaining budget
    compressible_scores = np.array([importance_scores[i] for i in compressible_indices])
    compressible_token_counts_list = [chunk_token_counts[i] for i in compressible_indices]

    # Adjusted compression ratio for compressible chunks
    adjusted_compression = 1 - (tokens_available_for_compressible / compressible_tokens) if compressible_tokens > 0 else 0

    # Token-aware selection of compressible chunks
    threshold = self._adaptive_threshold_token_aware(
        compressible_scores,
        compressible_token_counts_list,
        adjusted_compression,
        compressible_tokens
    )

    retained_compressible = [i for i in compressible_indices
                            if importance_scores[i] >= threshold]
    retained_indices = critical_indices + retained_compressible

    logger.info(f"Retained: {len(critical_indices)} critical + {len(retained_compressible)} compressible chunks")
```

## Step 5: Calculate structural retention metric

**File**: `src/visiondrop.py` - Add before the final return statement

**Find the section** where `CompressedContext` is created and **add before it**:

```python
# NEW: Calculate structural retention
critical_retained = [i for i in critical_indices if i in retained_indices]
structural_retention = (len(critical_retained) / len(critical_indices)) if critical_indices else 1.0

logger.info(f"Structural retention: {structural_retention:.1%} "
            f"({len(critical_retained)}/{len(critical_indices)} critical elements preserved)")
```

## Step 6: Update CompressedContext return value

**File**: `src/visiondrop.py` - Modify the return statement

**Find**:
```python
return CompressedContext(
    original_tokens=original_tokens,
    compressed_tokens=compressed_tokens,
    compression_ratio=compression_ratio,
    retained_indices=retained_indices,
    quality_score=quality_score,
    compressed_text=" ".join(retained_chunks),
)
```

**Replace with**:
```python
return CompressedContext(
    original_tokens=original_tokens,
    compressed_tokens=compressed_tokens,
    compression_ratio=compression_ratio,
    retained_indices=retained_indices,
    quality_score=quality_score,
    compressed_text=" ".join(retained_chunks),
    # Smart compression metrics
    content_type=content_type.value,
    critical_elements_preserved=len(critical_retained),
    structural_retention=structural_retention,
)
```

---

## Step 7: Update Compression Server API

**File**: `src/compression_server.py`

### 7a: Add file_path parameter to endpoint

**Find**:
```python
@app.post("/compress")
async def compress_text(
    context: str = Body(...),
    query: Optional[str] = Body(None),
    tool_id: Optional[str] = Body(None),
    session_id: Optional[str] = Body(None),
):
```

**Replace with**:
```python
@app.post("/compress")
async def compress_text(
    context: str = Body(...),
    query: Optional[str] = Body(None),
    tool_id: Optional[str] = Body(None),
    session_id: Optional[str] = Body(None),
    file_path: str = Body(""),  # ADD THIS
):
```

### 7b: Pass file_path to compressor

**Find**:
```python
result = await compressor.compress(context, query=query)
```

**Replace with**:
```python
result = await compressor.compress(context, query=query, file_path=file_path)
```

### 7c: Add new metrics to response

**Find the return statement** and **add new fields**:

```python
return {
    "compressed_text": result.compressed_text,
    "original_tokens": result.original_tokens,
    "compressed_tokens": result.compressed_tokens,
    "compression_ratio": result.compression_ratio,
    "quality_score": result.quality_score,
    # NEW FIELDS
    "content_type": result.content_type,
    "critical_elements_preserved": result.critical_elements_preserved,
    "structural_retention": result.structural_retention,
}
```

---

## Step 8: Update MCP Tool

**File**: `mcp_server/omnimemory_mcp.py`

**Find** (around line 1200):
```python
compress_response = await client.post(
    f"{COMPRESSION_URL}/compress",
    json={
        "context": original_content,
        "query": query if query else None,
        "tool_id": "claude-code",
        "session_id": CURRENT_SESSION_ID or "mcp-session",
    },
    timeout=30.0
)
```

**Replace with**:
```python
compress_response = await client.post(
    f"{COMPRESSION_URL}/compress",
    json={
        "context": original_content,
        "query": query if query else None,
        "tool_id": "claude-code",
        "session_id": CURRENT_SESSION_ID or "mcp-session",
        "file_path": file_path,  # ADD THIS - we already have it from function parameter!
    },
    timeout=30.0
)
```

---

## Step 9: Test End-to-End

**Create test file**: `/tmp/test_smart_compression_e2e.py`

```python
#!/usr/bin/env python3
"""End-to-end test of smart compression"""

import requests
import json

# Test Python code with WebScraper class
code = """
import requests
from typing import List

class WebScraper:
    '''A web scraper class'''

    def __init__(self, url: str):
        self.url = url

    def fetch_page(self) -> str:
        # Fetch the page
        return requests.get(self.url).text

    def parse_data(self, html: str) -> List[str]:
        '''Parse HTML data'''
        return html.split()
"""

# Save to temp file
with open("/tmp/test_scraper.py", "w") as f:
    f.write(code)

# Compress with file_path
response = requests.post(
    "http://localhost:8001/compress",
    json={
        "context": code,
        "tool_id": "test",
        "session_id": "smart-compression-test",
        "file_path": "/tmp/test_scraper.py",  # Triggers smart parsing
    }
)

data = response.json()

print("=" * 80)
print("SMART COMPRESSION TEST RESULTS")
print("=" * 80)
print(f"Content type:               {data.get('content_type', 'N/A')}")
print(f"Original tokens:            {data['original_tokens']}")
print(f"Compressed tokens:          {data['compressed_tokens']}")
print(f"Compression ratio:          {data['compression_ratio']:.1%}")
print(f"Quality score:              {data['quality_score']:.1%}")
print(f"Critical elements preserved: {data.get('critical_elements_preserved', 'N/A')}")
print(f"Structural retention:       {data.get('structural_retention', 'N/A'):.1%}")
print()

# Check what was preserved
compressed = data['compressed_text']
print("🎯 Critical Elements Check:")
print("-" * 80)
preserved = {
    "WebScraper": "WebScraper" in compressed,
    "fetch_page": "fetch_page" in compressed,
    "parse_data": "parse_data" in compressed,
    "import requests": "import requests" in compressed or "requests" in compressed,
}

for elem, found in preserved.items():
    status = "✅ PRESERVED" if found else "❌ LOST"
    print(f"{status}: {elem}")

print()
retention_rate = sum(preserved.values()) / len(preserved)
print(f"Manual validation: {retention_rate:.1%} retention")

if retention_rate >= 0.8 and data['compression_ratio'] >= 0.80:
    print("\n✅ SUCCESS: Smart compression working correctly!")
    print("   - High structural retention (80%+)")
    print("   - Good compression (80%+)")
else:
    print("\n⚠️  NEEDS TUNING:")
    if retention_rate < 0.8:
        print(f"   - Structural retention too low: {retention_rate:.1%}")
    if data['compression_ratio'] < 0.80:
        print(f"   - Compression too low: {data['compression_ratio']:.1%}")
```

Run: `python3 /tmp/test_smart_compression_e2e.py`

---

## Expected Results

**Before (Current - Blind Compression)**:
```
Compression: 93%
Quality: 99%
Structural Retention: 16.7% ❌
Critical Elements: WebScraper ❌, fetch_page ❌, parse_data ❌
```

**After (Smart Compression)**:
```
Compression: 85-90%
Quality: 99%
Structural Retention: 90%+ ✅
Critical Elements: WebScraper ✅, fetch_page ✅, parse_data ✅
Content Type: code
```

---

## Summary of Changes

**Files Modified**: 3
- `src/visiondrop.py`: ~50 lines added/modified
- `src/compression_server.py`: ~10 lines added
- `mcp_server/omnimemory_mcp.py`: 1 line added

**New File Created**: 1
- `src/code_parser.py`: 811 lines (already done ✅)

**Testing**: 1 test file
- `/tmp/test_smart_compression_e2e.py`: End-to-end validation

---

## Time Estimate

- Step 1-6 (visiondrop.py): ~1 hour
- Step 7 (compression_server.py): ~15 min
- Step 8 (omnimemory_mcp.py): ~5 min
- Step 9 (testing): ~30 min
- Debugging/iteration: ~30 min

**Total**: ~2.5 hours to complete

---

## Rollout Plan

1. Make changes in development branch
2. Test with `/tmp/test_smart_compression_e2e.py`
3. Verify structural retention > 80%
4. Restart compression service
5. Test via MCP tool: `omnimemory_get_context(file_path="/tmp/test_scraper.py")`
6. Monitor dashboard for new metrics
7. Deploy to production

---

## Troubleshooting

### Import error: "No module named 'code_parser'"
**Solution**: Ensure `code_parser.py` is in `src/` directory

### Structural retention shows 0%
**Solution**: Check that `file_path` is being passed through entire stack

### Compression too aggressive (>95%)
**Solution**: Reduce `target_compression` to 0.85 for code files

### Critical elements not preserved
**Solution**: Check parser is detecting elements correctly with test script

---

**Status**: Ready for implementation. All design complete, code structure in place, just needs integration.
