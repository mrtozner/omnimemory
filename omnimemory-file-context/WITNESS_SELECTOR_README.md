# Witness Selector - MMR-based Code Snippet Selection

Selects 3-5 most representative code snippets from a file using Maximal Marginal Relevance (MMR) to ensure diversity and relevance for the Tri-Index architecture.

## Features

- **MMR Algorithm**: Balances relevance and diversity
- **Multi-language Support**: Python, TypeScript, JavaScript
- **Efficient**: Reuses existing MLX embeddings
- **Smart Extraction**: Extracts functions, classes, imports, types, decorators

## Quick Start

```python
from witness_selector import WitnessSelector
import asyncio

async def main():
    selector = WitnessSelector()
    await selector.initialize()

    with open('myfile.py', 'r') as f:
        content = f.read()

    witnesses = await selector.select_witnesses(content, max_witnesses=5)

    for w in witnesses:
        print(f"{w['type']:20s} | Line {w['line']:3d} | Score: {w['score']:.3f}")
        print(f"  {w['text']}")

asyncio.run(main())
```

## Output Format

Each witness is a dictionary with:

```python
{
    "text": "def authenticate_user(username, password):",
    "type": "function_signature",
    "line": 10,
    "score": 0.95
}
```

## Witness Types

- `function_signature` - Function/method definitions
- `class_declaration` - Class definitions
- `import` - Import statements
- `type_definition` - TypeScript interfaces/types
- `decorated_definition` - Decorated functions/classes

## MMR Algorithm

MMR balances two objectives:

1. **Relevance**: How well the snippet represents the entire file
2. **Diversity**: How different it is from already selected snippets

Formula: `MMR = λ × Relevance - (1-λ) × MaxSimilarity`

- `λ = 0.7` (default): 70% relevance, 30% diversity
- Higher λ: More relevant but possibly redundant
- Lower λ: More diverse but possibly less representative

## Integration with Tri-Index

The Witness Selector is designed for the Tri-Index architecture:

```
Full File → Witness Selector → 3-5 Witnesses
                                ↓
                          Index these snippets
                                ↓
                        Fast file identification
```

Witnesses serve as "fingerprints" that allow rapid file identification without processing entire files.

## Tests

Run tests:

```bash
cd omnimemory-file-context
python3 test_witness_selector.py
```

All tests should pass:
- ✅ MMR selection with diverse code
- ✅ TypeScript/JavaScript extraction
- ✅ Diversity verification
- ✅ Small file handling
- ✅ Empty file handling

## Performance

- **Embedding**: ~50ms per witness (cached)
- **MMR Selection**: ~100ms for 20 candidates
- **Total**: ~500ms per file (5 witnesses)

Uses MLX acceleration on Apple Silicon for fast embeddings.

## Dependencies

- MLXEmbeddingService (from `omnimemory-embeddings`)
- NumPy
- asyncio

## Technical Details

### Candidate Extraction

The selector extracts candidates by parsing code structure:

1. **Python**: Functions, classes, imports, decorators
2. **TypeScript/JavaScript**: Functions, classes, interfaces, types, imports

### Selection Process

1. Extract candidates from file
2. Embed file and all candidates
3. Calculate relevance scores (cosine similarity to file)
4. Select first witness (highest relevance)
5. Select remaining witnesses using MMR:
   - Balance relevance to file
   - Maximize diversity from selected witnesses

### Why MMR?

Traditional approaches select only the most relevant snippets, which can be redundant:

```python
# Without MMR (all similar)
def get_user_by_id(...)
def get_user_by_name(...)
def get_user_by_email(...)
```

MMR ensures diversity:

```python
# With MMR (diverse types)
class User:
def get_user_by_id(...)
from user import UserManager
```

This provides better file "coverage" for the Tri-Index.
