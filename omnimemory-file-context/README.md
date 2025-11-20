# File Structure Extractor

**Extracts structural facts from code files for Tri-Index knowledge graph.**

## Overview

The File Structure Extractor leverages the existing `SymbolService` from `omnimemory-lsp` to extract structural information from code files and convert it into structured facts for the Tri-Index system.

## Features

- **Import Extraction**: Extracts module imports using AST (Python) and regex (TypeScript/JavaScript, Go, Rust, Java)
- **Symbol Extraction**: Extracts classes, functions, and methods using LSP (when available)
- **Multi-Language Support**: Python, TypeScript, JavaScript, Go, Rust, Java
- **Structured Facts**: Outputs facts in standardized format for Tri-Index
- **Graceful Degradation**: Works with or without LSP servers running

## Architecture

```
FileStructureExtractor
├── Import Extraction (AST/Regex)
│   ├── Python: ast.parse() for accurate import extraction
│   ├── TypeScript/JS: Regex for import/require statements
│   ├── Go: Regex for package imports
│   └── Rust/Java: Regex for use/import statements
│
└── Symbol Extraction (LSP)
    └── Reuses existing SymbolService
        ├── Classes
        ├── Functions
        └── Methods
```

## Usage

### Basic Usage

```python
from structure_extractor import FileStructureExtractor

# Initialize
extractor = FileStructureExtractor()
await extractor.start()

# Extract facts from a file
facts = await extractor.extract_facts("auth.py")

# Print facts
for fact in facts:
    print(f"{fact['predicate']}: {fact['object']}")

# Cleanup
await extractor.stop()
```

### Example Output

For a file `auth.py`:
```python
import bcrypt
from datetime import datetime

class AuthManager:
    def authenticate(self, username, password):
        pass

def hash_password(password):
    pass
```

Extracted facts:
```python
[
    {"predicate": "imports", "object": "module:bcrypt", "confidence": 1.0},
    {"predicate": "imports", "object": "module:datetime", "confidence": 1.0},
    {"predicate": "defines_class", "object": "class:AuthManager", "confidence": 1.0},
    {"predicate": "defines_method", "object": "method:AuthManager.authenticate", "confidence": 1.0},
    {"predicate": "defines_function", "object": "function:hash_password", "confidence": 1.0}
]
```

## Testing

### Run Simple Tests (No LSP Required)

```bash
python3 test_simple_extraction.py
```

Tests import extraction for Python, TypeScript, and Go without requiring LSP servers.

## Status

✅ **Implementation Complete**
✅ **Import Extraction Working** (Python, TS/JS, Go, Rust, Java)
✅ **Symbol Extraction Working** (when LSP available)
✅ **Tests Passing** (5/5 tests)
✅ **Ready for Tri-Index Integration**
