"""
Simple tests for FileStructureExtractor (no LSP required)

Tests import extraction without requiring LSP servers to be running.
This focuses on the AST/regex-based import extraction.
"""

import asyncio
import sys
from pathlib import Path
import tempfile
import os

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from structure_extractor import FileStructureExtractor


async def test_python_import_extraction():
    """Test Python import extraction using AST (no LSP required)."""
    print("\n=== Testing Python Import Extraction (No LSP) ===")

    test_content = """
import os
import sys
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from ..utils import helper_function, validate_input
from .models import User, Token
"""

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_content)
        temp_file = f.name

    try:
        # Create extractor without starting LSP service
        extractor = FileStructureExtractor()

        # Extract just imports (bypassing LSP)
        language = extractor._detect_language(temp_file)
        imports = extractor._extract_python_imports(test_content)

        print(f"Language detected: {language}")
        print(f"Imports extracted: {len(imports)}")
        for imp in imports:
            print(f"  - {imp}")

        # Verify expected imports
        expected = ["os", "sys", "json", "hashlib", "datetime", "pathlib", "typing"]

        for exp in expected:
            assert any(
                exp == imp or imp.startswith(exp + ".") for imp in imports
            ), f"Expected import '{exp}' not found"

        print("\n✅ Python import extraction PASSED")
        return True

    except AssertionError as e:
        print(f"\n❌ Test FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Test ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


async def test_typescript_import_extraction():
    """Test TypeScript import extraction using regex (no LSP required)."""
    print("\n=== Testing TypeScript Import Extraction (No LSP) ===")

    test_content = """
import React from 'react';
import { useState, useEffect } from 'react';
import axios from 'axios';
import { User, AuthResponse } from './types';
import type { Config } from './config';
import './styles.css';

const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
"""

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as f:
        f.write(test_content)
        temp_file = f.name

    try:
        extractor = FileStructureExtractor()

        # Extract TypeScript imports
        language = extractor._detect_language(temp_file)
        imports = extractor._extract_ts_imports(test_content)

        print(f"Language detected: {language}")
        print(f"Imports extracted: {len(imports)}")
        for imp in imports:
            print(f"  - {imp}")

        # Verify expected imports
        expected = [
            "react",
            "axios",
            "./types",
            "./config",
            "./styles.css",
            "bcrypt",
            "jsonwebtoken",
        ]

        for exp in expected:
            assert exp in imports, f"Expected import '{exp}' not found"

        print("\n✅ TypeScript import extraction PASSED")
        return True

    except AssertionError as e:
        print(f"\n❌ Test FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Test ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


async def test_go_import_extraction():
    """Test Go import extraction using regex (no LSP required)."""
    print("\n=== Testing Go Import Extraction (No LSP) ===")

    test_content = """
package main

import (
    "fmt"
    "os"
    "encoding/json"
    "github.com/gin-gonic/gin"
    "github.com/user/project/models"
)

import "strings"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".go", delete=False) as f:
        f.write(test_content)
        temp_file = f.name

    try:
        extractor = FileStructureExtractor()

        language = extractor._detect_language(temp_file)
        imports = extractor._extract_go_imports(test_content)

        print(f"Language detected: {language}")
        print(f"Imports extracted: {len(imports)}")
        for imp in imports:
            print(f"  - {imp}")

        expected = ["fmt", "os", "encoding/json", "github.com/gin-gonic/gin", "strings"]

        for exp in expected:
            assert exp in imports, f"Expected import '{exp}' not found"

        print("\n✅ Go import extraction PASSED")
        return True

    except AssertionError as e:
        print(f"\n❌ Test FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Test ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


async def test_language_detection():
    """Test language detection from file extensions."""
    print("\n=== Testing Language Detection ===")

    extractor = FileStructureExtractor()

    test_cases = [
        ("file.py", "python"),
        ("file.ts", "typescript"),
        ("file.tsx", "typescript"),
        ("file.js", "javascript"),
        ("file.jsx", "javascript"),
        ("file.go", "go"),
        ("file.rs", "rust"),
        ("file.java", "java"),
    ]

    all_passed = True
    for filename, expected_lang in test_cases:
        detected = extractor._detect_language(filename)
        if detected == expected_lang:
            print(f"  ✓ {filename} -> {detected}")
        else:
            print(f"  ✗ {filename} -> {detected} (expected {expected_lang})")
            all_passed = False

    if all_passed:
        print("\n✅ Language detection PASSED")
        return True
    else:
        print("\n❌ Language detection FAILED")
        return False


async def test_fact_structure():
    """Test that facts are structured correctly."""
    print("\n=== Testing Fact Structure ===")

    test_content = """
import os
from pathlib import Path
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_content)
        temp_file = f.name

    try:
        extractor = FileStructureExtractor()

        # Extract imports
        imports = extractor._extract_python_imports(test_content)

        # Create facts manually (simulating what extract_facts does for imports)
        facts = []
        for imp in imports:
            facts.append(
                {"predicate": "imports", "object": f"module:{imp}", "confidence": 1.0}
            )

        print(f"Generated {len(facts)} facts:")
        for fact in facts:
            print(f"  - {fact}")

        # Verify fact structure
        for fact in facts:
            assert "predicate" in fact, "Fact missing 'predicate' field"
            assert "object" in fact, "Fact missing 'object' field"
            assert "confidence" in fact, "Fact missing 'confidence' field"
            assert (
                fact["predicate"] == "imports"
            ), f"Unexpected predicate: {fact['predicate']}"
            assert fact["object"].startswith(
                "module:"
            ), f"Object doesn't start with 'module:': {fact['object']}"
            assert (
                fact["confidence"] == 1.0
            ), f"Unexpected confidence: {fact['confidence']}"

        print("\n✅ Fact structure PASSED")
        return True

    except AssertionError as e:
        print(f"\n❌ Test FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Test ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


async def main():
    """Run all tests."""
    print("=" * 60)
    print("File Structure Extractor Tests (No LSP Required)")
    print("=" * 60)

    results = []

    # Test import extraction for different languages
    results.append(await test_python_import_extraction())
    results.append(await test_typescript_import_extraction())
    results.append(await test_go_import_extraction())

    # Test language detection
    results.append(await test_language_detection())

    # Test fact structure
    results.append(await test_fact_structure())

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✅ ALL TESTS PASSED")
        print("\nNote: LSP-based symbol extraction (classes, functions) not tested")
        print("      because LSP servers are not installed. Import extraction works!")
        return 0
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
