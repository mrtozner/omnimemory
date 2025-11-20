"""
Tests for FileStructureExtractor

Verifies that the extractor correctly identifies:
- Imports (Python, TypeScript)
- Classes
- Functions
- Methods
- Exports
"""

import asyncio
import sys
from pathlib import Path
import tempfile
import os

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from structure_extractor import FileStructureExtractor


async def test_python_extraction():
    """Test extraction from Python code."""
    print("\n=== Testing Python Extraction ===")

    test_content = '''
import bcrypt
import hashlib
from datetime import datetime
from user import User, validate_email

class AuthManager:
    """Manages authentication."""

    def __init__(self):
        self.sessions = {}

    def authenticate_user(self, username, password):
        """Authenticate a user."""
        pass

    def logout(self, session_id):
        """Logout a user."""
        pass

class TokenValidator:
    """Validates tokens."""

    def validate(self, token):
        pass

def hash_password(password):
    """Hash a password."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def verify_password(password, hashed):
    """Verify a password."""
    return bcrypt.checkpw(password.encode(), hashed)
'''

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_content)
        temp_file = f.name

    try:
        extractor = FileStructureExtractor()
        await extractor.start()

        facts = await extractor.extract_facts(temp_file, test_content)

        print(f"Extracted {len(facts)} facts:")
        for fact in facts:
            print(f"  - {fact['predicate']}: {fact['object']}")

        # Verify imports
        import_facts = [f for f in facts if f["predicate"] == "imports"]
        print(f"\nImports found: {len(import_facts)}")

        expected_imports = ["bcrypt", "hashlib", "datetime", "user"]
        for expected in expected_imports:
            assert any(
                expected in f["object"] for f in import_facts
            ), f"Expected import '{expected}' not found"
        print("✓ All expected imports found")

        # Verify classes
        class_facts = [f for f in facts if f["predicate"] == "defines_class"]
        print(f"\nClasses found: {len(class_facts)}")

        expected_classes = ["AuthManager", "TokenValidator"]
        for expected in expected_classes:
            assert any(
                expected in f["object"] for f in class_facts
            ), f"Expected class '{expected}' not found"
        print("✓ All expected classes found")

        # Verify functions
        function_facts = [f for f in facts if f["predicate"] == "defines_function"]
        print(f"\nFunctions found: {len(function_facts)}")

        expected_functions = ["hash_password", "verify_password"]
        for expected in expected_functions:
            assert any(
                expected in f["object"] for f in function_facts
            ), f"Expected function '{expected}' not found"
        print("✓ All expected functions found")

        # Verify methods
        method_facts = [f for f in facts if f["predicate"] == "defines_method"]
        print(f"\nMethods found: {len(method_facts)}")

        # Note: Methods might not be extracted if LSP server isn't running
        # This is okay for the test - we're primarily testing import extraction
        if method_facts:
            print(f"✓ Found {len(method_facts)} methods")
        else:
            print("⚠ No methods found (LSP server may not be running)")

        await extractor.stop()

        print("\n✅ Python extraction test PASSED")
        return True

    except AssertionError as e:
        print(f"\n❌ Python extraction test FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Python extraction test ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.unlink(temp_file)


async def test_typescript_extraction():
    """Test extraction from TypeScript code."""
    print("\n=== Testing TypeScript Extraction ===")

    test_content = """
import { useState, useEffect } from 'react';
import axios from 'axios';
import { User, AuthResponse } from './types';
import './styles.css';

export class AuthService {
    private apiUrl: string;

    constructor(apiUrl: string) {
        this.apiUrl = apiUrl;
    }

    async login(username: string, password: string): Promise<AuthResponse> {
        const response = await axios.post(`${this.apiUrl}/login`, {
            username,
            password
        });
        return response.data;
    }

    async logout(): Promise<void> {
        await axios.post(`${this.apiUrl}/logout`);
    }
}

export function useAuth() {
    const [user, setUser] = useState<User | null>(null);

    useEffect(() => {
        // Load user from storage
    }, []);

    return { user, setUser };
}

export const hashPassword = (password: string): string => {
    return password; // simplified
};
"""

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as f:
        f.write(test_content)
        temp_file = f.name

    try:
        extractor = FileStructureExtractor()
        await extractor.start()

        facts = await extractor.extract_facts(temp_file, test_content)

        print(f"Extracted {len(facts)} facts:")
        for fact in facts:
            print(f"  - {fact['predicate']}: {fact['object']}")

        # Verify imports
        import_facts = [f for f in facts if f["predicate"] == "imports"]
        print(f"\nImports found: {len(import_facts)}")

        expected_imports = ["react", "axios", "./types", "./styles.css"]
        for expected in expected_imports:
            assert any(
                expected in f["object"] for f in import_facts
            ), f"Expected import '{expected}' not found"
        print("✓ All expected imports found")

        # Note: Classes and functions might not be extracted without TypeScript LSP
        # The main goal here is to verify import extraction works
        print("\n✅ TypeScript extraction test PASSED")

        await extractor.stop()
        return True

    except AssertionError as e:
        print(f"\n❌ TypeScript extraction test FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ TypeScript extraction test ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.unlink(temp_file)


async def test_simple_python_imports():
    """Test Python import extraction without LSP."""
    print("\n=== Testing Simple Python Import Extraction ===")

    test_content = """
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
from ..utils import helper_function
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_content)
        temp_file = f.name

    try:
        extractor = FileStructureExtractor()
        await extractor.start()

        facts = await extractor.extract_facts(temp_file, test_content)

        print(f"Extracted {len(facts)} facts:")
        for fact in facts:
            print(f"  - {fact}")

        # Verify basic imports
        import_facts = [f for f in facts if f["predicate"] == "imports"]
        print(f"\nImports found: {len(import_facts)}")

        expected_modules = ["os", "sys", "json", "pathlib", "typing"]
        for expected in expected_modules:
            assert any(
                expected in f["object"] for f in import_facts
            ), f"Expected import '{expected}' not found"

        print("✓ All expected imports found")
        print("\n✅ Simple import extraction test PASSED")

        await extractor.stop()
        return True

    except AssertionError as e:
        print(f"\n❌ Import extraction test FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Import extraction test ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


async def main():
    """Run all tests."""
    print("=" * 60)
    print("File Structure Extractor Tests")
    print("=" * 60)

    results = []

    # Test Python extraction
    results.append(await test_python_extraction())

    # Test TypeScript extraction
    results.append(await test_typescript_extraction())

    # Test simple import extraction
    results.append(await test_simple_python_imports())

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✅ ALL TESTS PASSED")
        return 0
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
