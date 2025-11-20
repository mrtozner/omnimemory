"""
Example: Using FileStructureExtractor

Demonstrates how to extract structural facts from code files.
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from structure_extractor import FileStructureExtractor


async def example_basic_extraction():
    """Basic example: Extract facts from a Python file."""
    print("=" * 60)
    print("Example 1: Basic Extraction")
    print("=" * 60)

    # Sample Python code
    sample_code = '''
import os
import sys
from pathlib import Path
from typing import Dict, List

class FileProcessor:
    """Process files."""

    def __init__(self, root_dir):
        self.root_dir = root_dir

    def process(self, filename):
        """Process a file."""
        pass

def validate_path(path):
    """Validate a file path."""
    return Path(path).exists()
'''

    # Create extractor
    extractor = FileStructureExtractor()

    # Extract facts (no need to start LSP for import extraction)
    print("\nExtracting facts from sample code...\n")

    # Create a temporary file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(sample_code)
        temp_file = f.name

    try:
        facts = await extractor.extract_facts(temp_file, sample_code)

        print(f"Extracted {len(facts)} facts:\n")

        # Group facts by predicate
        facts_by_predicate = {}
        for fact in facts:
            predicate = fact["predicate"]
            if predicate not in facts_by_predicate:
                facts_by_predicate[predicate] = []
            facts_by_predicate[predicate].append(fact["object"])

        # Print grouped facts
        for predicate, objects in sorted(facts_by_predicate.items()):
            print(f"{predicate}:")
            for obj in objects:
                print(f"  - {obj}")
            print()

    finally:
        # Cleanup
        import os

        if os.path.exists(temp_file):
            os.unlink(temp_file)


async def example_multiple_files():
    """Example: Extract facts from multiple files."""
    print("=" * 60)
    print("Example 2: Multiple File Extraction")
    print("=" * 60)

    files = {
        "auth.py": """
import bcrypt
from datetime import datetime

class AuthManager:
    def authenticate(self, username, password):
        pass

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())
""",
        "database.py": """
import sqlite3
from typing import Optional

class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)

def create_tables():
    pass
""",
        "utils.ts": """
import axios from 'axios';
import { User } from './types';

export class APIClient {
    async fetchUser(id: string): Promise<User> {
        const response = await axios.get(`/users/${id}`);
        return response.data;
    }
}

export function formatDate(date: Date): string {
    return date.toISOString();
}
""",
    }

    extractor = FileStructureExtractor()

    import tempfile
    import os

    for filename, content in files.items():
        print(f"\n--- {filename} ---")

        # Create temporary file
        suffix = "." + filename.split(".")[-1]
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            facts = await extractor.extract_facts(temp_file, content)

            # Count facts by type
            imports = [f for f in facts if f["predicate"] == "imports"]
            classes = [f for f in facts if f["predicate"] == "defines_class"]
            functions = [f for f in facts if f["predicate"] == "defines_function"]
            methods = [f for f in facts if f["predicate"] == "defines_method"]

            print(f"Imports: {len(imports)}")
            for imp in imports[:3]:  # Show first 3
                print(f"  - {imp['object']}")
            if len(imports) > 3:
                print(f"  ... and {len(imports) - 3} more")

            if classes:
                print(f"Classes: {len(classes)}")
                for cls in classes:
                    print(f"  - {cls['object']}")

            if functions:
                print(f"Functions: {len(functions)}")
                for func in functions:
                    print(f"  - {func['object']}")

            if methods:
                print(f"Methods: {len(methods)}")
                for method in methods:
                    print(f"  - {method['object']}")

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


async def example_fact_filtering():
    """Example: Filter facts by type."""
    print("\n" + "=" * 60)
    print("Example 3: Filtering Facts")
    print("=" * 60)

    sample_code = """
import requests
from typing import Dict, List
from .models import User

class UserService:
    def get_user(self, user_id):
        pass

    def create_user(self, data):
        pass

def validate_email(email):
    pass
"""

    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(sample_code)
        temp_file = f.name

    try:
        extractor = FileStructureExtractor()
        facts = await extractor.extract_facts(temp_file, sample_code)

        print("\nAll imports:")
        import_facts = [f for f in facts if f["predicate"] == "imports"]
        for fact in import_facts:
            print(f"  {fact['object']}")

        print("\nExternal imports only (excluding typing and relative imports):")
        external_imports = [
            f
            for f in import_facts
            if not any(x in f["object"] for x in ["typing", "models"])
        ]
        for fact in external_imports:
            print(f"  {fact['object']}")

        print("\nAll symbol definitions:")
        definition_facts = [
            f
            for f in facts
            if f["predicate"] in ["defines_class", "defines_function", "defines_method"]
        ]
        for fact in definition_facts:
            print(f"  {fact['predicate']}: {fact['object']}")

    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


async def main():
    """Run all examples."""
    await example_basic_extraction()
    await example_multiple_files()
    await example_fact_filtering()

    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Integrate with Tri-Index to build knowledge graph")
    print("2. Cache extracted facts to avoid re-processing")
    print("3. Add support for more languages as needed")


if __name__ == "__main__":
    asyncio.run(main())
