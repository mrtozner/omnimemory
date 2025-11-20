"""
Test suite for WitnessSelector
"""

import asyncio
from witness_selector import WitnessSelector


async def test_mmr_selection():
    """Test MMR selection with diverse code snippets."""
    selector = WitnessSelector()
    await selector.initialize()

    test_content = '''
import bcrypt
from user import User

class AuthManager:
    """Manages authentication."""

    def authenticate_user(self, username, password):
        """Authenticate a user."""
        pass

    def logout_user(self, session_id):
        """Logout a user."""
        pass

def hash_password(password):
    """Hash a password."""
    return bcrypt.hash(password)

def verify_password(password, hashed):
    """Verify a password."""
    return bcrypt.verify(password, hashed)
'''

    witnesses = await selector.select_witnesses(test_content, max_witnesses=5)

    # Verify diversity
    assert len(witnesses) <= 5, f"Too many witnesses: {len(witnesses)}"
    assert len(witnesses) >= 3, f"Too few witnesses: {len(witnesses)}"

    # Verify structure
    for w in witnesses:
        assert "text" in w, "Missing 'text' field"
        assert "type" in w, "Missing 'type' field"
        assert "line" in w, "Missing 'line' field"
        assert "score" in w, "Missing 'score' field"

    # Verify types
    types = [w["type"] for w in witnesses]
    assert (
        "function_signature" in types or "class_declaration" in types
    ), "Should have at least one function or class"

    # Verify scores
    for w in witnesses:
        assert 0 <= w["score"] <= 1.0, f"Invalid score: {w['score']}"

    print(f"✓ Selected {len(witnesses)} diverse witnesses")
    for w in witnesses:
        print(f"  {w['type']:20s} | Line {w['line']:3d} | Score: {w['score']:.3f}")
        print(f"    {w['text'][:60]}...")


async def test_typescript_extraction():
    """Test extraction of TypeScript/JavaScript constructs."""
    selector = WitnessSelector()
    await selector.initialize()

    test_content = """
import { User } from './types';
import * as bcrypt from 'bcrypt';

export interface AuthConfig {
    secret: string;
    expiresIn: number;
}

export class AuthService {
    constructor(private config: AuthConfig) {}
}

export async function authenticate(
    username: string,
    password: string
): Promise<User> {
    // Implementation
    return null;
}
"""

    witnesses = await selector.select_witnesses(test_content, max_witnesses=5)

    assert len(witnesses) >= 3, "Should extract multiple witnesses"

    types = [w["type"] for w in witnesses]
    print(f"\n✓ Extracted {len(witnesses)} TypeScript/JavaScript witnesses")
    for w in witnesses:
        print(f"  {w['type']:20s} | Line {w['line']:3d}")
        print(f"    {w['text'][:60]}...")


async def test_diversity():
    """Test that MMR ensures diversity."""
    selector = WitnessSelector()
    await selector.initialize()

    # File with many similar functions
    test_content = """
def get_user_by_id(user_id):
    pass

def get_user_by_name(name):
    pass

def get_user_by_email(email):
    pass

def update_user(user_id, data):
    pass

def delete_user(user_id):
    pass

class User:
    pass
"""

    witnesses = await selector.select_witnesses(test_content, max_witnesses=5)

    # Should select diverse items, not just similar functions
    types = [w["type"] for w in witnesses]

    print(f"\n✓ Diversity test: Selected {len(witnesses)} witnesses")
    for w in witnesses:
        print(f"  {w['type']:20s} | Score: {w['score']:.3f}")
        print(f"    {w['text'][:60]}...")

    # Should include the class (most different from functions)
    assert "class_declaration" in types, "Should select diverse types including class"


async def test_small_file():
    """Test with file smaller than max_witnesses."""
    selector = WitnessSelector()
    await selector.initialize()

    test_content = """
def foo():
    pass

def bar():
    pass
"""

    witnesses = await selector.select_witnesses(test_content, max_witnesses=5)

    # Should return all candidates when fewer than max
    assert len(witnesses) == 2, f"Should return all 2 candidates, got {len(witnesses)}"

    print(f"\n✓ Small file test: Returned all {len(witnesses)} witnesses")


async def test_empty_file():
    """Test with empty file."""
    selector = WitnessSelector()
    await selector.initialize()

    test_content = """
# Just comments
# No code
"""

    witnesses = await selector.select_witnesses(test_content, max_witnesses=5)

    assert len(witnesses) == 0, "Should return empty list for file with no code"

    print("\n✓ Empty file test: Correctly returned 0 witnesses")


async def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing WitnessSelector with MMR Algorithm")
    print("=" * 70)

    try:
        await test_mmr_selection()
        await test_typescript_extraction()
        await test_diversity()
        await test_small_file()
        await test_empty_file()

        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
