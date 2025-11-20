"""
Example Usage of TierManager

Demonstrates how to use the tier manager for progressive compression.
"""

import asyncio
from datetime import datetime, timedelta
from tier_manager import TierManager


async def example_basic_usage():
    """Basic usage example."""
    print("=" * 60)
    print("Example 1: Basic Tier Management")
    print("=" * 60)

    # Initialize tier manager
    mgr = TierManager()

    # Create initial metadata for a new file
    file_path = "src/auth.py"
    file_content = """
import bcrypt
from user import User

class AuthManager:
    def authenticate_user(self, username: str, password: str) -> User:
        # Hash check logic here
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        return User(username=username)

    def logout_user(self, user: User):
        # Logout logic
        pass
"""

    metadata = mgr.create_metadata(file_path, file_content)
    print(f"\nCreated metadata for {file_path}")
    print(f"  Tier: {metadata['tier']}")
    print(f"  Hash: {metadata['file_hash'][:16]}...")

    # Determine current tier
    current_tier = mgr.determine_tier(metadata)
    print(f"\nCurrent tier: {current_tier}")

    # Simulate the tri-index that would be created by file indexing
    tri_index = {
        "witnesses": [
            "import bcrypt",
            "class AuthManager:",
            "def authenticate_user(self, username: str, password: str) -> User:",
        ],
        "facts": [
            {"predicate": "imports", "object": "bcrypt"},
            {"predicate": "imports", "object": "User"},
            {"predicate": "defines_class", "object": "AuthManager"},
            {"predicate": "defines_function", "object": "authenticate_user"},
            {"predicate": "defines_function", "object": "logout_user"},
        ],
        "classes": ["AuthManager"],
        "functions": ["authenticate_user", "logout_user"],
        "imports": ["bcrypt", "User"],
    }

    # Get content for current tier
    tier_content = await mgr.get_tier_content(
        current_tier, tri_index, original_content=file_content
    )

    print(f"\nTier content:")
    print(f"  Tokens: {tier_content['tokens']}")
    print(f"  Quality: {tier_content['quality']:.0%}")
    print(f"  Compression: {tier_content['compression_ratio']:.0%}")
    print(f"\nContent preview:")
    print(tier_content["content"][:200] + "...")


async def example_tier_transitions():
    """Demonstrate tier transitions over time."""
    print("\n" + "=" * 60)
    print("Example 2: Tier Transitions Over Time")
    print("=" * 60)

    mgr = TierManager()

    # Create metadata
    base_time = datetime.now()
    metadata = {
        "file_path": "src/utils.py",
        "file_hash": "abc123def456",
        "tier": "FRESH",
        "tier_entered_at": base_time,
        "last_accessed": base_time,
        "access_count": 0,
        "created_at": base_time,
    }

    tri_index = {
        "witnesses": ["def format_date():", "def parse_json():"],
        "facts": [
            {"predicate": "defines_function", "object": "format_date"},
            {"predicate": "defines_function", "object": "parse_json"},
        ],
        "classes": [],
        "functions": ["format_date", "parse_json"],
        "imports": ["datetime", "json"],
    }

    # Simulate time progression
    time_points = [
        (timedelta(minutes=30), "30 minutes"),
        (timedelta(hours=2), "2 hours"),
        (timedelta(hours=12), "12 hours"),
        (timedelta(days=3), "3 days"),
        (timedelta(days=10), "10 days"),
    ]

    print("\nTier progression:")
    for delta, label in time_points:
        metadata["tier_entered_at"] = base_time - delta
        tier = mgr.determine_tier(metadata)
        tier_content = await mgr.get_tier_content(tier, tri_index)

        print(f"\n  After {label}:")
        print(f"    Tier: {tier}")
        print(f"    Tokens: {tier_content['tokens']}")
        print(f"    Quality: {tier_content['quality']:.0%}")
        print(f"    Savings: {tier_content['compression_ratio']:.0%}")


async def example_hot_file_promotion():
    """Demonstrate hot file auto-promotion."""
    print("\n" + "=" * 60)
    print("Example 3: Hot File Auto-Promotion")
    print("=" * 60)

    mgr = TierManager()

    # Create old file (would be ARCHIVE tier)
    base_time = datetime.now()
    metadata = {
        "file_path": "src/config.py",
        "file_hash": "xyz789",
        "tier": "ARCHIVE",
        "tier_entered_at": base_time - timedelta(days=30),  # 30 days old
        "last_accessed": base_time - timedelta(days=1),
        "access_count": 0,
        "created_at": base_time - timedelta(days=30),
    }

    print(f"\nInitial state (30 days old):")
    print(f"  Tier: {mgr.determine_tier(metadata)}")
    print(f"  Access count: {metadata['access_count']}")

    # Simulate multiple accesses
    print("\nSimulating multiple accesses in short time...")
    for i in range(5):
        metadata = mgr.update_access(metadata)
        print(f"  Access {i+1}: count = {metadata['access_count']}")

    # Check if promotion is needed
    should_promote = mgr.should_promote(metadata)
    new_tier = mgr.determine_tier(metadata)

    print(f"\nAfter 5 accesses:")
    print(f"  Should promote: {should_promote}")
    print(f"  New tier: {new_tier}")
    print(f"  Reason: Hot file (frequent access detected)")


async def example_file_modification():
    """Demonstrate file modification detection."""
    print("\n" + "=" * 60)
    print("Example 4: File Modification Detection")
    print("=" * 60)

    mgr = TierManager()

    # Original file
    original_content = "def old_function():\n    pass"
    metadata = mgr.create_metadata("src/api.py", original_content)
    metadata["tier_entered_at"] = datetime.now() - timedelta(days=5)  # 5 days old

    print(f"\nOriginal file (5 days old):")
    print(f"  Tier: {mgr.determine_tier(metadata)}")
    print(f"  Hash: {metadata['file_hash'][:16]}...")

    # Modified file
    modified_content = "def new_function():\n    print('updated')"
    new_hash = mgr.calculate_hash(modified_content)

    print(f"\nFile modified:")
    print(f"  New hash: {new_hash[:16]}...")

    # Check tier with current_hash
    metadata["current_hash"] = new_hash
    new_tier = mgr.determine_tier(metadata)

    print(f"\nAfter modification:")
    print(f"  New tier: {new_tier}")
    print(f"  Reason: File content changed (hash mismatch)")


async def main():
    """Run all examples."""
    await example_basic_usage()
    await example_tier_transitions()
    await example_hot_file_promotion()
    await example_file_modification()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
