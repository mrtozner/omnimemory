#!/usr/bin/env python3
"""
Example usage of UnifiedCacheManager
Demonstrates all 3 tiers and team sharing
"""

from unified_cache_manager import UnifiedCacheManager
import json


def main():
    print("=" * 70)
    print("UnifiedCacheManager Example")
    print("=" * 70)

    # Initialize cache manager
    try:
        cache = UnifiedCacheManager(
            redis_host="localhost", redis_port=6379, redis_db=0, enable_compression=True
        )
        print("\n✅ Connected to Redis\n")
    except Exception as e:
        print(f"\n❌ Failed to connect to Redis: {e}")
        print("   Make sure Redis is running: redis-server")
        return

    # L1 TIER: User Cache
    print("=" * 70)
    print("L1 TIER: User Session Cache (1hr TTL)")
    print("=" * 70)

    user_id = "alice"
    file_path = "/src/authentication.py"

    # Cache read result
    read_result = {
        "content": "def authenticate(user, password): ...",
        "size": 1024,
        "language": "python",
    }
    cache.cache_read_result(user_id, file_path, read_result)
    print(f"✅ Cached read result for user '{user_id}'")

    # Cache search result
    search_result = {
        "results": [
            {"file_path": "/src/auth.py", "score": 0.95},
            {"file_path": "/src/login.py", "score": 0.87},
        ]
    }
    cache.cache_search_result(user_id, "authentication", "tri_index", search_result)
    print(f"✅ Cached search result for user '{user_id}'")

    # Retrieve cached data
    cached_read = cache.get_read_result(user_id, file_path)
    print(f"✅ Retrieved read result: {cached_read['language']}")

    # L2 TIER: Repository Cache (SHARED)
    print("\n" + "=" * 70)
    print("L2 TIER: Repository Cache (7 day TTL, SHARED by team)")
    print("=" * 70)

    repo_id = "project-alpha"
    file_hash = "abc123def456"

    # Cache compressed file
    compressed_content = b"[compressed file content]"
    metadata = {
        "original_size": 10000,
        "compressed_size": 2000,
        "compression_ratio": 5.0,
    }
    cache.cache_file_compressed(repo_id, file_hash, compressed_content, metadata)
    print(f"✅ Cached compressed file for repo '{repo_id}'")

    # Cache embeddings
    embeddings = [0.1, 0.2, 0.3, 0.4, 0.5]
    cache.cache_embeddings(repo_id, file_hash, embeddings)
    print(f"✅ Cached {len(embeddings)} embeddings")

    # Cache BM25 index
    bm25_index = {"authentication": 0.95, "login": 0.87, "user": 0.75}
    cache.cache_bm25_index(repo_id, file_hash, bm25_index)
    print(f"✅ Cached BM25 index with {len(bm25_index)} terms")

    # Retrieve cached data
    cached_embeddings = cache.get_embeddings(repo_id, file_hash)
    print(f"✅ Retrieved embeddings: {len(cached_embeddings)} dimensions")

    # L3 TIER: Workflow Context
    print("\n" + "=" * 70)
    print("L3 TIER: Workflow Context (30 day TTL)")
    print("=" * 70)

    session_id = "session_abc123"
    workflow_context = {
        "workflow_name": "authentication_implementation",
        "current_role": "developer",
        "recent_files": ["/src/auth.py", "/src/login.py"],
        "workflow_step": "implementation",
    }
    cache.cache_workflow_context(session_id, workflow_context)
    print(f"✅ Cached workflow context for session '{session_id}'")

    # Retrieve workflow context
    cached_workflow = cache.get_workflow_context(session_id)
    print(f"✅ Retrieved workflow: {cached_workflow['workflow_name']}")

    # TEAM MANAGEMENT
    print("\n" + "=" * 70)
    print("TEAM MANAGEMENT: Shared Repository Cache")
    print("=" * 70)

    team_id = "team-engineering"

    # Add repository to team
    cache.add_repo_to_team(team_id, repo_id)
    print(f"✅ Added repo '{repo_id}' to team '{team_id}'")

    # Add team members
    cache.add_member_to_team(team_id, "alice", "developer")
    cache.add_member_to_team(team_id, "bob", "developer")
    cache.add_member_to_team(team_id, "charlie", "admin")
    print(f"✅ Added 3 members to team '{team_id}'")

    # Check membership
    is_member = cache.is_team_member(team_id, "alice")
    print(f"✅ Alice is team member: {is_member}")

    # Get team info
    team_repos = cache.get_team_repos(team_id)
    team_members = cache.get_team_members(team_id)
    print(f"✅ Team has {len(team_repos)} repos and {len(team_members)} members")

    # CACHE INVALIDATION
    print("\n" + "=" * 70)
    print("CACHE INVALIDATION: File Update")
    print("=" * 70)

    # Invalidate file when it's modified
    invalidated_count = cache.invalidate_file(repo_id, file_path)
    print(f"✅ Invalidated {invalidated_count} cache entries for modified file")

    # STATISTICS
    print("\n" + "=" * 70)
    print("CACHE STATISTICS")
    print("=" * 70)

    stats = cache.get_stats()
    print(f"Memory Used: {stats.memory_used_mb:.2f} MB")
    print(f"L1 Keys (user): {stats.l1_keys}")
    print(f"L2 Keys (repo): {stats.l2_keys}")
    print(f"L3 Keys (workflow): {stats.l3_keys}")
    print(f"Team Keys: {stats.team_keys}")
    print(f"Total Keys: {stats.total_keys}")
    print(f"Hit Rate: {stats.hit_rate}%")

    # Get specific cache sizes
    user_size = cache.get_user_cache_size(user_id)
    repo_size = cache.get_repo_cache_size(repo_id)
    print(f"\nUser '{user_id}' cache size: {user_size:,} bytes")
    print(f"Repo '{repo_id}' cache size: {repo_size:,} bytes")

    # HEALTH CHECK
    print("\n" + "=" * 70)
    print("HEALTH CHECK")
    print("=" * 70)

    health = cache.health_check()
    print(f"Healthy: {health['healthy']}")
    print(f"Latency: {health['latency_ms']} ms")
    print(f"Compression Enabled: {health['compression_enabled']}")
    print(f"Eviction Policy: {health['eviction_policy']}")

    print("\n" + "=" * 70)
    print("✅ Example Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
