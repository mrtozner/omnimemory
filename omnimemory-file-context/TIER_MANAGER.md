# OmniMemory File Context - Tier Manager

Progressive compression tier management based on Factory.ai's approach.

## Overview

The Tier Manager implements a **4-tier progressive compression system** that automatically adjusts file content based on age and access patterns:

| Tier | Age | Quality | Savings | Use Case |
|------|-----|---------|---------|----------|
| **FRESH** | 0-1h | 100% | 0% | Recently modified/accessed files |
| **RECENT** | 1-24h | 95% | 60% | Recently used files |
| **AGING** | 1-7d | 85% | 90% | Older files, rarely accessed |
| **ARCHIVE** | 7d+ | 70% | 98% | Cold storage, structural outline only |

## Key Features

### 1. Automatic Tier Transitions

Files automatically move through tiers based on:
- **Time**: Fresh → Recent → Aging → Archive
- **Access patterns**: Hot files stay FRESH
- **Modifications**: Changes trigger promotion to FRESH

### 2. Auto-Promotion

Files are automatically promoted to FRESH when:
- **3+ accesses in 24h** (hot file detection)
- **File modification detected** (hash change)

### 3. Progressive Compression

Each tier returns different content:

#### FRESH (100% quality)
- Returns full original content
- No compression applied
- For active development files

#### RECENT (95% quality, 60% savings)
- Returns witnesses (key code snippets)
- Structure preserved
- Sufficient for understanding file purpose

#### AGING (85% quality, 90% savings)
- Returns facts + top 2 witnesses
- Structured metadata (imports, classes, functions)
- Minimal context for reference

#### ARCHIVE (70% quality, 98% savings)
- Returns structural outline only
- File statistics (X classes, Y functions)
- Just enough to know file exists

## Usage

### Basic Usage

```python
from tier_manager import TierManager
import asyncio

async def main():
    # Initialize manager
    mgr = TierManager()

    # Create metadata for new file
    file_path = "src/auth.py"
    content = "import bcrypt\n\nclass AuthManager:\n    pass"
    metadata = mgr.create_metadata(file_path, content)

    # Determine current tier
    tier = mgr.determine_tier(metadata)
    print(f"Current tier: {tier}")  # FRESH

    # Get tier-appropriate content
    tri_index = {
        "witnesses": ["class AuthManager:"],
        "facts": [{"predicate": "defines_class", "object": "AuthManager"}],
        "classes": ["AuthManager"],
        "functions": [],
        "imports": ["bcrypt"]
    }

    tier_content = await mgr.get_tier_content(
        tier,
        tri_index,
        original_content=content
    )

    print(f"Tokens: {tier_content['tokens']}")
    print(f"Quality: {tier_content['quality']}")
    print(f"Savings: {tier_content['compression_ratio']:.0%}")

asyncio.run(main())
```

### Access Tracking

```python
# Update metadata after file access
metadata = mgr.update_access(metadata)

# Check if promotion is needed
if mgr.should_promote(metadata):
    updated = await mgr.promote_to_fresh(file_id)
```

### File Modification Detection

```python
# Calculate hash for new content
new_hash = mgr.calculate_hash(modified_content)

# Add to metadata for comparison
metadata["current_hash"] = new_hash

# Tier determination will detect modification
tier = mgr.determine_tier(metadata)  # Returns FRESH if hash changed
```

## Integration with VisionDrop

The tier manager integrates with VisionDrop compression for cold storage:

```python
from visiondrop import VisionDropCompressor

# Initialize with VisionDrop support
compressor = VisionDropCompressor()
mgr = TierManager(compressor=compressor)

# FRESH tier can decompress from cold storage if needed
tier_content = await mgr.get_tier_content(
    "FRESH",
    tri_index,
    original_content=None  # Will use compressed_original if available
)
```

## File Metadata Structure

```python
{
    "file_path": "src/auth.py",
    "file_hash": "abc123def456...",      # SHA256 of content
    "tier": "FRESH",                     # Current tier
    "tier_entered_at": datetime.now(),   # When entered current tier
    "last_accessed": datetime.now(),     # Last access time
    "access_count": 0,                   # Accesses in last 24h
    "created_at": datetime.now(),        # File creation time
    "current_hash": "xyz789..."          # Optional: for change detection
}
```

## Tri-Index Structure

The tier manager expects a tri-index dictionary:

```python
{
    "witnesses": [                       # Key code snippets
        "class AuthManager:",
        "def authenticate_user(...):",
        "import bcrypt"
    ],
    "facts": [                           # Structured facts
        {"predicate": "imports", "object": "bcrypt"},
        {"predicate": "defines_class", "object": "AuthManager"},
        {"predicate": "defines_function", "object": "authenticate_user"}
    ],
    "classes": ["AuthManager"],          # Class names
    "functions": ["authenticate_user"],  # Function names
    "imports": ["bcrypt", "user"],       # Import statements
    "compressed_original": "..."         # Optional: VisionDrop compressed content
}
```

## Performance Characteristics

### Token Savings

Based on typical code files:

| Tier | Tokens (avg) | Savings | Quality |
|------|--------------|---------|---------|
| FRESH | 1000 | 0% | 100% |
| RECENT | 400 | 60% | 95% |
| AGING | 100 | 90% | 85% |
| ARCHIVE | 20 | 98% | 70% |

### Access Patterns

- **Hot files** (3+ accesses/24h): Stay in FRESH tier
- **Warm files** (1-2 accesses/24h): Move to RECENT
- **Cold files** (no accesses): Progress to AGING → ARCHIVE

### Memory Impact

For 10,000 files:
- All FRESH: 10M tokens
- All ARCHIVE: 200K tokens (98% savings)
- Mixed (typical): 2-3M tokens (70-80% savings)

## Testing

Run the test suite:

```bash
python3 tier_manager.py
```

Run examples:

```bash
python3 tier_manager_example.py
```

## Architecture

```
TierManager
├── determine_tier()          # Calculate appropriate tier
├── get_tier_content()        # Return tier-specific content
├── should_promote()          # Check promotion criteria
├── promote_to_fresh()        # Promote to FRESH tier
├── create_metadata()         # Initialize file metadata
├── update_access()           # Track access patterns
└── calculate_hash()          # Content change detection

Content Builders
├── _build_witness_summary()  # RECENT tier content
├── _build_fact_summary()     # AGING tier content
└── _build_outline()          # ARCHIVE tier content
```

## Design Principles

1. **Automatic**: No manual tier management required
2. **Progressive**: Graceful degradation as files age
3. **Adaptive**: Hot files stay fresh automatically
4. **Efficient**: 60-98% token savings based on tier
5. **Quality-aware**: Balance compression vs information retention

## Future Enhancements

- [ ] Machine learning for optimal tier thresholds
- [ ] Per-project tier configuration
- [ ] Semantic importance scoring for witnesses
- [ ] Compression strategy per file type
- [ ] Real-time tier adjustment based on query patterns

## Related Components

- **VisionDrop**: Compression engine (94.4% reduction, 91% quality)
- **Tri-Index**: Knowledge graph representation
- **File Indexer**: Creates tri-index from source files
- **Context Manager**: Orchestrates tier-based context serving

## License

Part of the OmniMemory project.
