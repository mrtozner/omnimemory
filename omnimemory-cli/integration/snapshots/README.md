# OmniMemory CLI Snapshot Functionality

The snapshot functionality provides comprehensive command execution snapshots with semantic context, vector storage integration, and intelligent retrieval capabilities.

## Features

### 1. Command Execution Snapshots with Semantic Context
- Automatic capture of commands, outputs, and exit codes
- Rich semantic context including working directory, environment, git info, and docker context
- Command structure analysis and file path extraction
- System information integration

### 2. Vector Storage Integration
- FAISS-based vector storage for semantic similarity search
- Python interface for advanced vector operations
- Embedding generation from snapshot content
- High-performance similarity search

### 3. Snapshot Summarization (≤ 500 characters)
- Automatic generation of concise summaries
- Preservation of key information within character limits
- Context-aware summarization based on importance
- Structured summary format

### 4. Context-Aware Snapshot Creation
- Intelligent importance detection based on command type and context
- Four importance levels: Low, Medium, High, Critical
- Auto-skipping of low-importance commands (configurable)
- Force creation option for important commands

### 5. Snapshot Query Interface
- Natural language queries for semantic search
- Filtering by importance, time range, working directory, and tags
- Similarity scoring and ranking
- Multiple output formats (human-readable, JSON, plain)

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CLI Commands  │────│ Snapshot Manager │────│ Vector Storage  │
│                 │    │                  │    │   (FAISS)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   File Storage   │
                       │   (JSON + Meta)  │
                       └──────────────────┘
```

## Usage Examples

### Creating Snapshots

```bash
# Create a snapshot from a git commit
omni snapshot create --command "git commit -m 'feat: add new feature'" \
                     --output "[main 1234567] feat: add new feature" \
                     --exit-code 0 \
                     --execution-time 250

# Force create a snapshot for a simple command
omni snapshot create --command "ls -la" --force

# Create with custom title and tags
omni snapshot create --command "cargo test" \
                     --title "Rust test suite" \
                     --tag rust --tag testing
```

### Searching Snapshots

```bash
# Semantic search for docker builds
omni snapshot query --query "docker builds that failed"

# Search with filters
omni snapshot query --query "git commits" \
                    --min-importance High \
                    --since 1d \
                    --tag git --tag commit

# Search with similarity threshold
omni snapshot query --query "error debugging" \
                    --min-similarity 0.8 \
                    --limit 5
```

### Managing Snapshots

```bash
# List recent snapshots
omni snapshot list --limit 20

# Filter by importance
omni snapshot list --importance High

# Show snapshot details
omni snapshot show --id snap-12345

# Delete snapshot
omni snapshot delete --id snap-12345 --force

# View statistics
omni snapshot stats

# Cleanup old snapshots
omni snapshot cleanup --dry-run
omni snapshot cleanup --force
```

## Configuration

### Snapshot Config Structure

```toml
[snapshot]
max_summary_length = 500
auto_importance = true
semantic_search_enabled = true

[snapshot.vector_storage]
enabled = true
index_path = "/home/user/.omnimemory/snapshots/vector_index"
dimension = 768
metric_type = "cosine"
index_type = "Flat"

[snapshot.auto_cleanup]
enabled = true
low_importance_after_days = 7
medium_importance_after_days = 30
high_importance_after_days = 90
critical_importance_never_delete = true
max_snapshots_per_directory = 1000
```

## Importance Levels

### Critical
- Commands with `sudo`, `rm`, `git reset` (dangerous operations)
- Commands with unusual exit codes (< 0 or > 100)
- Configurable by command patterns

### High
- Successful `git commit`, `docker build` operations
- Failed commands (non-zero exit codes)
- Commands with significant side effects

### Medium
- Development tools: `cargo test`, `npm install`, `pip install`
- Build operations and package management
- Commands that modify project state

### Low
- Informational commands: `ls`, `pwd`, `which`
- Commands with minimal side effects
- Auto-skipped by default (can be forced)

## File Structure

```
integration/snapshots/
├── mod.rs                 # Module definitions
├── models.rs              # Data structures and enums
├── manager.rs             # Core snapshot management logic
├── vector_storage.rs      # Vector storage interface
├── vector_storage_interface.py  # Python vector operations
└── examples.rs            # Usage examples and tests
```

## Integration with Python Vector Storage

The Rust CLI integrates with the Python vector storage through subprocess calls:

1. **Initialization**: Python script sets up FAISS index
2. **Embedding Addition**: Commands are embedded and added to vector store
3. **Semantic Search**: Query embeddings are generated and searched
4. **Statistics**: Index statistics and performance metrics

### Python Interface

The Python interface provides these methods:
- `initialize()` - Set up vector storage
- `add_embedding()` - Add snapshot embeddings
- `search()` - Semantic similarity search
- `delete_embedding()` - Remove embeddings
- `get_stats()` - Storage statistics
- `optimize_index()` - Index optimization

## Performance Considerations

### Memory Usage
- Vector storage: ~2.4MB per 1000 embeddings (768-dim, float32)
- Snapshot metadata: ~1KB per snapshot
- Auto-cleanup prevents unbounded growth

### Search Performance
- FAISS Flat index: <1ms for typical searches
- IVF indexes for datasets >100K embeddings
- Semantic search with configurable similarity thresholds

### Storage Optimization
- Automatic vector quantization (int8, float8, binary)
- Normalized vectors for cosine similarity
- Compressed snapshot storage

## Security and Privacy

### Local-First Storage
- All snapshots stored locally by default
- No external transmission of command data
- User-controlled retention policies

### Sensitive Data Handling
- Environment variables filtered by default
- User prompts for potentially sensitive commands
- Configurable data retention and cleanup

### Access Control
- Snapshot access based on user permissions
- Audit logging for snapshot operations
- Configurable data export and sharing

## Future Enhancements

### Planned Features
1. **Snapshot Sharing**: Export and import snapshot collections
2. **Team Collaboration**: Shared snapshot repositories
3. **Advanced Analytics**: Usage patterns and optimization insights
4. **Integration Hooks**: API endpoints for external tools
5. **Machine Learning**: Automatic tag generation and classification

### Performance Improvements
1. **Concurrent Processing**: Parallel embedding generation
2. **Caching**: LRU cache for frequent queries
3. **Index Optimization**: Dynamic index restructuring
4. **Streaming**: Real-time snapshot processing

## Testing

Run the snapshot tests:
```bash
cargo test snapshot
```

Integration tests demonstrate:
- Snapshot creation with various command types
- Semantic search functionality
- Importance determination logic
- Vector storage integration
- Auto-cleanup operations

## Troubleshooting

### Common Issues

1. **Vector Storage Not Available**
   - Ensure Python dependencies are installed
   - Check FAISS installation: `pip install faiss-cpu`
   - Verify Python script permissions

2. **Snapshot Creation Fails**
   - Check permissions for snapshot directory
   - Verify sufficient disk space
   - Review command syntax and required fields

3. **Search Returns No Results**
   - Try lowering similarity threshold
   - Check if snapshots exist with matching tags
   - Verify vector storage initialization

4. **High Memory Usage**
   - Enable auto-cleanup
   - Adjust snapshot retention periods
   - Consider using quantized embeddings

### Debug Mode

Enable verbose logging:
```bash
omni snapshot query --query "test" --verbose
```

Check vector storage status:
```bash
omni snapshot stats --verbose
```

## Contributing

When adding new snapshot functionality:

1. **Follow the established patterns** in existing modules
2. **Add comprehensive tests** with realistic scenarios
3. **Update documentation** with usage examples
4. **Consider performance** implications of changes
5. **Maintain backward compatibility** where possible

## License

MIT License - see LICENSE file for details.