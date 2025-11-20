# OmniMemory CLI

A modern Rust CLI interface for OmniMemory - AI-powered command suggestion and failure analysis based on the MCP Gateway architecture.

## Features

- **🤖 AI-Powered Suggestions**: Context-aware command suggestions based on your workflow
- **🔍 Failure Analysis**: Intelligent analysis of command failures with remediation steps
- **📋 Context Assembly**: Gather and format context for AI operations
- **👤 Profile Management**: Multiple configuration profiles for different workflows
- **🔍 Fact Database**: Query historical command and operation data
- **⚙️ Configuration Management**: Flexible preference system with scope-based inheritance
- **🔧 Daemon Management**: Control the OmniMemory background service
- **🩺 System Diagnostics**: Comprehensive health checks and troubleshooting

## Installation

```bash
# Build from source
cargo build --release

# Install globally
cargo install --path .
```

## Usage

### Command Suggestions
```bash
# Get AI-powered suggestions
omni suggest

# Include current context
omni suggest --context

# Generate suggestions without execution (dry run)
omni suggest --dry-run

# Focus on specific domains
omni suggest --domain git docker

# JSON output for scripting
omni suggest --json
```

### Failure Analysis
```bash
# Analyze the last failed operation
omni why-failed

# Analyze specific operation by ID
omni why-failed --id op-123

# Analyze multiple recent operations
omni why-failed --last 5

# Get debugging information
omni why-failed --debug

# Include remediation suggestions
omni why-failed --suggest-fix
```

### Context Assembly
```bash
# Gather all context
omni context

# Include specific context types
omni context --include recent_commands working_directory

# Exclude specific types
omni context --exclude git_status environment

# Set maximum context size
omni context --max-size 5KB

# Filter by time range
omni context --since 1h
```

### Profile Management
```bash
# List all profiles
omni profile list

# Show current profile
omni profile show

# Create new profile
omni profile create developer

# Switch to profile
omni profile set developer

# Delete profile
omni profile delete temp
```

### Fact Database Query
```bash
# Search for facts
omni facts "git status"

# Query by entity
omni facts --entity docker

# Get timeline
omni facts --format timeline

# Get statistics
omni facts --format statistics

# Limit results
omni facts --limit 20
```

### Configuration Management
```bash
# Set preference
omni pref set suggestion.max_results 10

# Get preference
omni pref get suggestion.max_results

# List all preferences
omni pref list

# List with filter
omni pref list --filter suggestion

# Reset preferences
omni pref reset
omni pref reset --scope user --keys suggestion.max_results context.max_size
```

### Daemon Management
```bash
# Start daemon
omni daemon start

# Stop daemon
omni daemon stop

# Check status
omni daemon status

# Restart daemon
omni daemon restart

# View logs
omni daemon logs
omni daemon logs --follow
omni daemon logs --level error
```

### System Diagnostics
```bash
# Run basic health check
omni doctor

# Comprehensive check
omni doctor --comprehensive

# Check specific components
omni doctor --check system storage

# JSON output
omni doctor --format json
```

## Command-Line Options

### Global Options

- `--config <FILE>`: Custom configuration file path
- `--output <FORMAT>`: Output format (`human`, `json`, `plain`, `table`)
- `--verbose, -v`: Increase verbosity (can be used multiple times)
- `--quiet, -q`: Suppress warnings and non-essential output
- `--no-input`: Disable interactive prompts
- `--timeout <DURATION>`: Timeout for operations (e.g., "30s", "1m")

## Configuration

### Configuration File Locations

- **User**: `~/.config/omnimemory/config.toml`
- **Project**: `.omnimemory/config.toml`
- **System**: `/etc/omnimemory/config.toml`

### Preference Scope Precedence

1. Command-line flags (highest)
2. Environment variables
3. Project configuration
4. User configuration  
5. System configuration (lowest)

### Default Preferences

```toml
[suggestion]
max_results = 5

[context]
max_size = "10KB"

[output]
format = "human"

[interactive]
enabled = true

[tools]
docker.enabled = true
git.enabled = true
```

## Architecture

This CLI is built following the MCP Gateway architecture specifications:

- **JSON-RPC over stdio**: Strict MCP protocol compliance
- **Unix Domain Sockets**: High-performance IPC for auxiliary processes
- **Capability Negotiation**: Dynamic tool discovery and registration
- **Security-First**: User consent, privacy boundaries, and sandboxing
- **Modular Design**: Separate command implementations with clean interfaces

## Output Formats

### Human Format (Default)
Rich, formatted output with colors and emojis for readability:

```
🎯 AI-Powered Suggestions
Generated in 42ms using model omni-suggest-v1.2.3

1. Review Git Status
   Check current git status and staged changes
   Category: git | Confidence: 95.0% | Time: < 5s
   Command: git status
   Tags: git, status, quick
```

### JSON Format
Structured data for scripting and automation:

```json
{
  "suggestions": [
    {
      "id": "sug-001",
      "title": "Review Git Status",
      "command": "git status",
      "confidence": 0.95,
      "category": "git"
    }
  ]
}
```

### Plain Format
Tabular output without formatting, suitable for piping:

```
sug-001 | git status | 95% | git
sug-002 | cargo test | 88% | development
```

## Error Handling

The CLI provides comprehensive error handling:

- **Structured Errors**: JSON-formatted error responses
- **Exit Codes**: Standard exit codes (0 = success, non-zero = failure)
- **Progress Indicators**: Visual feedback for long operations
- **Fallback Modes**: Graceful degradation when services are unavailable

## Interactive Features

When stdout is a TTY, the CLI enables:

- **Rich Formatting**: Colors, emojis, and visual indicators
- **Progress Bars**: Visual feedback for operations
- **Confirmation Prompts**: Safety checks for destructive operations
- **Tab Completion**: Shell integration for command completion

## Development

### Building

```bash
# Debug build
cargo build

# Release build
cargo build --release

# Run tests
cargo test

# Check formatting
cargo fmt --check

# Lint code
cargo clippy
```

### Project Structure

```
omnimemory-cli/
├── src/
│   ├── main.rs           # CLI entry point and argument parsing
│   └── commands/         # Command implementations
│       ├── mod.rs        # Command interface and utilities
│       ├── suggest.rs    # Suggest command
│       ├── why_failed.rs # Failure analysis command
│       ├── context.rs    # Context assembly command
│       ├── profile.rs    # Profile management command
│       ├── facts.rs      # Fact database query command
│       ├── pref.rs       # Preference management command
│       ├── daemon.rs     # Daemon management command
│       └── doctor.rs     # Health check command
├── Cargo.toml
└── README.md
```

## Contributing

1. Follow Rust coding standards
2. Add tests for new functionality
3. Update documentation for new commands
4. Ensure compatibility with MCP Gateway architecture
5. Test on multiple platforms (Linux, macOS, Windows)

## License

MIT License - see LICENSE file for details.

## Related Projects

- [MCP Gateway](https://modelcontextprotocol.io/): Core MCP protocol implementation
- [OmniMemory Core](https://github.com/omnimemory/core): Memory storage and indexing
- [OmniMemory AI](https://github.com/omnimemory/ai): AI integration and suggestion engine