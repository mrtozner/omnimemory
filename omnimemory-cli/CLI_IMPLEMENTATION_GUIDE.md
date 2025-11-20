# OmniMemory CLI Implementation Guide

**Purpose**: Comprehensive guide for implementing the OmniMemory CLI-first architecture
**Target**: Phase 5A implementation (Weeks 1-2)
**Goal**: CLI as primary interface with agent-optimized help text

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Implementation Roadmap](#implementation-roadmap)
4. [Command Specifications](#command-specifications)
5. [Service Integration](#service-integration)
6. [Agent Optimization](#agent-optimization)
7. [Testing Strategy](#testing-strategy)
8. [Examples](#examples)

---

## Overview

### Vision

Transform OmniMemory from MCP-only to CLI-first:

**Benefits**:
- Works for humans, teams, AND agents
- No MCP context overhead (92%+ reduction)
- Natural developer experience
- Easy to wrap in MCP later
- No vendor lock-in

### Design Principles

1. **Human-First**: Natural CLI patterns developers expect
2. **Agent-Friendly**: Help text optimized for LLM learning
3. **Progressive**: Can wrap in MCP if needed
4. **Scriptable**: Works in CI/CD pipelines
5. **Observable**: Rich output, clear errors

---

## Architecture

### High-Level Structure

```
omnimemory-cli/
├── pyproject.toml              # UV project config
├── README.md                   # User-facing documentation
├── src/
│   ├── omnimemory/
│   │   ├── __init__.py
│   │   ├── cli.py              # Main CLI entry point
│   │   ├── commands/
│   │   │   ├── __init__.py
│   │   │   ├── read.py         # Read command
│   │   │   ├── compress.py     # Compress command
│   │   │   ├── search.py       # Search command
│   │   │   ├── cache.py        # Cache management
│   │   │   └── stats.py        # Statistics
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── compression.py  # Compression service client
│   │   │   ├── embeddings.py   # Embeddings service client
│   │   │   └── metrics.py      # Metrics service client
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── cache.py        # Local cache
│   │   │   ├── config.py       # Configuration
│   │   │   └── output.py       # Rich output helpers
│   │   └── models/
│   │       ├── __init__.py
│   │       └── schemas.py      # Data models
│   └── scripts/                # UV single-file scripts
│       ├── omnimemory_compress.py
│       ├── omnimemory_search.py
│       └── ...
├── tests/
│   ├── test_commands.py
│   ├── test_services.py
│   └── test_integration.py
└── docs/
    ├── CLI_USAGE.md
    ├── AGENT_GUIDE.md
    └── DEVELOPMENT.md
```

### Component Responsibilities

**CLI Entry Point** (`cli.py`):
- Argument parsing (Click)
- Command routing
- Global options (--verbose, --config)
- Version management

**Commands** (`commands/`):
- Individual command implementations
- Input validation
- Service orchestration
- Output formatting

**Services** (`services/`):
- HTTP clients for OmniMemory services
- Request/response handling
- Error handling and retries
- Service health checks

**Utils** (`utils/`):
- Local caching (SQLite)
- Configuration management (TOML)
- Rich terminal output
- Logging

---

## Implementation Roadmap

### Week 1: Core CLI (Days 1-5)

#### Day 1: Project Setup & Infrastructure

**Tasks**:

1. **Initialize Project**:
   ```bash
   mkdir omnimemory-cli
   cd omnimemory-cli
   uv init
   ```

2. **Add Dependencies**:
   ```bash
   uv add click rich httpx pydantic
   uv add --dev pytest pytest-asyncio pytest-mock
   ```

3. **Create Project Structure**:
   ```bash
   mkdir -p src/omnimemory/{commands,services,utils,models}
   mkdir -p tests
   mkdir -p docs
   ```

4. **Basic CLI Entry Point**:
   ```python
   # src/omnimemory/cli.py
   import click
   from rich.console import Console

   console = Console()

   @click.group()
   @click.version_option(version="0.1.0")
   @click.option('--verbose', '-v', is_flag=True, help='Verbose output')
   @click.pass_context
   def cli(ctx, verbose):
       """OmniMemory - AI memory optimization CLI"""
       ctx.ensure_object(dict)
       ctx.obj['verbose'] = verbose

   if __name__ == '__main__':
       cli()
   ```

**Deliverables**:
- Project initialized with UV
- Dependencies installed
- Project structure created
- Basic CLI skeleton running

#### Day 2: Core Commands (Read & Compress)

**Tasks**:

1. **Implement Read Command**:
   ```python
   # src/omnimemory/commands/read.py
   import click
   from pathlib import Path
   from rich.console import Console
   from rich.syntax import Syntax

   console = Console()

   @click.command()
   @click.argument('file', type=click.Path(exists=True))
   @click.option('--compress', is_flag=True,
                 help='Auto-compress output')
   @click.option('--ratio', default=0.9, type=float,
                 help='Compression ratio (0.1-0.9)')
   @click.option('--cache/--no-cache', default=True,
                 help='Use cached version if available')
   @click.option('--format', type=click.Choice(['text', 'json']),
                 default='text', help='Output format')
   def read(file, compress, ratio, cache, format):
       """
       Read file content with optional compression.

       Examples:
         omnimemory read file.py
         omnimemory read file.py --compress --ratio 0.95
         omnimemory read file.py --no-cache --format json

       Token estimates:
         - Without compression: ~1 token per 4 characters
         - With compression (0.9): ~90% reduction
       """
       try:
           # Read file
           content = Path(file).read_text()
           original_size = len(content)

           if compress:
               from ..services.compression import compress_content
               result = compress_content(content, ratio=ratio, use_cache=cache)

               compressed = result['compressed_content']
               compressed_size = len(compressed)
               savings = (1 - compressed_size/original_size) * 100

               # Display stats
               console.print(f"[green]✓ Compressed:[/green] {original_size} → {compressed_size} bytes")
               console.print(f"[cyan]Savings:[/cyan] {savings:.1f}%")
               console.print()

               if format == 'json':
                   import json
                   click.echo(json.dumps(result, indent=2))
               else:
                   console.print(compressed)
           else:
               if format == 'json':
                   import json
                   click.echo(json.dumps({'content': content}))
               else:
                   # Syntax highlighting
                   suffix = Path(file).suffix[1:]  # Remove leading dot
                   syntax = Syntax(content, suffix, theme="monokai", line_numbers=True)
                   console.print(syntax)

       except Exception as e:
           console.print(f"[red]✗ Error reading file:[/red] {e}")
           raise click.Abort()
   ```

2. **Implement Compress Command**:
   ```python
   # src/omnimemory/commands/compress.py
   import click
   from pathlib import Path
   from rich.console import Console
   from rich.progress import Progress

   console = Console()

   @click.command()
   @click.argument('file', type=click.Path(exists=True))
   @click.option('--mode', type=click.Choice(['speed', 'balanced', 'quality']),
                 default='balanced', help='Compression mode')
   @click.option('--ratio', type=float, help='Override compression ratio')
   @click.option('--output', '-o', default='-',
                 help='Output file (- for stdout)')
   @click.option('--stats', is_flag=True,
                 help='Show compression statistics')
   def compress(file, mode, ratio, output, stats):
       """
       Compress file content to save tokens.

       Modes:
         speed:    75-85% savings, fastest
         balanced: 60-70% savings, recommended
         quality:  40-50% savings, best quality

       Examples:
         omnimemory compress file.py --mode balanced
         omnimemory compress file.py --ratio 0.95 --output compressed.txt
         omnimemory compress file.py --stats

       Token estimates:
         speed:    ~400 tokens per 10KB file
         balanced: ~600 tokens per 10KB file
         quality:  ~800 tokens per 10KB file
       """
       try:
           # Read file
           content = Path(file).read_text()
           original_size = len(content)

           # Compress with progress
           with Progress() as progress:
               task = progress.add_task("[cyan]Compressing...", total=100)

               from ..services.compression import compress_content
               result = compress_content(
                   content,
                   mode=mode,
                   ratio=ratio,
                   progress_callback=lambda p: progress.update(task, completed=p)
               )

           compressed = result['compressed_content']
           compressed_size = len(compressed)
           savings = (1 - compressed_size/original_size) * 100

           # Show statistics
           if stats:
               from rich.table import Table
               table = Table(title="Compression Statistics")
               table.add_column("Metric", style="cyan")
               table.add_column("Value", style="green")

               table.add_row("Original Size", f"{original_size:,} bytes")
               table.add_row("Compressed Size", f"{compressed_size:,} bytes")
               table.add_row("Savings", f"{savings:.1f}%")
               table.add_row("Mode", mode)
               table.add_row("Time", f"{result.get('time_ms', 0):.0f}ms")

               console.print(table)
               console.print()

           # Output
           if output == '-':
               click.echo(compressed)
           else:
               Path(output).write_text(compressed)
               console.print(f"[green]✓ Compressed content written to:[/green] {output}")

       except Exception as e:
           console.print(f"[red]✗ Compression failed:[/red] {e}")
           raise click.Abort()
   ```

**Deliverables**:
- Read command functional
- Compress command functional
- Rich terminal output
- Error handling

#### Day 3: Search & Cache Commands

**Tasks**:

1. **Implement Search Command**:
   ```python
   # src/omnimemory/commands/search.py
   import click
   from rich.console import Console
   from rich.table import Table

   console = Console()

   @click.command()
   @click.argument('query')
   @click.option('--semantic', is_flag=True,
                 help='Use semantic search (embeddings)')
   @click.option('--limit', default=10, type=int,
                 help='Maximum results')
   @click.option('--threshold', default=0.7, type=float,
                 help='Similarity threshold (0.0-1.0)')
   @click.option('--format', type=click.Choice(['table', 'json', 'simple']),
                 default='table', help='Output format')
   def search(query, semantic, limit, threshold, format):
       """
       Search OmniMemory for relevant context.

       Search types:
         --semantic: Uses embeddings for semantic similarity
         (default):  Keyword-based search

       Examples:
         omnimemory search "authentication"
         omnimemory search "user login" --semantic --limit 5
         omnimemory search "error" --threshold 0.8 --format json

       Token estimates:
         - ~50 tokens per result
         - Total: ~500 tokens for 10 results
       """
       try:
           from ..services.embeddings import semantic_search
           from ..services.compression import keyword_search

           # Perform search
           if semantic:
               results = semantic_search(query, limit=limit, threshold=threshold)
           else:
               results = keyword_search(query, limit=limit)

           if not results:
               console.print("[yellow]No results found[/yellow]")
               return

           # Format output
           if format == 'json':
               import json
               click.echo(json.dumps(results, indent=2))

           elif format == 'table':
               table = Table(title=f"Search Results: {query}")
               table.add_column("Score", justify="right", style="cyan")
               table.add_column("Content", style="green")
               table.add_column("Source", style="yellow")

               for result in results:
                   table.add_row(
                       f"{result['score']:.2f}",
                       result['content'][:80] + "..." if len(result['content']) > 80 else result['content'],
                       result.get('source', 'Unknown')
                   )

               console.print(table)

           else:  # simple
               for i, result in enumerate(results, 1):
                   console.print(f"{i}. [{result['score']:.2f}] {result['content']}")

       except Exception as e:
           console.print(f"[red]✗ Search failed:[/red] {e}")
           raise click.Abort()
   ```

2. **Implement Cache Management**:
   ```python
   # src/omnimemory/commands/cache.py
   import click
   from rich.console import Console
   from rich.table import Table

   console = Console()

   @click.group()
   def cache():
       """Manage OmniMemory cache."""
       pass

   @cache.command()
   @click.argument('file', required=False)
   def lookup(file):
       """
       Look up file in cache.

       Examples:
         omnimemory cache lookup file.py
         omnimemory cache lookup
       """
       from ..utils.cache import get_cache_client

       cache_client = get_cache_client()

       if file:
           result = cache_client.lookup(file)
           if result:
               console.print(f"[green]✓ Found in cache[/green]")
               console.print(f"Cached: {result['cached_at']}")
               console.print(f"Size: {result['size']} bytes")
           else:
               console.print(f"[yellow]Not in cache[/yellow]")
       else:
           entries = cache_client.list_all()
           if entries:
               table = Table(title="Cache Entries")
               table.add_column("File", style="cyan")
               table.add_column("Size", justify="right", style="green")
               table.add_column("Cached At", style="yellow")

               for entry in entries:
                   table.add_row(
                       entry['file'],
                       f"{entry['size']:,} bytes",
                       entry['cached_at']
                   )

               console.print(table)
           else:
               console.print("[yellow]Cache is empty[/yellow]")

   @cache.command()
   @click.option('--all', is_flag=True, help='Clear entire cache')
   @click.argument('file', required=False)
   def clear(all, file):
       """
       Clear cache entries.

       Examples:
         omnimemory cache clear --all
         omnimemory cache clear file.py
       """
       from ..utils.cache import get_cache_client

       cache_client = get_cache_client()

       if all:
           count = cache_client.clear_all()
           console.print(f"[green]✓ Cleared {count} cache entries[/green]")
       elif file:
           success = cache_client.clear(file)
           if success:
               console.print(f"[green]✓ Cleared cache for {file}[/green]")
           else:
               console.print(f"[yellow]File not in cache[/yellow]")
       else:
           console.print("[red]Specify --all or a file[/red]")
           raise click.Abort()

   @cache.command()
   def stats():
       """Show cache statistics."""
       from ..utils.cache import get_cache_client

       cache_client = get_cache_client()
       stats_data = cache_client.get_stats()

       table = Table(title="Cache Statistics")
       table.add_column("Metric", style="cyan")
       table.add_column("Value", justify="right", style="green")

       table.add_row("Total Entries", f"{stats_data['total_entries']:,}")
       table.add_row("Total Size", f"{stats_data['total_size']:,} bytes")
       table.add_row("Hit Rate", f"{stats_data['hit_rate']:.1f}%")
       table.add_row("Hits", f"{stats_data['hits']:,}")
       table.add_row("Misses", f"{stats_data['misses']:,}")

       console.print(table)
   ```

**Deliverables**:
- Search command (semantic + keyword)
- Cache management commands
- Rich table output
- JSON output option

#### Day 4: Stats & Config Commands

**Tasks**:

1. **Implement Stats Command**:
   ```python
   # src/omnimemory/commands/stats.py
   import click
   from rich.console import Console
   from rich.table import Table
   from rich.panel import Panel

   console = Console()

   @click.command()
   @click.option('--tool', help='Filter by tool (e.g., claude-code)')
   @click.option('--format', type=click.Choice(['table', 'json', 'summary']),
                 default='table', help='Output format')
   @click.option('--period', type=click.Choice(['hour', 'day', 'week', 'month']),
                 default='day', help='Time period')
   def stats(tool, format, period):
       """
       Get OmniMemory statistics.

       Examples:
         omnimemory stats
         omnimemory stats --tool claude-code
         omnimemory stats --period week --format json

       Metrics:
         - Token savings
         - Cache hit rate
         - Operation counts
         - Performance metrics
       """
       try:
           from ..services.metrics import get_stats

           stats_data = get_stats(tool=tool, period=period)

           if format == 'json':
               import json
               click.echo(json.dumps(stats_data, indent=2))

           elif format == 'summary':
               console.print(Panel(
                   f"""
[bold cyan]Token Savings:[/bold cyan] {stats_data['token_savings_pct']:.1f}%
[bold green]Cache Hit Rate:[/bold green] {stats_data['cache_hit_rate']:.1f}%
[bold yellow]Total Operations:[/bold yellow] {stats_data['total_operations']:,}
                   """,
                   title="OmniMemory Statistics",
                   expand=False
               ))

           else:  # table
               # Services table
               services_table = Table(title="Service Statistics")
               services_table.add_column("Service", style="cyan")
               services_table.add_column("Requests", justify="right", style="green")
               services_table.add_column("Avg Time", justify="right", style="yellow")
               services_table.add_column("Success Rate", justify="right", style="blue")

               for service in stats_data['services']:
                   services_table.add_row(
                       service['name'],
                       f"{service['requests']:,}",
                       f"{service['avg_time_ms']:.0f}ms",
                       f"{service['success_rate']:.1f}%"
                   )

               console.print(services_table)

               # Savings table
               savings_table = Table(title="Token Savings")
               savings_table.add_column("Period", style="cyan")
               savings_table.add_column("Tokens Saved", justify="right", style="green")
               savings_table.add_column("$ Saved", justify="right", style="yellow")

               for period_data in stats_data['savings_by_period']:
                   savings_table.add_row(
                       period_data['period'],
                       f"{period_data['tokens_saved']:,}",
                       f"${period_data['cost_saved']:.2f}"
                   )

               console.print(savings_table)

       except Exception as e:
           console.print(f"[red]✗ Failed to get stats:[/red] {e}")
           raise click.Abort()
   ```

2. **Implement Config Commands**:
   ```python
   # src/omnimemory/commands/config.py
   import click
   from rich.console import Console
   from rich.syntax import Syntax

   console = Console()

   @click.group()
   def config():
       """Manage OmniMemory configuration."""
       pass

   @config.command()
   def show():
       """Show current configuration."""
       from ..utils.config import load_config

       config_data = load_config()

       import json
       config_json = json.dumps(config_data, indent=2)
       syntax = Syntax(config_json, "json", theme="monokai", line_numbers=True)
       console.print(syntax)

   @config.command()
   @click.argument('key')
   @click.argument('value')
   def set(key, value):
       """
       Set configuration value.

       Examples:
         omnimemory config set compression.mode balanced
         omnimemory config set cache.ttl 3600
       """
       from ..utils.config import update_config

       try:
           update_config(key, value)
           console.print(f"[green]✓ Set {key} = {value}[/green]")
       except Exception as e:
           console.print(f"[red]✗ Failed to set config:[/red] {e}")
           raise click.Abort()

   @config.command()
   @click.option('--workspace', default='.', help='Workspace path')
   def init(workspace):
       """Initialize OmniMemory configuration."""
       from ..utils.config import initialize_config
       from pathlib import Path

       workspace_path = Path(workspace).resolve()

       try:
           config_path = initialize_config(workspace_path)
           console.print(f"[green]✓ Initialized configuration[/green]")
           console.print(f"Config file: {config_path}")
       except Exception as e:
           console.print(f"[red]✗ Initialization failed:[/red] {e}")
           raise click.Abort()
   ```

**Deliverables**:
- Stats command with rich output
- Config management commands
- Service health monitoring
- Token savings tracking

#### Day 5: Testing & Integration

**Tasks**:

1. **Unit Tests**:
   ```python
   # tests/test_commands.py
   import pytest
   from click.testing import CliRunner
   from omnimemory.cli import cli

   @pytest.fixture
   def runner():
       return CliRunner()

   def test_read_command(runner, tmp_path):
       # Create test file
       test_file = tmp_path / "test.txt"
       test_file.write_text("Hello, OmniMemory!")

       # Test read
       result = runner.invoke(cli, ['read', str(test_file)])
       assert result.exit_code == 0
       assert "Hello, OmniMemory!" in result.output

   def test_compress_command(runner, tmp_path):
       # Create test file
       test_file = tmp_path / "test.txt"
       test_file.write_text("x" * 1000)

       # Test compress
       result = runner.invoke(cli, ['compress', str(test_file), '--stats'])
       assert result.exit_code == 0
       assert "Savings" in result.output

   def test_search_command(runner):
       result = runner.invoke(cli, ['search', 'test'])
       assert result.exit_code == 0

   def test_cache_commands(runner):
       result = runner.invoke(cli, ['cache', 'stats'])
       assert result.exit_code == 0
   ```

2. **Integration Tests**:
   ```python
   # tests/test_integration.py
   import pytest
   from omnimemory.services.compression import compress_content
   from omnimemory.services.embeddings import semantic_search

   @pytest.mark.integration
   def test_compression_service():
       content = "Test content" * 100
       result = compress_content(content)
       assert 'compressed_content' in result
       assert len(result['compressed_content']) < len(content)

   @pytest.mark.integration
   def test_embeddings_service():
       results = semantic_search("test query")
       assert isinstance(results, list)

   @pytest.mark.integration
   def test_end_to_end_workflow(runner, tmp_path):
       # Create file
       test_file = tmp_path / "test.py"
       test_file.write_text("def hello(): pass")

       # Read and compress
       result = runner.invoke(cli, [
           'read', str(test_file), '--compress', '--ratio', '0.9'
       ])
       assert result.exit_code == 0
       assert "Compressed" in result.output
   ```

**Deliverables**:
- Unit test suite (80%+ coverage)
- Integration tests with services
- CI/CD configuration
- Test documentation

### Week 2: Integration & Polish (Days 6-10)

#### Day 6: Service Integration

**Implementation Details**: See [Service Integration](#service-integration) section below

#### Day 7: Caching Layer

**Implementation Details**: See cache implementation in utils section

#### Day 8: Agent Optimization

**Implementation Details**: See [Agent Optimization](#agent-optimization) section below

#### Day 9: Documentation

**Tasks**:
1. User guide with examples
2. Agent integration guide
3. API documentation
4. Troubleshooting guide

#### Day 10: Performance & Release

**Tasks**:
1. Performance benchmarks
2. Token usage measurements
3. Release preparation
4. Deployment documentation

---

## Command Specifications

### omnimemory read

**Purpose**: Read file with optional compression

**Signature**:
```bash
omnimemory read <file> [OPTIONS]
```

**Options**:
- `--compress`: Auto-compress output
- `--ratio FLOAT`: Compression ratio (0.1-0.9)
- `--cache/--no-cache`: Use cache
- `--format [text|json]`: Output format

**Help Text** (Agent-Optimized):
```
Usage: omnimemory read <file> [OPTIONS]

Read file content with optional compression.

OPTIONS:
  --compress          Auto-compress output to save tokens
  --ratio FLOAT       Compression ratio (0.1-0.9) [default: 0.9]
  --cache/--no-cache  Use cached version if available [default: cache]
  --format TEXT       Output format: text, json [default: text]

TOKEN ESTIMATES:
  Without compression: ~1 token per 4 characters
  With 0.9 compression: ~90% token reduction
  With cache hit: Instant retrieval, same savings

EXAMPLES:
  # Read file without compression
  omnimemory read auth.py

  # Read and compress
  omnimemory read auth.py --compress --ratio 0.9

  # Read from cache, JSON output
  omnimemory read auth.py --compress --format json

AGENT USAGE:
  This command is ideal for reading large files. Use --compress
  to reduce tokens. Cache is automatic for frequently accessed files.
```

### omnimemory compress

**Purpose**: Compress file to save tokens

**Signature**:
```bash
omnimemory compress <file> [OPTIONS]
```

**Options**:
- `--mode [speed|balanced|quality]`: Compression mode
- `--ratio FLOAT`: Override compression ratio
- `--output FILE`: Output file (- for stdout)
- `--stats`: Show compression statistics

**Help Text** (Agent-Optimized):
```
Usage: omnimemory compress <file> [OPTIONS]

Compress file content to save tokens.

MODES:
  speed:    75-85% savings, fastest, good for iteration
  balanced: 60-70% savings, recommended for most use cases
  quality:  40-50% savings, best quality preservation

OPTIONS:
  --mode TEXT         Compression mode [default: balanced]
  --ratio FLOAT       Override compression ratio (0.1-0.9)
  --output FILE       Output file, use - for stdout [default: -]
  --stats             Show compression statistics

TOKEN ESTIMATES:
  speed mode:    ~400 tokens per 10KB file
  balanced mode: ~600 tokens per 10KB file
  quality mode:  ~800 tokens per 10KB file

EXAMPLES:
  # Compress with balanced mode (recommended)
  omnimemory compress large_file.py --mode balanced

  # Compress with custom ratio, save to file
  omnimemory compress data.json --ratio 0.95 --output compressed.txt

  # Show compression statistics
  omnimemory compress large_file.py --stats

AGENT USAGE:
  Use this command when working with large files to reduce token
  consumption. Balanced mode is recommended for most use cases.
  Use quality mode when preserving exact details is critical.
```

### omnimemory search

**Purpose**: Search memory for relevant context

**Signature**:
```bash
omnimemory search <query> [OPTIONS]
```

**Options**:
- `--semantic`: Use semantic search
- `--limit INT`: Maximum results
- `--threshold FLOAT`: Similarity threshold
- `--format [table|json|simple]`: Output format

**Help Text** (Agent-Optimized):
```
Usage: omnimemory search <query> [OPTIONS]

Search OmniMemory for relevant context.

SEARCH TYPES:
  Keyword (default): Fast keyword matching
  Semantic:          Uses embeddings for semantic similarity

OPTIONS:
  --semantic           Use semantic search (embeddings)
  --limit INT          Maximum results [default: 10]
  --threshold FLOAT    Similarity threshold (0.0-1.0) [default: 0.7]
  --format TEXT        Output format: table, json, simple [default: table]

TOKEN ESTIMATES:
  ~50 tokens per result
  10 results = ~500 tokens total

EXAMPLES:
  # Keyword search
  omnimemory search "authentication"

  # Semantic search with limit
  omnimemory search "user login flow" --semantic --limit 5

  # JSON output for parsing
  omnimemory search "error handling" --format json

AGENT USAGE:
  Use semantic search when looking for conceptually similar content.
  Use keyword search for exact term matching (faster).
  Adjust threshold to balance precision vs recall.
```

### omnimemory cache

**Purpose**: Manage local cache

**Signature**:
```bash
omnimemory cache [lookup|clear|stats] [OPTIONS]
```

**Subcommands**:
- `lookup [FILE]`: Look up file in cache
- `clear [--all|FILE]`: Clear cache entries
- `stats`: Show cache statistics

**Help Text** (Agent-Optimized):
```
Usage: omnimemory cache [COMMAND] [OPTIONS]

Manage OmniMemory local cache.

COMMANDS:
  lookup [FILE]  Look up file in cache, or list all entries
  clear          Clear cache entries (--all or specific file)
  stats          Show cache statistics

EXAMPLES:
  # Look up specific file
  omnimemory cache lookup auth.py

  # List all cached files
  omnimemory cache lookup

  # Clear specific file
  omnimemory cache clear auth.py

  # Clear entire cache
  omnimemory cache clear --all

  # Show cache stats
  omnimemory cache stats

AGENT USAGE:
  Check cache before reading large files to avoid re-compression.
  Clear cache if file content has changed and you need fresh compression.
  Monitor cache hit rate with 'stats' to optimize performance.
```

### omnimemory stats

**Purpose**: Get performance statistics

**Signature**:
```bash
omnimemory stats [OPTIONS]
```

**Options**:
- `--tool TEXT`: Filter by tool
- `--format [table|json|summary]`: Output format
- `--period [hour|day|week|month]`: Time period

**Help Text** (Agent-Optimized):
```
Usage: omnimemory stats [OPTIONS]

Get OmniMemory performance statistics.

METRICS:
  - Token savings (percentage and absolute)
  - Cache hit rate
  - Operation counts by type
  - Service performance metrics
  - Cost savings estimates

OPTIONS:
  --tool TEXT    Filter by tool (e.g., claude-code)
  --format TEXT  Output format: table, json, summary [default: table]
  --period TEXT  Time period: hour, day, week, month [default: day]

EXAMPLES:
  # Overall stats
  omnimemory stats

  # Stats for specific tool
  omnimemory stats --tool claude-code

  # Weekly summary
  omnimemory stats --period week --format summary

AGENT USAGE:
  Use this command to monitor token savings and optimize operations.
  Check cache hit rate to ensure efficient operation.
  Use JSON format for programmatic parsing.
```

---

## Service Integration

### Compression Service Client

```python
# src/omnimemory/services/compression.py
import httpx
from typing import Dict, Any, Optional, Callable
from ..models.schemas import CompressionRequest, CompressionResult

COMPRESSION_URL = "http://localhost:8001"

class CompressionServiceError(Exception):
    """Compression service error."""
    pass

def compress_content(
    content: str,
    mode: str = "balanced",
    ratio: Optional[float] = None,
    use_cache: bool = True,
    progress_callback: Optional[Callable[[float], None]] = None
) -> Dict[str, Any]:
    """
    Compress content using OmniMemory compression service.

    Args:
        content: Content to compress
        mode: Compression mode (speed, balanced, quality)
        ratio: Override compression ratio (0.1-0.9)
        use_cache: Use cached compression if available
        progress_callback: Optional progress callback

    Returns:
        Compression result with compressed content and stats

    Raises:
        CompressionServiceError: If compression fails
    """
    # Build request
    request_data = {
        "content": content,
        "mode": mode,
        "use_cache": use_cache
    }
    if ratio is not None:
        request_data["ratio"] = ratio

    try:
        # Call compression service
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{COMPRESSION_URL}/compress",
                json=request_data
            )
            response.raise_for_status()

            result = response.json()

            # Simulate progress if callback provided
            if progress_callback:
                progress_callback(100)

            return result

    except httpx.TimeoutException:
        raise CompressionServiceError("Compression service timeout")
    except httpx.HTTPError as e:
        raise CompressionServiceError(f"Compression failed: {e}")
    except Exception as e:
        raise CompressionServiceError(f"Unexpected error: {e}")

def decompress_content(compressed: str) -> str:
    """Decompress content."""
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{COMPRESSION_URL}/decompress",
                json={"compressed_content": compressed}
            )
            response.raise_for_status()

            result = response.json()
            return result['content']

    except Exception as e:
        raise CompressionServiceError(f"Decompression failed: {e}")

def check_compression_service() -> bool:
    """Check if compression service is available."""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{COMPRESSION_URL}/health")
            return response.status_code == 200
    except:
        return False
```

### Embeddings Service Client

```python
# src/omnimemory/services/embeddings.py
import httpx
from typing import List, Dict, Any

EMBEDDINGS_URL = "http://localhost:8000"

def semantic_search(
    query: str,
    limit: int = 10,
    threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Perform semantic search using embeddings.

    Args:
        query: Search query
        limit: Maximum results
        threshold: Similarity threshold (0.0-1.0)

    Returns:
        List of search results with scores and content
    """
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{EMBEDDINGS_URL}/semantic_search",
                json={
                    "query": query,
                    "limit": limit,
                    "threshold": threshold
                }
            )
            response.raise_for_status()

            result = response.json()
            return result.get('results', [])

    except Exception as e:
        raise Exception(f"Semantic search failed: {e}")

def keyword_search(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Perform keyword-based search."""
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{EMBEDDINGS_URL}/search",
                json={"query": query, "limit": limit}
            )
            response.raise_for_status()

            result = response.json()
            return result.get('results', [])

    except Exception as e:
        raise Exception(f"Keyword search failed: {e}")
```

### Metrics Service Client

```python
# src/omnimemory/services/metrics.py
import httpx
from typing import Dict, Any, Optional

METRICS_URL = "http://localhost:8003"

def get_stats(
    tool: Optional[str] = None,
    period: str = "day"
) -> Dict[str, Any]:
    """
    Get OmniMemory statistics.

    Args:
        tool: Filter by tool name
        period: Time period (hour, day, week, month)

    Returns:
        Statistics dictionary
    """
    try:
        params = {"period": period}
        if tool:
            params["tool"] = tool

        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{METRICS_URL}/stats",
                params=params
            )
            response.raise_for_status()

            return response.json()

    except Exception as e:
        raise Exception(f"Failed to get stats: {e}")
```

---

## Agent Optimization

### Prime Prompts

Create `CLAUDE_CODE_PROMPT.md`:

```markdown
# OmniMemory CLI for Claude Code

You now have access to the `omnimemory` CLI for token optimization.

## Key Commands

### Read files with compression
```bash
omnimemory read <file> --compress --ratio 0.9
```
Use this instead of Read tool for large files. Saves 90% tokens.

### Compress content
```bash
omnimemory compress <file> --mode balanced
```
Three modes: speed (85% savings), balanced (70% savings), quality (50% savings).

### Search memory
```bash
omnimemory search "<query>" --semantic --limit 5
```
Find relevant context without reading all files.

### Check cache
```bash
omnimemory cache lookup <file>
```
Before reading large files, check if compressed version is cached.

## When to Use

1. **Large files** (>5KB): Use `omnimemory read --compress`
2. **Repeated reads**: Check cache first
3. **Finding context**: Use semantic search before reading files
4. **Monitoring**: Check stats occasionally to track savings

## Token Savings

- Without OmniMemory: ~1 token per 4 characters
- With compression (0.9): ~90% reduction
- With cache hits: Instant, same savings
- Typical workflow savings: 85-95%

## Examples

```bash
# Read and compress large file
omnimemory read src/large_file.py --compress --ratio 0.9

# Search for relevant context
omnimemory search "authentication flow" --semantic --limit 5

# Check what's in cache
omnimemory cache lookup

# Get performance stats
omnimemory stats
```

Use these commands naturally in your workflow to save tokens.
```

---

## Testing Strategy

### Unit Tests

**Coverage Goals**:
- Commands: 80%+
- Services: 90%+
- Utils: 85%+
- Overall: 80%+

**Test Structure**:
```python
tests/
├── test_commands.py           # Command tests
├── test_services.py           # Service client tests
├── test_utils.py              # Utility tests
├── test_integration.py        # Integration tests
├── test_agent_usage.py        # Agent simulation tests
└── fixtures/
    ├── sample_files.py
    └── mock_services.py
```

### Integration Tests

Test actual service integration:
```python
@pytest.mark.integration
def test_full_compression_workflow():
    # Start services
    # Create file
    # Compress via CLI
    # Verify result
    # Check cache
    pass
```

### Agent Usage Tests

Simulate agent using CLI:
```python
def test_agent_learns_from_help():
    """Test that agent can learn from --help text."""
    runner = CliRunner()

    # Agent reads help
    result = runner.invoke(cli, ['compress', '--help'])
    assert "speed:" in result.output
    assert "balanced:" in result.output

    # Agent uses learned info
    result = runner.invoke(cli, ['compress', 'test.py', '--mode', 'balanced'])
    assert result.exit_code == 0
```

---

## Examples

### Example 1: Basic File Reading

```bash
# Human usage
$ omnimemory read auth.py --compress
✓ Compressed: 5,234 → 523 bytes
Savings: 90.0%

[Compressed content shown with syntax highlighting]
```

### Example 2: Agent Workflow

```bash
# Agent: "I need to understand the authentication flow"

# Step 1: Search for relevant files
$ omnimemory search "authentication" --semantic --limit 3
Score  Content                                 Source
0.92   def authenticate(username, password)... auth.py
0.85   class AuthenticationService...         auth_service.py
0.78   def verify_token(token)...            token_utils.py

# Step 2: Read most relevant file (compressed)
$ omnimemory read auth.py --compress --ratio 0.9
[Compressed content, 90% smaller]

# Result: Agent found and read relevant file using ~500 tokens
# vs traditional approach: ~5,000 tokens
# Savings: 90%
```

### Example 3: CI/CD Integration

```bash
# In CI/CD pipeline
#!/bin/bash

# Compress large log file before uploading
omnimemory compress build.log --mode speed --output build.compressed.log

# Upload compressed version (10x smaller)
aws s3 cp build.compressed.log s3://bucket/
```

---

## Next Steps

1. **Week 1**: Implement core CLI (Days 1-5)
2. **Week 2**: Service integration and polish (Days 6-10)
3. **Testing**: Comprehensive test suite
4. **Documentation**: User and agent guides
5. **Release**: Deploy and monitor usage

---

## Appendices

### A. pyproject.toml

```toml
[project]
name = "omnimemory-cli"
version = "0.1.0"
description = "CLI interface for OmniMemory AI memory optimization"
authors = [{name = "Your Name", email = "your@email.com"}]
requires-python = ">=3.8"
dependencies = [
    "click>=8.1.0",
    "rich>=13.0.0",
    "httpx>=0.24.0",
    "pydantic>=2.0.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-mock>=3.10.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.0.250"
]

[project.scripts]
omnimemory = "omnimemory.cli:cli"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### B. Installation

```bash
# Development installation
cd omnimemory-cli
uv sync
uv run omnimemory --help

# Production installation
uv tool install omnimemory-cli
omnimemory --help
```

### C. Configuration File

```toml
# ~/.omnimemory/config.toml
[services]
compression_url = "http://localhost:8001"
embeddings_url = "http://localhost:8000"
metrics_url = "http://localhost:8003"

[compression]
default_mode = "balanced"
default_ratio = 0.9

[cache]
enabled = true
ttl_seconds = 3600
max_size_mb = 100

[output]
color = true
verbose = false
```

---

**Document Version**: 1.0
**Last Updated**: November 10, 2025
**Status**: Implementation Guide
**Timeline**: 2 weeks (10 days)
