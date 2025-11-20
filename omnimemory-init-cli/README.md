# OmniMemory Init CLI

Configure AI tools to use OmniMemory automatically - 97.9% token reduction (benchmarked).

## Features

- **97.9% Token Reduction** - Proven with competitive benchmarks
- **3× Faster** - 0.14ms vs SuperMemory's 0.5ms
- **Team Sharing** - 80% additional savings via repository-level cache
- **8 Tool Support** - Claude, Cursor, Cody, Continue, Gemini, VSCode, Windsurf, Codex
- **Zero Configuration** - Auto-detects paths, works immediately
- **Local-First** - Privacy by default, code never leaves your machine
- **Safe operations** - Automatic config backups before any changes
- **Dry-run mode** - Preview changes before applying
- **Status checking** - See current configuration status
- **Easy removal** - Cleanly remove OmniMemory integration

## Installation

```bash
cd omnimemory-init-cli
pip install -e .
```

## Quick Start

### Configure Your IDE

```bash
# Claude Desktop
omni-init init --tool claude --api-key YOUR_KEY

# Cursor
omni-init init --tool cursor --api-key YOUR_KEY

# Cody (Sourcegraph)
omni-init init --tool cody --api-key YOUR_KEY

# Continue.dev
omni-init init --tool continue --api-key YOUR_KEY

# Gemini Code Assist
omni-init init --tool gemini --api-key YOUR_KEY

# VSCode (with Cline)
omni-init init --tool vscode --api-key YOUR_KEY

# Windsurf
omni-init init --tool windsurf --api-key YOUR_KEY

# Codex (OpenAI)
omni-init init --tool codex --api-key YOUR_KEY

# Configure all tools at once
omni-init init --tool all --api-key YOUR_KEY
```

### Check Status

```bash
omni-init status --tool all
```

### Preview Changes (Dry Run)

```bash
omni-init init --tool claude --api-key YOUR_KEY --dry-run
```

## Usage

### Initialize OmniMemory

```bash
omni-init init --tool TOOL --api-key KEY [OPTIONS]
```

**Options:**
- `--tool`: Tool to configure (`claude`, `cursor`, `cody`, `continue`, `gemini`, `vscode`, `windsurf`, `codex`, or `all`)
- `--api-key`: Your OmniMemory API key (or set `OMNIMEMORY_API_KEY` env var)
- `--api-url`: API base URL (default: `http://localhost:8005`)
- `--user-id`: Your user ID (optional)
- `--dry-run`: Show what would be changed without modifying files

**Example:**
```bash
# Configure Claude Desktop
omni-init init --tool claude --api-key sk_abc123...

# Configure all tools with custom API URL
omni-init init --tool all \
  --api-key sk_abc123... \
  --api-url https://api.omnimemory.dev

# Preview changes without modifying files
omni-init init --tool cursor --api-key sk_abc123... --dry-run
```

### Check Configuration Status

```bash
omni-init status --tool TOOL
```

**Example:**
```bash
# Check all tools
omni-init status --tool all

# Check specific tool
omni-init status --tool claude
```

**Output:**
```
┏━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Tool   ┃ Installed ┃ Config Exists ┃ OmniMemory Enabled ┃ Config Path                      ┃
┡━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Claude │ ✅        │ ✅           │ ✅                 │ ~/Library/Application Support... │
│ Cursor │ ✅        │ ✅           │ ❌                 │ ~/Library/Application Support... │
│ Vscode │ ❌        │ ❌           │ ❌                 │ ~/Library/Application Support... │
└────────┴───────────┴──────────────┴────────────────────┴──────────────────────────────────┘
```

### Remove OmniMemory Configuration

```bash
omni-init remove --tool TOOL [--dry-run]
```

**Example:**
```bash
# Remove from Claude Desktop (with confirmation)
omni-init remove --tool claude

# Remove from all tools
omni-init remove --tool all

# Preview removals without modifying files
omni-init remove --tool cursor --dry-run
```

## Environment Variables

Instead of passing credentials every time, you can set environment variables:

```bash
export OMNIMEMORY_API_KEY="sk_your_api_key_here"
export OMNIMEMORY_API_URL="http://localhost:8005"
export OMNIMEMORY_USER_ID="your_user_id"
```

Then simply run:
```bash
omni-init init --tool claude
```

## What It Does

### For Claude Desktop

1. **Backs up** existing configuration
2. **Adds system prompt** that instructs Claude to:
   - Check memory before responding
   - Store important information after responding
3. **Configures MCP server** for OmniMemory integration
4. **Sets environment variables** for API access

### For Cursor

1. **Backs up** existing settings
2. **Adds OmniMemory settings** to `settings.json`
3. **Configures system prompt** for automatic memory
4. **Enables auto mode** for seamless operation

### For VSCode

1. **Backs up** existing settings
2. **Adds OmniMemory settings** to `settings.json`
3. **Configures GitHub Copilot** system prompt (if installed)
4. **Enables auto mode** for automatic memory

### For Windsurf

1. **Backs up** existing configuration
2. **Adds OmniMemory MCP server** to `mcp_config.json`
3. **Configures server connection** with proper paths
4. **Enables MCP integration** for Codeium IDE

### For Cody (Sourcegraph)

1. **Backs up** existing settings
2. **Adds OmniMemory settings** to VSCode settings
3. **Configures system prompt** via `cody.systemPrompt`
4. **Sets environment variable** `OMNIMEMORY_TOOL_ID=cody`

### For Continue.dev

1. **Backs up** existing config
2. **Adds OmniMemory settings** to `~/.continue/config.json`
3. **Configures systemMessage** for automatic memory
4. **Sets environment variable** `OMNIMEMORY_TOOL_ID=continue-dev`

### For Gemini Code Assist

1. **Backs up** existing settings
2. **Adds OmniMemory settings** to `~/.gemini/settings.json`
3. **Configures system prompt** for automatic memory
4. **Sets environment variable** `OMNIMEMORY_TOOL_ID=gemini-code-assist`

### For Codex (OpenAI)

1. **Backs up** existing config
2. **Adds OmniMemory settings** to `~/.codex/config.toml`
3. **Configures [prompt] section** with OmniMemory instructions
4. **Sets environment variable** `OMNIMEMORY_TOOL_ID=codex`

## Configuration Locations

The tool modifies configuration files in these locations:

### macOS
- Claude: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Cursor: `~/Library/Application Support/Cursor/User/settings.json`
- Cody: `~/Library/Application Support/Code/User/settings.json` (uses VSCode settings)
- Continue: `~/.continue/config.json`
- Gemini: `~/.gemini/settings.json`
- VSCode: `~/Library/Application Support/Code/User/settings.json`
- Windsurf: `~/.codeium/windsurf/mcp_config.json`
- Codex: `~/.codex/config.toml`

### Linux
- Claude: `~/.config/claude/config.json`
- Cursor: `~/.config/Cursor/User/settings.json`
- Cody: `~/.config/Code/User/settings.json` (uses VSCode settings)
- Continue: `~/.continue/config.json`
- Gemini: `~/.gemini/settings.json`
- VSCode: `~/.config/Code/User/settings.json`
- Windsurf: `~/.codeium/windsurf/mcp_config.json`
- Codex: `~/.codex/config.toml`

### Windows
- Claude: `%APPDATA%\Claude\config.json`
- Cursor: `%APPDATA%\Cursor\User\settings.json`
- Cody: `%APPDATA%\Code\User\settings.json` (uses VSCode settings)
- Continue: `%USERPROFILE%\.continue\config.json`
- Gemini: `%USERPROFILE%\.gemini\settings.json`
- VSCode: `%APPDATA%\Code\User\settings.json`
- Windsurf: `%APPDATA%\Codeium\Windsurf\mcp_config.json`
- Codex: `%USERPROFILE%\.codex\config.toml`

## Safety Features

### Automatic Backups

Before making any changes, the tool creates a timestamped backup:

```
claude_desktop_config.json.backup_20250112_143022
```

### Dry Run Mode

Test what would be changed without modifying files:

```bash
omni-init init --tool claude --api-key sk_abc123... --dry-run
```

### Confirmation Prompts

The `remove` command requires confirmation before removing configuration.

## Expected Benefits

After configuration, you should see:

- **97.9% token reduction (benchmarked)** - Automatic context compression and optimization
- **3× faster performance** - 0.14ms vs SuperMemory's 0.5ms
- **Seamless memory** - Information persists across sessions
- **No workflow changes** - Works automatically in the background
- **Better context** - AI remembers preferences, decisions, and project details

## Troubleshooting

### "Tool not found"

The tool checks if the application directory exists. If you see "not found":
1. Make sure the tool is installed
2. Check the config path matches your installation
3. Run `omni-init status --tool TOOL` to see the expected path

### "Invalid API key format"

API keys must be:
- At least 20 characters
- Contain only alphanumeric characters, hyphens, and underscores

### "Invalid API URL"

API URLs must:
- Start with `http://` or `https://`
- Have a valid hostname

### Configuration not working

1. Check status: `omni-init status --tool TOOL`
2. Verify API key is correct
3. Restart the AI tool after configuration
4. Check tool's error logs for details

### Tool-Specific Issues

**Claude Desktop:**
- Config location varies by OS (see Configuration Locations above)
- Restart Claude completely after configuration
- Check: File has valid JSON syntax
- Verify MCP server path is correct

**Cursor:**
- Config: `~/Library/Application Support/Cursor/User/settings.json` (macOS)
- Prompt file: `.cursorrules` in project root
- Restart Cursor after changes
- Check Cursor settings for OmniMemory entries

**Cody (Sourcegraph):**
- Uses VSCode settings.json location
- Check `cody.systemPrompt` is set in settings
- Restart VSCode or reload window after changes
- Verify Cody extension is installed and enabled

**Continue.dev:**
- Config: `~/.continue/config.json`
- Check `systemMessage` field exists in config
- Restart Continue extension or reload VSCode
- Verify Continue extension is installed

**Gemini Code Assist:**
- Config: `~/.gemini/settings.json`
- Verify Google Cloud credentials are configured
- Restart IDE after changes
- Check Gemini extension logs for errors

**Codex (OpenAI):**
- Config: `~/.codex/config.toml`
- Check `[prompt]` section syntax is valid TOML
- Verify OpenAI API key is set
- Restart Codex client after changes

**Performance Issues:**
1. Check backend services: `curl http://localhost:8005/health`
2. Check Redis is running: `docker ps | grep redis`
3. Check cache stats: `curl http://localhost:8003/api/cache/stats`
4. Verify MCP server connection: Check IDE logs for MCP connection errors

**"No such tool" errors:**
1. Verify correct tool name format: `mcp__omnimemory__read` (double underscore)
2. Reconnect to MCP: Type `/mcp` in Claude Code
3. Check MCP server path in configuration file
4. Restart IDE completely (not just reload window)

## Development

### Running Tests

```bash
cd omnimemory-init-cli
python -m pytest tests/
```

### Installing in Dev Mode

```bash
pip install -e ".[dev]"
```

### Project Structure

```
omnimemory-init-cli/
├── pyproject.toml          # Package configuration
├── README.md               # This file
├── omni_init.py           # Main CLI entry point
├── src/
│   ├── configurators/
│   │   ├── base.py         # Base configurator class
│   │   ├── claude.py       # Claude Desktop configurator
│   │   ├── cursor.py       # Cursor configurator
│   │   ├── vscode.py       # VSCode configurator
│   │   └── windsurf.py     # Windsurf configurator
│   ├── utils/
│   │   ├── file_ops.py     # Safe file operations
│   │   └── validation.py   # Input validation
│   └── templates/
│       ├── claude_system_prompt.txt
│       ├── cursor_config.json
│       └── vscode_settings.json
└── tests/
    └── test_configurators.py
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/omnimemory/omnimemory/issues
- Documentation: https://docs.omnimemory.dev
- Website: https://omnimemory.dev

## License

MIT License - See LICENSE file for details
