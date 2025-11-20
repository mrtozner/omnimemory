"""Gemini CLI configurator for OMN1 MCP integration."""

import json
import platform
from pathlib import Path
from datetime import datetime
from typing import Optional


class GeminiConfigurator:
    """Configure Gemini CLI for OMN1 MCP server."""

    def __init__(self, venv_python: str, mcp_script: str):
        """
        Initialize configurator with MCP server paths.

        Args:
            venv_python: Absolute path to venv python executable
            mcp_script: Absolute path to MCP server script
        """
        self.venv_python = venv_python
        self.mcp_script = mcp_script

    def get_config_path(self) -> Path:
        """
        Get Gemini CLI config path.

        Returns:
            Path to Gemini CLI settings file
        """
        # Same path for all platforms
        return Path.home() / ".gemini" / "settings.json"

    def get_omnimemory_prompt(self) -> str:
        """
        Get the OmniMemory usage prompt to embed in Gemini config.

        Returns:
            String containing the complete prompt text
        """
        return """
# 🚨 OMNIMEMORY ENFORCEMENT - Read This First

## CRITICAL RULE: Always Try OmniMemory First

You have OmniMemory MCP tools that save 90%+ tokens. **Not using them wastes user's money.**

### MANDATORY Tool Usage Order:

**For ANY file read**:
1. ✅ FIRST: Try `mcp__omnimemory__read("file.py|overview")`
2. ❌ NEVER: Use standard Read without trying OmniMemory first
3. ✅ FALLBACK: Only use Read if OmniMemory errors

**For ANY code search**:
1. ✅ FIRST: Try `mcp__omnimemory__search("query|tri_index")`
2. ❌ NEVER: Use Grep without trying OmniMemory first
3. ✅ FALLBACK: Only use Grep if OmniMemory errors

**For ANY codebase exploration**:
1. ✅ FIRST: Try `mcp__omnimemory__search("query|tri_index|limit:5")`
2. ❌ NEVER: Use Task/Grep directly
3. ✅ FALLBACK: Only if OmniMemory unavailable

---

## Why This Matters (User Cost Impact)

**Without OmniMemory** (if you skip it):
- Reading 10 files: 50,000 tokens = $0.75
- User pays full price

**With OmniMemory** (if you use it):
- Reading 10 files: 3,000 tokens = $0.045
- User saves $0.705 (94%)

**Your responsibility**: Don't waste user's money by forgetting to use OmniMemory.

---

## Available MCP Tools (Delimiter-Based Syntax):

1. **mcp__omnimemory__read(input_string)** - Unified reading
   Format: `"file_path|mode|options"`

   Examples:
   - `mcp__omnimemory__read("file.py")` → Full compressed (90% savings)
   - `mcp__omnimemory__read("file.py|overview")` → Structure only (98% savings)
   - `mcp__omnimemory__read("file.py|symbol:Settings")` → Specific symbol (99% savings)
   - `mcp__omnimemory__read("file.py|references:func")` → Find usages

   Options: `|details`, `|nocompress`, `|lang:python`

2. **mcp__omnimemory__search(input_string)** - Unified search
   Format: `"query|mode|options"`

   Examples:
   - `mcp__omnimemory__search("authentication")` → Semantic search (default)
   - `mcp__omnimemory__search("authentication|tri_index")` → Hybrid search (BEST)
   - `mcp__omnimemory__search("auth|tri_index|limit:10")` → With limit
   - `mcp__omnimemory__search("Settings|references:SettingsManager")` → Find references

   Options: `|limit:N`, `|minrel:0.8`, `|file:path`

## MANDATORY Usage Patterns:

### When user asks about code:
INSTEAD OF:
- grep "pattern" → Read 50 files → 100K tokens

USE:
- mcp__omnimemory__search("query|tri_index") → Read 3 files → 2K tokens

### When reading files:
INSTEAD OF:
- Read tool → Full file → 10K tokens

USE:
- mcp__omnimemory__read("file.py") → Compressed → 1K tokens (default mode)
- mcp__omnimemory__read("file.py|overview") → Structure only → 200 tokens

### When exploring codebase:
INSTEAD OF:
- Grep → Find all matches → Read everything

USE:
- mcp__omnimemory__search("query|tri_index|limit:5") → Top 5 relevant only

## Token Savings Reporting:

After EVERY operation, report:
"🔍 Used OmniMemory: [tool name]
📊 Files: [X relevant / Y total found]
📉 Tokens saved: ~[amount] ([percentage]%)
💰 Cost saved: $[amount]"

## Example Usage:

User: "How does authentication work?"

You MUST:
1. mcp__omnimemory__search("authentication implementation|tri_index")
2. Get top 3 files
3. mcp__omnimemory__read(each_file) - compressed automatically
4. Report: "Used tri_index search, found 3/47 relevant files, saved ~45K tokens (95%), saved $0.68"

## Dashboard:
- View metrics: http://localhost:8004
- All operations tracked silently
- No token cost for metrics
"""

    def configure(self) -> Path:
        """
        Configure Gemini CLI with MCP server and embed OmniMemory prompts.

        Returns:
            Path to config file that was modified
        """
        config_path = self.get_config_path()

        # Create backup if file exists
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)

                # Create backup
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                backup_path = config_path.with_suffix(f".json.backup-{timestamp}")
                with open(backup_path, "w") as f:
                    json.dump(config, f, indent=2)
                print(f"✅ Created backup: {backup_path}")
            except Exception as e:
                print(f"Warning: Could not create backup: {e}")
                config = {}
        else:
            config = {}

        # Ensure mcpServers key exists
        if "mcpServers" not in config:
            config["mcpServers"] = {}

        # Add OMN1 MCP server
        config["mcpServers"]["omn1"] = {
            "command": self.venv_python,
            "args": [self.mcp_script],
            "env": {
                "OMNIMEMORY_TOOL_ID": "gemini-code-assist"  # Identifies which tool is using OmniMemory
            },
            "timeout": 60000,
            "trust": False,
        }

        # Embed OmniMemory prompt in Gemini's systemPrompt field
        config["systemPrompt"] = self.get_omnimemory_prompt()

        # Write config
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"✅ Embedded OmniMemory prompts in {config_path}")
        return config_path
