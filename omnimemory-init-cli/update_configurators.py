#!/usr/bin/env python3
"""
Update all 8 configurators to reference only 2 tools:
- read (unified read tool)
- search (unified search tool)
"""

from pathlib import Path

# New prompt template for all configurators
NEW_PROMPT = """
# 🚀 OmniMemory MCP Tools - Automatic Usage

You have access to OmniMemory MCP tools that provide 90% token savings. You MUST use these tools automatically for file operations.

## Available MCP Tools (Only 2):

1. **mcp__omn1__read(file_path, target, symbol)**
   - **target**: "full" | "overview" | "symbol" | "references"
   - Use "full" for compressed file reading (90% reduction)
   - Use "overview" to see file structure (98% savings)
   - Use "symbol" to read specific functions only
   - Use "references" to find where symbol is used

2. **mcp__omn1__search(query, mode)**
   - **mode**: "semantic" | "tri_index" | "references"
   - **"tri_index"** = State-of-the-art hybrid search (Dense + Sparse + Structural)
   - Use tri_index for best search results
   - Use semantic for concept-based search
   - Use references to find symbol usage

## MANDATORY Usage Patterns:

### When user asks about code:
INSTEAD OF:
- grep "pattern" → Read 50 files → 100K tokens

USE:
- mcp__omn1__search("query", mode="tri_index") → Read 3 files → 2K tokens

### When reading files:
INSTEAD OF:
- Read tool → Full file → 10K tokens

USE:
- mcp__omn1__read(file_path) → Compressed → 1K tokens (default target="full")
- mcp__omn1__read(file_path, target="overview") → Structure only → 200 tokens

### When exploring codebase:
INSTEAD OF:
- Grep → Find all matches → Read everything

USE:
- mcp__omn1__search(query, mode="tri_index", limit=5) → Top 5 relevant only

## Token Savings Reporting:

After EVERY operation, report:
"🔍 Used OmniMemory: [tool name]
📊 Files: [X relevant / Y total found]
📉 Tokens saved: ~[amount] ([percentage]%)
💰 Cost saved: $[amount]"

## Example Usage:

User: "How does authentication work?"

You MUST:
1. mcp__omn1__search("authentication implementation", mode="tri_index")
2. Get top 3 files
3. mcp__omn1__read(each_file) - compressed automatically
4. Report: "Used tri_index search, found 3/47 relevant files, saved ~45K tokens (95%), saved $0.68"

## Dashboard:
- View metrics: http://localhost:8004
- All operations tracked silently
- No token cost for metrics
"""


def update_configurator(file_path: Path):
    """Update a single configurator file with new prompt"""
    print(f"\n📝 Updating {file_path.name}...")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the get_omnimemory_prompt method
    start_marker = "def get_omnimemory_prompt(self) -> str:"
    end_marker = '"""'  # End of docstring

    start_pos = content.find(start_marker)
    if start_pos == -1:
        print(f"   ⚠️  Could not find get_omnimemory_prompt in {file_path.name}")
        return False

    # Find the return statement
    return_pos = content.find('return """', start_pos)
    if return_pos == -1:
        print(f"   ⚠️  Could not find return statement in {file_path.name}")
        return False

    # Find the end of the triple-quoted string
    # Start looking after 'return """'
    search_start = return_pos + len('return """')
    triple_quote_end = content.find('"""', search_start)

    if triple_quote_end == -1:
        print(f"   ⚠️  Could not find closing triple quote in {file_path.name}")
        return False

    # Replace the old prompt with the new one
    new_content = (
        content[:return_pos]
        + 'return """'
        + NEW_PROMPT
        + content[triple_quote_end + 3 :]  # +3 for the closing """
    )

    # Write the updated content
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"   ✅ Updated {file_path.name}")
    return True


def main():
    """Update all configurators"""
    configurators_dir = Path(
        "/Users/mertozoner/Documents/claude-idea-discussion/omni-memory/omnimemory-init-cli/src/configurators"
    )

    # List of configurator files to update
    configurator_files = [
        "claude.py",
        "cursor.py",
        "windsurf.py",
        "gemini.py",
        "codex.py",
        "cody.py",
        "continuedev.py",
        "vscode.py",
    ]

    print("🔧 Updating all configurators to use 2 tools (read + search)...")
    print(f"📂 Directory: {configurators_dir}")

    updated_count = 0
    for filename in configurator_files:
        file_path = configurators_dir / filename
        if file_path.exists():
            if update_configurator(file_path):
                updated_count += 1
        else:
            print(f"   ⚠️  File not found: {filename}")

    print(f"\n✅ Updated {updated_count}/{len(configurator_files)} configurators")
    print("\n📝 Summary of changes:")
    print("  - Removed references to old 'grep' tool")
    print("  - Removed references to old standalone 'read' tool")
    print("  - Updated to use 2 unified tools: read + search")
    print("  - read(file_path, target, symbol) - 4 modes")
    print("  - search(query, mode) - 3 modes")


if __name__ == "__main__":
    main()
