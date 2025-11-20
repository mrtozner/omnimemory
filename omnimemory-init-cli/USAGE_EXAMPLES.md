# OmniMemory Init CLI - Usage Examples

## Quick Start

### 1. Install the CLI

```bash
cd omnimemory-init-cli
pip install -e .
```

### 2. Set Environment Variables (Optional)

```bash
export OMNIMEMORY_API_KEY="sk_your_api_key_here"
export OMNIMEMORY_API_URL="http://localhost:8005"
export OMNIMEMORY_USER_ID="your_user_id"
```

### 3. Configure Your Tool

```bash
# Using environment variables
python3 omni_init.py init --tool claude

# Or pass credentials directly
python3 omni_init.py init --tool claude --api-key sk_abc123...
```

## Common Scenarios

### Scenario 1: First-Time Setup for Claude Desktop

```bash
# Check current status
python3 omni_init.py status --tool claude

# Preview changes (dry run)
python3 omni_init.py init --tool claude --api-key sk_abc123... --dry-run

# Apply configuration
python3 omni_init.py init --tool claude --api-key sk_abc123...

# Restart Claude Desktop
# macOS: Quit and reopen Claude Desktop
# Linux: killall claude && claude
```

**Expected output:**
```
╭─────────────────────────────────────────╮
│ OmniMemory Init CLI                     │
│ Configure AI tools for automatic memory │
╰─────────────────────────────────────────╯


🔧 Configuring Claude...
✓ Backed up config to: /Users/you/Library/Application Support/Claude/claude_desktop_config.json.backup_20250112_143022

📝 Changes for claude:
  • Added automatic memory system prompt
  • Added OmniMemory MCP server

✅ Claude configured successfully!
   Restart Claude to activate OmniMemory.

╭──────────────────────────────────────────────────────╮
│ ✅ Configuration Complete!                           │
│                                                      │
│ OmniMemory is now configured for your AI tools.     │
│ Restart your tools to activate automatic memory.    │
│                                                      │
│ Expected benefits:                                   │
│ • 70-85% reduction in API costs                      │
│ • Automatic context preservation                    │
│ • Seamless cross-session memory                     │
╰──────────────────────────────────────────────────────╯
```

### Scenario 2: Configure All Tools at Once

```bash
# Check what tools are installed
python3 omni_init.py status --tool all

# Configure all installed tools
python3 omni_init.py init --tool all --api-key sk_abc123...
```

**Expected output:**
```
🔧 Configuring Claude...
⚠️  Claude not found. Skipping.

🔧 Configuring Cursor...
✓ Backed up config to: ~/.../settings.json.backup_20250112_143022

📝 Changes for cursor:
  • Set omnimemory.enabled
  • Set omnimemory.apiKey
  • Set omnimemory.apiUrl
  • Set omnimemory.userId
  • Set omnimemory.autoMode
  • Set cursor.chat.systemPrompt
  • Set omnimemory.searchBeforeResponse
  • Set omnimemory.storeAfterResponse

✅ Cursor configured successfully!
   Restart cursor to activate OmniMemory.

🔧 Configuring Vscode...
✓ Backed up config to: ~/.../settings.json.backup_20250112_143022
...
```

### Scenario 3: Custom API URL (Self-Hosted)

```bash
# Using custom OmniMemory instance
python3 omni_init.py init \
  --tool claude \
  --api-key sk_abc123... \
  --api-url https://omnimemory.yourcompany.com \
  --user-id john_doe
```

### Scenario 4: Check Configuration Status

```bash
# Check specific tool
python3 omni_init.py status --tool cursor

# Check all tools
python3 omni_init.py status --tool all
```

**Expected output:**
```
╭─────────────────────────╮
│ OmniMemory Status Check │
╰─────────────────────────╯

┏━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Tool   ┃ Installed ┃ Config Exists ┃ OmniMemory Enabled ┃ Config Path        ┃
┡━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ Claude │ ✅        │ ✅            │ ✅                 │ ~/Library/...      │
│ Cursor │ ✅        │ ✅            │ ❌                 │ ~/Library/...      │
│ Vscode │ ✅        │ ✅            │ ❌                 │ ~/Library/...      │
└────────┴───────────┴───────────────┴────────────────────┴────────────────────┘
```

### Scenario 5: Remove OmniMemory (Clean Uninstall)

```bash
# Preview what would be removed (dry run)
python3 omni_init.py remove --tool claude --dry-run

# Remove configuration (with confirmation)
python3 omni_init.py remove --tool claude

# Remove from all tools
python3 omni_init.py remove --tool all
```

**Expected output:**
```
╭────────────────────────────────────╮
│ Remove OmniMemory Configuration    │
╰────────────────────────────────────╯
Are you sure you want to remove OmniMemory configuration? [y/N]: y

🗑️  Removing from Claude...
✓ Backed up config to: .../claude_desktop_config.json.backup_20250112_143022

📝 Removals for claude:
  • Removed OmniMemory MCP server
  • Removed OmniMemory system prompt

✅ Removed from Claude
```

### Scenario 6: Dry Run to Preview Changes

```bash
# Preview changes without modifying files
python3 omni_init.py init --tool cursor --api-key sk_abc123... --dry-run

# Preview removal without modifying files
python3 omni_init.py remove --tool cursor --dry-run
```

**Expected output:**
```
🔍 DRY RUN MODE - No files will be modified

🔧 Configuring Cursor...

📝 Changes for cursor:
  • Set omnimemory.enabled
  • Set omnimemory.apiKey
  • Set omnimemory.apiUrl
  • Set omnimemory.userId
  • Set omnimemory.autoMode
  • Set cursor.chat.systemPrompt
  • Set omnimemory.searchBeforeResponse
  • Set omnimemory.storeAfterResponse

🔍 Dry run - no files modified
```

## Testing the Configuration

After configuring your tool, test that OmniMemory is working:

### For Claude Desktop

1. Restart Claude Desktop
2. Open a new conversation
3. Ask: "Do you have memory enabled?"
4. Claude should mention OmniMemory or the search_memory tool

### For Cursor

1. Restart Cursor
2. Open a new chat
3. Check settings: `Cmd+,` → Search "omnimemory"
4. Verify settings are present

### For VSCode

1. Restart VSCode
2. Open settings: `Cmd+,` → Search "omnimemory"
3. Verify settings are present
4. Test with GitHub Copilot chat (if installed)

## Troubleshooting

### Issue: "Invalid API key format"

```bash
# API keys must be at least 20 characters
# and contain only alphanumeric, hyphens, underscores

# ❌ Wrong
--api-key short

# ✅ Correct
--api-key sk_abcdefghij1234567890
```

### Issue: "Tool not found"

```bash
# Check if tool is installed
python3 omni_init.py status --tool claude

# If status shows "Installed: ❌", install the tool first
```

### Issue: Configuration not taking effect

```bash
# 1. Verify configuration
python3 omni_init.py status --tool claude

# 2. Check the config file directly
# macOS Claude:
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json

# macOS Cursor:
cat ~/Library/Application\ Support/Cursor/User/settings.json

# 3. Restart the tool completely
# Make sure to quit, not just close window
```

### Issue: Need to revert changes

```bash
# 1. Find the backup file
ls -la ~/Library/Application\ Support/Claude/

# Look for: claude_desktop_config.json.backup_TIMESTAMP

# 2. Restore from backup
cp ~/Library/Application\ Support/Claude/claude_desktop_config.json.backup_20250112_143022 \
   ~/Library/Application\ Support/Claude/claude_desktop_config.json

# 3. Restart the tool
```

## Advanced Usage

### Using Environment Files

Create a `.env` file:

```bash
# .env
OMNIMEMORY_API_KEY=sk_your_api_key_here
OMNIMEMORY_API_URL=http://localhost:8005
OMNIMEMORY_USER_ID=your_user_id
```

Load it:
```bash
# Bash/Zsh
source .env
python3 omni_init.py init --tool claude

# Or use direnv
echo "source .env" > .envrc
direnv allow
python3 omni_init.py init --tool claude
```

### Scripted Deployment

```bash
#!/bin/bash
# deploy_omnimemory.sh

set -e

API_KEY="sk_your_api_key_here"
API_URL="http://localhost:8005"

echo "Deploying OmniMemory to all tools..."

# Configure all tools
python3 omni_init.py init \
  --tool all \
  --api-key "$API_KEY" \
  --api-url "$API_URL"

echo "Deployment complete!"
echo "Please restart your AI tools."
```

### CI/CD Integration

```yaml
# .github/workflows/deploy-omnimemory.yml
name: Deploy OmniMemory

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v2

      - name: Install OmniMemory Init CLI
        run: |
          cd omnimemory-init-cli
          pip install -e .

      - name: Configure Tools
        env:
          OMNIMEMORY_API_KEY: ${{ secrets.OMNIMEMORY_API_KEY }}
        run: |
          python3 omni_init.py init --tool all
```

## Next Steps

After configuring OmniMemory:

1. **Test it**: Have a conversation, close the tool, reopen, and ask about previous conversation
2. **Monitor costs**: Check your AI provider dashboard for reduced API usage
3. **Customize**: Edit system prompts in the config files if needed
4. **Share**: Deploy to your team using the scripted approach

## Support

- Documentation: https://docs.omnimemory.dev
- Issues: https://github.com/omnimemory/omnimemory/issues
- Website: https://omnimemory.dev
