# 🚀 OmniMemory MCP Quick Start

> **🔄 Updated for OMN1 Consolidation (2025)**: This guide uses the new consolidated tools (`omn1_read`, `omn1_search`, etc.) which replace the previous tools. See [OMN1 Migration Guide](../docs/OMN1_MIGRATION_GUIDE.md) for details on the consolidation.

## ✅ Installation Complete!

Your MCP server is installed, tested, and configured!

---

## 🎯 What Just Happened

1. ✅ Installed MCP server with 30 dependencies
2. ✅ Tested all 8 tools successfully
3. ✅ Created Claude Code config at `~/.config/claude/config.json`
4. ✅ All OmniMemory services running (ports 8000-8002)

---

## 🔄 Final Step: Restart Claude Code

**You need to restart Claude Code for it to load the MCP server.**

### How to Restart:
1. Close your current terminal/Claude Code session
2. Open a new terminal
3. Start Claude Code again

---

## ✓ Verify It Worked

After restarting, type:
```bash
/mcp list
```

**Expected output:**
```
Available MCP servers:
  omnimemory
    Core Tools (4):
      - read (compressed file reading)
      - grep (semantic-enhanced pattern search)
      - omn1_read (universal file reader with modes)
      - omn1_search (universal search: semantic/tri_index/references)

    Supporting Tools:
      - omn1_compress
      - omn1_workflow_context
      - omn1_resume_workflow
      - omn1_checkpoint_conversation
      - omn1_search_checkpoints_semantic
      - omn1_get_stats
```

---

## 💡 How to Use

### Automatic Mode (Recommended)

Claude will automatically use OmniMemory tools when appropriate.

**Example conversations:**

**1. Token Compression**
```
You: I have a really long context to share [paste 5000 words]

Claude: Let me compress this context to save tokens...
[Automatically calls omn1_compress]
Claude: Compressed from 12,000 to 672 tokens (94.4% savings)
```

**2. Semantic Code Search**
```
You: Find all authentication error handling in the codebase

Claude: [Calls omn1_search with mode="semantic"]
Claude: [Searches using semantic similarity]
Claude: Found 8 patterns across services/, auth/, utils/...
```

**2b. High-Precision Tri-Index Search**
```
You: Find authentication code with high precision

Claude: [Calls omn1_search with mode="tri_index"]
Claude: [Searches Dense (vectors) + Sparse (BM25) + Structural (code facts)]
Claude: [Fused with RRF + witness reranking]
Claude: Found 5 highly relevant files with precise matches
```

**3. File Structure Overview**
```
You: Show me the structure of auth.py

Claude: [Calls omn1_read with target="overview"]
Claude: Shows classes, functions, imports (98% token savings)
```

**4. Workflow Learning**
```
[After you debug 3 issues the same way]

Claude: I notice you typically: read logs → check DB → run tests
Claude: Would you like me to suggest these automatically?
```

### Manual Mode

You can explicitly ask Claude to use tools:

```
You: Use omn1_compress to compress this text for me
You: Search for similar code with omn1_search
You: Show me OmniMemory stats with omn1_get_stats
```

---

## 📊 Watch It Work

Open the dashboard to see metrics update in real-time:

```bash
open http://localhost:8050
```

**You'll see:**
- Token savings incrementing
- Compression ratios improving  
- Cache hit rates increasing
- Workflow patterns being learned
- ROI calculation ($ saved)

---

## 🧪 Test Right Now

Try this in your next Claude Code session:

```
You: Use omn1_get_stats to show me the current statistics
```

Claude should call the tool and show you:
- Embeddings: total generated, cache hit rate
- Compression: tokens saved, compression ratio
- Procedural: patterns learned

---

## 🐛 Troubleshooting

### MCP Server Not Showing Up

```bash
# 1. Check config syntax
cat ~/.config/claude/config.json | python -m json.tool

# 2. Check paths exist
ls /Users/mertozoner/Documents/claude-idea-discussion/omni-memory/mcp_server/.venv/bin/python
ls /Users/mertozoner/Documents/claude-idea-discussion/omni-memory/mcp_server/omnimemory_mcp.py

# 3. Test server manually
cd /Users/mertozoner/Documents/claude-idea-discussion/omni-memory/mcp_server
./test_mcp_server.py
```

### Services Not Running

```bash
# Check health
curl http://localhost:8000/health  # Embeddings
curl http://localhost:8001/health  # Compression  
curl http://localhost:8002/health  # Procedural

# If any fail, restart from omnimemory-* directories:
cd omnimemory-embeddings && uv run src/api_server.py &
cd omnimemory-compression && uv run src/compression_server.py &
cd omnimemory-procedural && uv run src/procedural_server.py &
```

### Tools Not Working

Check Claude Code output for errors. The MCP server logs will appear there.

---

## 📚 Documentation

- `installation_complete.md` - Full setup guide
- `INTEGRATION_GUIDE.md` - Complete integration docs
- `mcp_server/README.md` - Technical reference
- `QUICKSTART.md` - This file

---

## 🎯 What's Next

After you verify it works:

1. **Use it naturally** - Claude will call tools automatically
2. **Watch the dashboard** - See metrics update live
3. **Try the examples** - Test compression, embeddings, workflows
4. **Share feedback** - See what works, what doesn't

---

## 🎉 Success Checklist

- [✅] MCP server installed
- [✅] Config created
- [✅] All services running
- [ ] Claude Code restarted
- [ ] `/mcp list` shows omnimemory
- [ ] Tested a tool call
- [ ] Dashboard shows metrics

**Last 3 items need YOU to complete after restart!**

---

## 💪 You're Ready!

The hard part is done. Just restart Claude Code and you're all set.

Claude will now automatically:
- Compress long contexts (save 94% tokens)
- Use cached embeddings (faster semantic search)
- Learn your workflow patterns (suggest next steps)

**See you after the restart! 🚀**
