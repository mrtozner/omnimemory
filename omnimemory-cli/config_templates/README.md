# OmniMemory Configuration Templates

This directory contains system prompt templates for different AI coding assistants.

## Available Templates

### 1. Passive Mode (Informative)

**File:** `claude_md_snippet.md`

**Use when:** You want to inform the AI about OmniMemory tools but not enforce usage.

**Characteristics:**
- Describes available tools
- Explains benefits
- Suggests usage patterns
- Does NOT mandate usage
- Suitable for exploration/testing

---

### 2. Active Mode (Mandatory)

#### For Claude Code / Claude Desktop

**File:** `claude_md_active_mode.md` (571 lines, 16KB)

**Use when:** You want to FORCE Claude to use memory tools in every interaction.

**Characteristics:**
- ✅ Mandatory memory checks before every response
- ✅ Mandatory memory storage after every response
- ✅ Detailed workflow patterns with examples
- ✅ Strong enforcement language
- ✅ Comprehensive error prevention guide
- ✅ Token savings tracking
- ✅ Advanced usage patterns
- ✅ Compliance checklist

**Installation:**
```bash
# For Claude Code (global)
cat claude_md_active_mode.md >> ~/.claude/CLAUDE.md

# For specific project
cat claude_md_active_mode.md >> .claude/CLAUDE.md
```

---

#### For Cursor IDE

**File:** `cursor_instructions_active_mode.md` (321 lines, 7.2KB)

**Use when:** You want to FORCE Cursor AI to use memory tools (concise version).

**Characteristics:**
- ✅ Mandatory memory protocol (condensed)
- ✅ Clear 3-step workflow
- ✅ Quick examples
- ✅ Essential patterns only
- ✅ Optimized for Cursor's prompt limits
- ✅ Strong enforcement language

**Installation:**
```bash
# For Cursor (Settings → Cursor Settings → Rules for AI)
cat cursor_instructions_active_mode.md
# Then paste into Cursor Settings
```

---

## Comparison: Passive vs Active Mode

| Feature | Passive Mode | Active Mode |
|---------|-------------|-------------|
| **Enforcement** | Optional | Mandatory |
| **Memory checks** | Suggested | Required before every response |
| **Memory storage** | Suggested | Required after every response |
| **Workflow pattern** | Described | Enforced with checklist |
| **Examples** | Few | Comprehensive with walkthroughs |
| **Token tracking** | Not mentioned | Mandatory reporting |
| **Compliance** | Low (AI decides) | High (AI must comply) |
| **Use case** | Exploration, testing | Production, serious usage |

---

## When to Use Which Template

### Use Passive Mode When:
- 🧪 Testing OmniMemory features
- 📚 Learning about available tools
- 🤔 Not sure if you need memory yet
- 🔄 Transitioning from no memory to memory
- 🎓 Training yourself on OmniMemory concepts

### Use Active Mode When:
- 🚀 Production usage (real projects)
- 💰 Maximizing token savings (70-95% reduction)
- 🎯 Need consistent memory usage
- 👤 Want personalized AI responses
- 🔁 Working across multiple sessions
- 📊 Need to track memory performance

---

## Expected Behavior Differences

### Passive Mode (claude_md_snippet.md)
```
User: "Set up a new project"
AI: "What framework would you prefer? React, Vue, or Angular?"
(May or may not check memory - up to AI)
```

### Active Mode (claude_md_active_mode.md)
```
User: "Set up a new project"
AI:
  1. [Calls omn1_search with mode="semantic"]
  2. [Finds: "User prefers React"]
  3. "I'll set up a React project (your preferred framework)..."
  4. [Calls omnimemory_store with project details]
(ALWAYS checks and stores - enforced)
```

---

## Installation Instructions

### Claude Code (Recommended)

**Global (All Projects):**
```bash
cd /Users/mertozoner/Documents/claude-idea-discussion/omni-memory/omnimemory-cli/config_templates
cat claude_md_active_mode.md >> ~/.claude/CLAUDE.md
```

**Project-Specific:**
```bash
cd /path/to/your/project
mkdir -p .claude
cd /Users/mertozoner/Documents/claude-idea-discussion/omni-memory/omnimemory-cli/config_templates
cat claude_md_active_mode.md >> /path/to/your/project/.claude/CLAUDE.md
```

---

### Cursor IDE

1. Open Cursor Settings (Cmd/Ctrl + ,)
2. Go to "Cursor Settings" tab
3. Find "Rules for AI" section
4. Copy content of `cursor_instructions_active_mode.md`
5. Paste into the text area
6. Save settings

---

### VS Code + Cline (Future)

**File:** `vscode_cline_active_mode.md` (Coming soon)

**Installation:**
```bash
# Will be added to Cline extension settings
```

---

### VS Code + Continue (Future)

**File:** `vscode_continue_active_mode.md` (Coming soon)

**Installation:**
```bash
# Will be added to Continue extension config
```

---

## Verification

### After Installing Active Mode

Test that memory is working:

**Test 1: Check memory search**
```
You: "I prefer TypeScript for all projects"
AI: [Should call omnimemory_store]

You: "Set up a new project"
AI: [Should call omn1_search with mode="semantic", find TypeScript preference]
Expected: "I'll set up a TypeScript project (your preference)..."
```

**Test 2: Session continuity**
```
Session 1:
You: "I'm working on a project called MyApp"
AI: [Stores this fact]

Session 2 (next day):
You: "Continue working on my project"
AI: [Retrieves "MyApp" from memory]
Expected: "Continuing with MyApp..."
```

**Test 3: Token savings**
```
Check that AI reports token savings after memory operations:
Expected: "Used semantic search. Prevented 45,000 tokens (95% savings)"
```

---

## Troubleshooting

### AI Not Using Memory

**Symptoms:**
- AI asks for information you already provided
- No `omn1_search` calls visible
- No token savings reported

**Solutions:**
1. Verify template was added to correct config file
2. Restart Claude Code / Cursor
3. Check MCP server is running: `ps aux | grep omnimemory`
4. Try explicitly saying: "Use memory tools as instructed"

---

### Memory Not Persisting

**Symptoms:**
- AI searches but finds no results
- Fresh session has no context

**Solutions:**
1. Check Qdrant is running: `curl http://localhost:6333`
2. Verify storage calls are happening: Check logs
3. Ensure `omnimemory_store` is being called (not just search)

---

### Too Verbose / Too Concise

**Issue:** Active mode templates enforce a style

**Solutions:**
- Edit template to add your style preferences
- Add custom instructions after the template
- Use passive mode if you want more control

---

## Customization

Both active mode templates can be customized:

### Add Your Preferences

```markdown
# After the template content, add:

## Additional User Preferences

- Always use functional programming style
- Keep code comments minimal
- Prefer composition over inheritance
- Maximum function length: 50 lines
```

### Adjust Memory Thresholds

```markdown
# Change default parameters:

omn1_search(
    query="...",
    mode="semantic",
    limit=10,              # Default: 5 (increase for more context)
    min_relevance=0.6      # Default: 0.7 (lower for more results)
)

omnimemory_store(
    content="...",
    importance_threshold=0.8  # Default: 0.7 (higher = more selective)
)
```

### Add Project-Specific Rules

```markdown
# Add after template:

## Project: MyApp (E-commerce Platform)

### Tech Stack
- Next.js 14 (App Router)
- Prisma + PostgreSQL
- Tailwind CSS
- Deployed on Vercel

### Memory Namespace
Always use namespace="myapp" for project-specific memories.
```

---

## Performance Metrics

### Expected Results with Active Mode

| Metric | Value |
|--------|-------|
| **Memory check overhead** | ~100-200ms per request |
| **Memory store overhead** | ~50ms per request (async) |
| **Token savings** | 70-95% vs full conversation |
| **User friction reduction** | 80%+ (fewer repeated questions) |
| **Context retention** | 100% across sessions |
| **Cost savings** | $0.50-$2.00 per extended session |

---

## Support

**Issues:**
- Memory not working? Check MCP server status
- Tokens not being saved? Verify template installation
- AI ignoring instructions? Try active mode templates

**Questions:**
- How memory works: See main OmniMemory docs
- MCP tools reference: Check MCP server documentation
- Advanced usage: See active mode template examples

---

## Version History

- **v1.0** - Initial release
  - `claude_md_snippet.md` (passive mode)

- **v2.0** - Active mode templates (2025-01-12)
  - `claude_md_active_mode.md` (Claude Code - comprehensive)
  - `cursor_instructions_active_mode.md` (Cursor - concise)
  - Strong enforcement of memory protocol
  - Detailed examples and workflows
  - Token savings tracking
  - Compliance checklists

- **v2.1** (Planned)
  - `vscode_cline_active_mode.md`
  - `vscode_continue_active_mode.md`

---

## Summary

- **Passive Mode**: Optional memory usage, good for learning
- **Active Mode**: Mandatory memory usage, best for production
- **Claude Code**: Use comprehensive template (16KB)
- **Cursor**: Use concise template (7.2KB)
- **Expected Impact**: 70-95% token savings, better UX

Choose active mode for serious usage. Choose passive mode for exploration.
