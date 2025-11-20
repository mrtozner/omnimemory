# OmniMemory Development Guide

## 🚀 Quick Start for Local Development

### **Option 1: Auto-Reload Enabled (Recommended)**

```bash
# Start with auto-reload
./dev.sh
```

✅ Code changes automatically restart the server
✅ No API key required
✅ Perfect for active development

### **Option 2: Manual Start**

```bash
# Start normally (requires manual restart after changes)
python3 -m src.compression_server
```

⚠️ Requires manual restart after code changes
✅ No API key required
✅ Use for production-like testing

---

## 🔑 API Key Behavior

### **Local Development (localhost)**

**No API key needed!** The server automatically bypasses authentication for local development.

```python
# ✅ Works without Authorization header
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8001/compress",
        json={"context": "Your text..."}
    )
```

### **Production Deployment**

API key **required** when accessed from external hosts:

```python
# ✅ Requires Authorization header
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://production-url:8001/compress",
        json={"context": "Your text..."},
        headers={"Authorization": "Bearer om_pro_..."}
    )
```

---

## 🔄 When Does Code Auto-Update?

### **Auto-Reload Enabled (ENV=development)**

| Change Type | Auto-Update? | Notes |
|-------------|--------------|-------|
| Python code changes (`.py`) | ✅ YES | Server restarts automatically |
| Config changes (`config.py`) | ✅ YES | Takes effect on restart |
| New files added | ✅ YES | Detected and loaded |
| Dependency changes (`requirements.txt`) | ❌ NO | Run `pip install -r requirements.txt` |
| Database schema changes | ❌ NO | Delete `~/.omnimemory/*.db` to recreate |

### **Auto-Reload Disabled (Production)**

| Change Type | Auto-Update? | Action Required |
|-------------|--------------|-----------------|
| Any Python changes | ❌ NO | Manually restart server |
| Config changes | ❌ NO | Restart + redeploy |
| Database changes | ❌ NO | Run migrations manually |

---

## 📝 Development Workflow

### **Making Code Changes**

```bash
# 1. Start development server (auto-reload enabled)
./dev.sh

# 2. Edit code in your IDE
# src/compression_server.py, src/visiondrop.py, etc.

# 3. Save file
# ✅ Server automatically restarts!

# 4. Test changes immediately
curl -X POST http://localhost:8001/compress \
  -H "Content-Type: application/json" \
  -d '{"context": "Test text"}'
```

### **Testing Without Restart**

Some changes don't require server restart:

```bash
# Test SDK changes (doesn't affect running server)
cd sdk
python examples/basic_usage.py

# Test integrations (doesn't affect running server)
cd integrations/langchain
python examples/langchain_example.py
```

---

## 🛠️ Common Development Tasks

### **Add New Dependencies**

```bash
# 1. Add to requirements.txt
echo "new-package>=1.0.0" >> requirements.txt

# 2. Install
pip install -r requirements.txt

# 3. Restart server (even with auto-reload)
# Press Ctrl+C, then ./dev.sh
```

### **Reset Database (Clean Slate)**

```bash
# Remove all stored data
rm ~/.omnimemory/*.db

# Server will recreate on next start
./dev.sh
```

### **Test API Authentication**

```bash
# Create test API key
python -m src.admin_cli create-key --user-id test --tier free

# Use the key (from external request simulation)
curl -X POST http://localhost:8001/compress \
  -H "Authorization: Bearer om_free_..." \
  -H "Content-Type: application/json" \
  -d '{"context": "Test"}'
```

---

## 🔍 Debugging Tips

### **Check Server Logs**

Auto-reload shows detailed logs:

```
INFO:     Will watch for changes in these directories: ['/path/to/omnimemory-compression']
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using WatchFiles
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     ✓ OmniTokenizer initialized
INFO:     ✓ ThreeTierCache initialized
INFO:     ✓ CompressionValidator initialized
INFO:     ✓ VisionDropCompressor initialized
INFO:     🚀 All services ready!
```

### **File Change Detection**

When you save a file, you'll see:

```
INFO:     WatchFiles detected changes in 'src/compression_server.py'. Reloading...
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [12346]
INFO:     Started server process [12347]
```

### **Common Issues**

**1. "Address already in use" error**

```bash
# Kill existing server
lsof -ti:8001 | xargs kill -9

# Restart
./dev.sh
```

**2. Changes not detected**

```bash
# Ensure ENV=development is set
export ENV=development
python3 -m src.compression_server
```

**3. Import errors after adding files**

```bash
# Make sure __init__.py exists
touch src/new_module/__init__.py

# Restart server
```

---

## 🚦 Running Tests

### **Unit Tests**

```bash
# Test compression features
python test_commercial_features.py

# Test server authentication
python test_server_auth.py
```

### **Integration Tests**

```bash
# Ensure server is running first
./dev.sh

# In another terminal
python examples/basic_usage.py
python examples/langchain_example.py
python examples/llamaindex_example.py
```

---

## 📊 Monitoring Development

### **Check Service Health**

```bash
curl http://localhost:8001/health
```

Response:
```json
{
  "status": "healthy",
  "service": "VisionDrop Compression with Enterprise Tokenization",
  "tokenizer_enabled": true,
  "cache_enabled": true,
  "validator_enabled": true
}
```

### **View Usage Stats**

```bash
curl http://localhost:8001/usage/stats
```

### **Check Cache Performance**

```bash
curl http://localhost:8001/cache/stats
```

---

## 🔐 Security in Development

### **Local Development Security**

- ✅ No API key needed for localhost
- ✅ Database stored locally (`~/.omnimemory/`)
- ✅ No external network access required

### **Production Security**

When deploying to production:

1. **Disable localhost bypass** (optional):
   - Set `OMNIMEMORY_REQUIRE_AUTH=true`
   - Forces API key even for localhost

2. **Use HTTPS**:
   - Never send API keys over HTTP in production

3. **Set admin key**:
   ```bash
   export OMNIMEMORY_ADMIN_KEY="secure-random-key"
   ```

---

## 🎯 Next Steps

1. **Start development server**: `./dev.sh`
2. **Make your changes**: Edit Python files
3. **Test automatically**: Server reloads on save
4. **No API key needed**: Just make HTTP requests to localhost:8001

Happy coding! 🚀
