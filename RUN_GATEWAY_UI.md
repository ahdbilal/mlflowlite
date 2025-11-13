# 🚀 How to Run the Gateway UI

## Quick Start (2 Steps)

### 1. Start the Gateway UI Server

```bash
cd /Users/ahmed.bilal/Desktop/gateway-oss
venv311/bin/python mlflowlite/ui/gateway_server.py
```

### 2. Open in Browser

```
http://localhost:5001/gateway
```

That's it! 🎉

---

## What You'll See

A modern web interface for managing LLM endpoints:

- **📋 Endpoints List** - View all configured endpoints
- **➕ Create Endpoint** - Add new provider configurations
- **✏️ Edit** - Modify existing endpoints
- **🗑️ Delete** - Remove unused endpoints
- **🧪 Test** - Verify endpoint configuration

**Pre-loaded Example Endpoints:**
- `prod-gpt4` - OpenAI GPT-4o
- `dev-claude` - Anthropic Claude Sonnet
- `azure-gpt35` - Azure OpenAI

---

## Full Setup (Both UIs)

Run MLflow + Gateway UI together:

```bash
# Terminal 1: Start MLflow UI
cd /Users/ahmed.bilal/Desktop/gateway-oss
venv311/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Terminal 2: Start Gateway UI
cd /Users/ahmed.bilal/Desktop/gateway-oss
venv311/bin/python mlflowlite/ui/gateway_server.py
```

**Or use the convenience script:**

```bash
./start_gateway_ui.sh
```

---

## Ports

| Service | URL |
|---------|-----|
| **Gateway UI** | http://localhost:5001/gateway |
| **MLflow UI** | http://localhost:5000 |

---

## Troubleshooting

### Port Already in Use

If you see "Address already in use":

```bash
# Stop any existing Gateway server
pkill -f gateway_server.py

# Then restart
venv311/bin/python mlflowlite/ui/gateway_server.py
```

### Flask Not Found

If you see "No module named 'flask'":

```bash
venv311/bin/pip install flask
```

### Wrong Python

Make sure you use the venv Python:

```bash
# ✅ Correct
venv311/bin/python mlflowlite/ui/gateway_server.py

# ❌ Wrong
python mlflowlite/ui/gateway_server.py
```

---

## Features to Try

1. **Search** - Filter endpoints by name, provider, or model
2. **Create Endpoint** - Click the blue button
3. **Provider Selection** - Choose from 6 LLM providers
4. **Edit Endpoint** - Click the edit button on any row
5. **Test Endpoint** - Verify configuration works

---

## What's Next?

This is a **mock implementation** - all data is stored in browser memory.

For production:
- Connect to backend database
- Integrate with secrets manager
- Add user authentication
- Hook into MLflowlite's completion API

See `mlflowlite/ui/README.md` for technical details.

