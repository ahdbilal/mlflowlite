# 🔗 Trace → Prompt Linkage

## What Is This?

**Every trace is automatically linked to the prompt version that generated it.**

This means you can:
1. ✅ **Filter traces** by prompt name
2. ✅ **See which prompt version** generated each result
3. ✅ **Compare performance** across prompt versions
4. ✅ **Track prompt evolution** and impact on quality/cost/speed

---

## How It Works

### 1. Create Agent with `prompt_name`

```python
from mlflowlite import Agent

agent = Agent(
    model="claude-3-5-sonnet",
    prompt="Analyze: {{ticket}}",
    prompt_name="support_bot"  # 👈 Triggers registration + linkage
)
```

This:
- Registers prompt to **MLflow Prompt Registry** (visible in "Prompts" tab)
- Assigns a **version number** (auto-incremented)
- Stores **full prompt template** with {{variables}}

### 2. Make Calls

```python
response = agent(ticket="User can't login")
```

Behind the scenes, mlflowlite:
- Fills {{ticket}} with your input
- Sends to LLM
- Creates an **MLflow trace** with:
  - `prompt_name = "support_bot"`
  - `prompt_version = 1`
  - `prompt_registry_name = "support_bot_prompt"`

### 3. View in MLflow UI

#### Option A: Experiments Tab
1. Go to **Experiments** → `llm_workspace`
2. Click on any run
3. See **Parameters**:
   ```
   prompt_name: support_bot
   prompt_version: 1
   prompt_registry_name: support_bot_prompt
   ```
4. Click the **prompt name** → Jump to Prompt Registry

#### Option B: Traces Tab
1. Go to **Traces**
2. Filter by `prompt_name = "support_bot"`
3. Group by `prompt_version`
4. Compare metrics (latency, tokens, cost) across versions

#### Option C: Prompts Tab
1. Go to **Prompts** → `support_bot_prompt`
2. See all versions
3. For each version, see:
   - Template changes
   - Metadata (when changed, why)
   - Link to traces that used this version

---

## Example Workflow

### Monday: Create v1
```python
agent = Agent(
    model="claude-3-5-sonnet",
    prompt="You are a support bot. Analyze: {{ticket}}",
    prompt_name="support_bot"
)

# Make 100 calls
for ticket in tickets:
    response = agent(ticket=ticket)
```

**MLflow UI shows:**
- 100 traces
- All have `prompt_version = 1`
- Avg latency: 2.5s, Avg tokens: 250, Avg cost: $0.003

---

### Tuesday: Improve to v2
```python
# Add a more concise version
agent.prompt_registry.add_version(
    system_prompt="Analyze: {{ticket}}\n\nProvide: ISSUE, CAUSE, FIX (one line each)",
    user_template="{{query}}",
    metadata={"change": "Made more concise"}
)

# Make 100 more calls (automatically uses v2)
for ticket in tickets:
    response = agent(ticket=ticket)
```

**MLflow UI now shows:**
- 200 traces total
- 100 with `prompt_version = 1`
- 100 with `prompt_version = 2`
- **Compare side-by-side:**
  - v1: 2.5s, 250 tokens, $0.003
  - v2: 1.8s, 180 tokens, $0.002 ✅ **28% faster, 20% cheaper!**

---

### Wednesday: Roll Back to v1
```python
# Oops, v2 is less helpful. Roll back:
agent.prompt_registry.add_version(
    system_prompt="You are a support bot. Analyze: {{ticket}}",
    user_template="{{query}}",
    metadata={"change": "Rollback to v1 style"}
)
```

**MLflow UI:**
- Can still see v2 traces (preserved forever)
- New traces use v3 (which is same as v1, but tracked separately)
- Full history: v1 → v2 → v3 with data for each

---

## Benefits

### 🔍 **Debugging**
**Problem:** "Why did this query fail?"  
**Solution:** Check trace → See `prompt_version = 2` → Check prompt changes → Ah, we removed error handling!

### 📊 **A/B Testing**
```python
# Half the team uses v1, half uses v2
# MLflow automatically tracks and compares
```

### 💰 **Cost Tracking**
- See which prompt versions are expensive
- Optimize the costly ones
- Track savings over time

### 📈 **Prompt Evolution**
- Full Git-like history of prompt changes
- See what worked and what didn't
- Data-driven prompt engineering

---

## Under the Hood

### What Gets Logged

#### MLflow Run Tags (for "Linked prompts" UI):
```python
{
    "mlflow.promptName": "support_bot_prompt",           # 👈 Prompt name
    "mlflow.promptVersion": "2",                         # 👈 Version number
    "mlflow.promptSource": "prompts:/support_bot_prompt/2"  # 👈 Full URI
}
```
These special `mlflow.*` tags tell the MLflow UI to show the prompt in the **"Linked prompts"** tab!

#### MLflow Run Parameters:
```python
{
    "model": "claude-3-5-sonnet",
    "temperature": 0.7,
    "message_count": 2,
    "prompt_name": "support_bot",           # 👈 User-friendly name
    "prompt_version": 2,                    # 👈 Specific version
    "prompt_registry_name": "support_bot_prompt"  # 👈 Full registry name
}
```

#### MLflow Trace Attributes:
```python
{
    "model": "claude-3-5-sonnet",
    "provider": "anthropic",
    "latency_seconds": 1.8,
    "prompt_tokens": 120,
    "completion_tokens": 60,
    "prompt_name": "support_bot",           # 👈 Filterable!
    "prompt_version": 2,                    # 👈 Groupable!
    "prompt_registry_name": "support_bot_prompt",
    "mlflow.promptName": "support_bot_prompt",  # 👈 MLflow UI linkage
    "mlflow.promptVersion": "2"                  # 👈 MLflow UI linkage
}
```

---

## Databricks Unity Catalog

Works seamlessly on Databricks with Unity Catalog:

```python
# Local
agent = Agent(
    prompt="Analyze: {{ticket}}",
    prompt_name="support_bot"
)
# Registers as: support_bot_prompt

# Databricks (auto-detects)
agent = Agent(
    prompt="Analyze: {{ticket}}",
    prompt_name="support_bot"
)
# Registers as: main.default.support_bot_prompt (Unity Catalog format)

# Custom schema
os.environ['MLFLOW_PROMPT_REGISTRY_UC_SCHEMA'] = 'ml.bilal'
agent = Agent(
    prompt="Analyze: {{ticket}}",
    prompt_name="support_bot"
)
# Registers as: ml.bilal.support_bot_prompt
```

**Same API, works everywhere!**

---

## API Reference

### Agent Constructor
```python
Agent(
    model: str,
    prompt: Optional[str] = None,        # Template with {{variables}}
    prompt_name: Optional[str] = None,   # Triggers registration + linkage
    # ... other params
)
```

**Key rule:** If you provide `prompt_name`, traces are automatically linked!

### Prompt Registry
```python
# Add new version
agent.prompt_registry.add_version(
    system_prompt="...",
    user_template="{{query}}",
    metadata={"change": "reason"}
)

# Get current version
current = agent.prompt_registry.get_latest()
print(current.version)  # e.g., 3

# List all versions
for item in agent.prompt_registry.list_versions():
    print(f"v{item['version']}: {item['metadata']}")
```

---

## Summary

| Feature | Status |
|---------|--------|
| Auto-register prompts | ✅ Works |
| Version prompts | ✅ Works |
| Link traces to prompts | ✅ **NEW!** |
| Filter by prompt_name | ✅ **NEW!** |
| Group by prompt_version | ✅ **NEW!** |
| Compare across versions | ✅ **NEW!** |
| Databricks Unity Catalog | ✅ Works |

**You're now doing prompt engineering with data, not guesswork! 🎉**

