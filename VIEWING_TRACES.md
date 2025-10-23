# Viewing Traces in Databricks

Your traces ARE working! Here's where to find the payload:

## Where to Look

### Option 1: Experiments UI
1. Go to **Experiments** in Databricks sidebar
2. Navigate to `/Users/your.email@company.com/mlflowlite`
3. Click on any run
4. You'll see:
   - **Parameters tab**: Model, temperature, message_count, response_preview
   - **Metrics tab**: latency_seconds, total_tokens, cost_usd, scores
   - **Artifacts tab**: `input.txt` and `output.txt` files ← **PAYLOAD HERE!**

### Option 2: Using MLflow UI
```python
# In your notebook
print(f"View traces at: /Users/{username}/mlflowlite")
print(f"Trace ID: {response.trace_id}")
```

Then in Databricks:
1. **Machine Learning** → **Experiments**
2. Find your experiment
3. Click on the run
4. Check **Artifacts** for full input/output

## What Gets Logged

For each `mla.query()` or `mla.completion()` call:

### Parameters (in Parameters tab)
- `model`: Which model was used
- `temperature`: Temperature setting
- `message_count`: Number of messages
- `response_preview`: First 200 characters of response
- `timeout`: Timeout setting

### Metrics (in Metrics tab)
- `latency_seconds`: How long it took
- `total_tokens`: Total tokens used
- `prompt_tokens`: Input tokens
- `completion_tokens`: Output tokens
- `cost_usd`: Estimated cost
- `score_helpfulness`: Quality score
- `score_conciseness`: Brevity score  
- `score_speed`: Speed score

### Artifacts (in Artifacts tab) ← **FULL PAYLOAD!**
- `input.txt`: Complete input (prompt + messages)
- `output.txt`: Complete response

## Example Code

```python
# Make a query
response = mla.query(
    model='claude-3-5-sonnet',
    prompt='Summarize this',
    input='Your text here'
)

# Get trace info
print(f"✅ Response: {response.content}")
print(f"💰 Cost: ${response.cost:.4f}")
print(f"🔍 Trace ID: {response.trace_id}")
print(f"\n📊 View full trace in Experiments:")
print(f"   Path: /Users/{username}/mlflowlite")
print(f"   Look in 'Artifacts' tab for input.txt and output.txt")
```

## Troubleshooting

### "I don't see Artifacts tab"
**Solution:** Update to latest mlflowlite version:
```python
%pip install git+https://github.com/ahdbilal/mlflowlite.git --upgrade
dbutils.library.restartPython()
```

### "Artifacts are empty"
**Possible cause:** Databricks workspace permissions

**Solution:** Ensure you have write access to your user folder:
```python
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mla.set_experiment_name(f'/Users/{username}/mlflowlite')
```

### "I see the trace but no details"
**Solution:** Click on the specific run (not just the experiment). The payload is in:
1. Click the run → **Artifacts** tab → `input.txt` and `output.txt`
2. Or **Parameters** tab → `response_preview` (truncated)

## Complete Example in Databricks

```python
# Setup
%pip install git+https://github.com/ahdbilal/mlflowlite.git
dbutils.library.restartPython()

# Configure
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

import mlflowlite as mla
mla.set_experiment_name(f'/Users/{username}/mlflowlite')

import os
os.environ['ANTHROPIC_API_KEY'] = dbutils.secrets.get(scope='keys', key='anthropic')

# Make a query
response = mla.query(
    model='claude-3-5-sonnet',
    prompt='Explain quantum computing in simple terms'
)

# Print results
print("✅ Response:", response.content)
print(f"\n📊 Metrics:")
print(f"   Cost: ${response.cost:.4f}")
print(f"   Tokens: {response.usage.get('total_tokens', 0)}")
print(f"   Latency: {response.latency:.2f}s")

print(f"\n🔍 View full trace:")
print(f"   1. Go to Experiments in sidebar")
print(f"   2. Open: /Users/{username}/mlflowlite")
print(f"   3. Click the latest run")
print(f"   4. Check 'Artifacts' tab for full input/output")
print(f"   5. Trace ID: {response.trace_id}")
```

## What You'll See

```
Artifacts/
├── input.txt           ← Full prompt and input
└── output.txt          ← Complete response

Parameters/
├── model               ← claude-3-5-sonnet
├── temperature         ← 0.7
├── message_count       ← 1
└── response_preview    ← First 200 chars

Metrics/
├── latency_seconds     ← 1.23
├── total_tokens        ← 150
├── cost_usd            ← 0.0045
├── score_helpfulness   ← 0.9
└── score_speed         ← 0.95
```

All the payload is there! Just look in the **Artifacts** tab. 🎉
