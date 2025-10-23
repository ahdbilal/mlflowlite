# Viewing Traces

## In Databricks

**mlflowlite uses Databricks autolog** - experiments are automatically managed for you!

### Where to Find Your Traces

**Option 1: Notebook Sidebar (Easiest)**
1. Look at the right sidebar in your Databricks notebook
2. Find the **"Experiment"** section
3. Click on the experiment link
4. All your `mla.query()` and `mla.completion()` calls are logged there

**Option 2: Experiments UI**
1. Go to **Machine Learning** → **Experiments** in Databricks
2. Find the experiment associated with your notebook
3. Click to view all runs

### What You'll See in Each Run

**Metrics Tab:**
- `latency_seconds` - How long the request took
- `total_tokens` - Total tokens used
- `prompt_tokens` - Input tokens
- `completion_tokens` - Output tokens
- `cost_usd` - Estimated cost
- `score_helpfulness` - Quality score
- `score_conciseness` - Brevity score
- `score_speed` - Speed score

**Parameters Tab:**
- `model` - Which model was used
- `temperature` - Temperature setting
- `message_count` - Number of messages
- `timeout` - Timeout configuration
- `response_preview` - First 200 characters

**Artifacts Tab:** ← **FULL PAYLOAD HERE!**
- `input.txt` - Complete input (prompt + messages)
- `output.txt` - Complete response

## In Local Environment

```bash
# Start MLflow UI
mlflow ui

# Open browser
open http://localhost:5000
```

Then:
1. Look for the `mlflowlite` experiment
2. Click on any run
3. Check Artifacts for `input.txt` and `output.txt`

## Example: Viewing a Trace

```python
import mlflowlite as mla

# Make a query
response = mla.query(
    model='claude-3-5-sonnet',
    prompt='Explain quantum computing in simple terms'
)

# Print trace info
print(f"✅ Response: {response.content[:100]}...")
print(f"💰 Cost: ${response.cost:.4f}")
print(f"⏱️  Latency: {response.latency:.2f}s")
print(f"🔢 Tokens: {response.usage.get('total_tokens', 0)}")
print(f"🔍 Trace ID: {response.trace_id}")
print(f"\n📊 View full trace in the Experiment sidebar →")
```

## Customizing Experiment Location (Optional)

By default, Databricks autolog manages experiments. But you can customize:

```python
# Set custom experiment location
mla.set_experiment_name('/Users/your.email@company.com/my_custom_exp')

# Now all queries will log there
response = mla.query(model='claude-3-5-sonnet', prompt='Hello')
```

For agents:
```python
from mlflowlite import Agent

agent = Agent(
    name="support_bot",
    model="claude-3-5-sonnet",
    experiment_name='/Users/your.email@company.com/my_agent_exp'  # optional
)
```

## Pro Tips

1. **Check the sidebar first** - In Databricks, the experiment link is right there in your notebook
2. **Click on individual runs** - The payload is in the Artifacts tab of each run, not the experiment overview
3. **Use trace_id** - Every response has a `trace_id` for exact lookup
4. **Filter by date** - In the Experiments UI, you can filter runs by date, model, cost, etc.

---

**Bottom line:** Just start using `mla.query()` - Databricks handles the rest automatically! 🎉
