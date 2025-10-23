# mlflowlite in Databricks

## Quick Setup

```python
# Install
%pip install git+https://github.com/ahdbilal/mlflowlite.git
dbutils.library.restartPython()

# Configure API key (using Databricks secrets - recommended)
import os
os.environ['ANTHROPIC_API_KEY'] = dbutils.secrets.get(scope='your_scope', key='anthropic_api_key')

# Or directly (not recommended for production)
os.environ['ANTHROPIC_API_KEY'] = 'your-key-here'

# Use it!
import mlflowlite as mla

response = mla.query(
    model='claude-3-5-sonnet',
    prompt='Explain quantum computing',
)

print(f"Response: {response.content}")
print(f"Cost: ${response.cost:.4f}")
print(f"Trace ID: {response.trace_id}")
```

**That's it!** Databricks automatically manages experiments using autolog. No manual experiment setup needed.

## How It Works

mlflowlite detects Databricks environment and:
- **Lets Databricks autolog handle experiment tracking**
- No need to set experiment names or paths
- Experiments appear automatically in your notebook's Experiment sidebar
- Full input/output logged as artifacts

## Viewing Traces

### In Databricks UI

Databricks automatically creates and manages your experiments using autolog.

1. **Navigate to Experiments**
   - Click on **Machine Learning** in the sidebar
   - Click on **Experiments**

2. **Find Your Experiment**
   - Databricks will auto-create an experiment in your notebook folder
   - Usually named after your notebook or in the default experiment location
   - Look in the "Experiment" section in your notebook sidebar

3. **View Runs**
   - Click on the experiment
   - You'll see all your API calls as runs
   - Each run shows:
     - **Metrics**: Cost, latency, tokens, scores
     - **Parameters**: Model, temperature, settings
     - **Artifacts**: `input.txt` (full prompt), `output.txt` (full response)

### What Gets Logged

Every `mla.query()` or `mla.completion()` call logs:

**Metrics:**
- `latency_seconds` - Request duration
- `total_tokens`, `prompt_tokens`, `completion_tokens`
- `cost_usd` - Estimated cost
- `score_helpfulness`, `score_conciseness`, `score_speed`

**Parameters:**
- `model`, `temperature`, `message_count`, `timeout`
- `response_preview` - First 200 characters

**Artifacts:**
- `input.txt` - Full prompt/messages
- `output.txt` - Complete response

## Complete Example

```python
# Setup
%pip install git+https://github.com/ahdbilal/mlflowlite.git
dbutils.library.restartPython()

import os
os.environ['ANTHROPIC_API_KEY'] = dbutils.secrets.get(scope='keys', key='anthropic')

# Use it
import mlflowlite as mla

response = mla.query(
    model='claude-3-5-sonnet',
    prompt='Summarize the key points',
    input='Your text here...'
)

print(f"✅ Response: {response.content}")
print(f"💰 Cost: ${response.cost:.4f}")
print(f"⏱️  Latency: {response.latency:.2f}s")
print(f"🔍 Trace ID: {response.trace_id}")
print(f"\n📊 View in Experiment sidebar →")
```

## Advanced: Custom Experiment Location

By default, Databricks autolog manages experiments. But you can customize:

```python
# Set custom experiment location (optional)
mla.set_experiment_name('/Users/your.email@company.com/my_custom_experiment')

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

## Troubleshooting

### Where are my experiments?

**Check the notebook sidebar:**
- Look for the "Experiment" section in your Databricks notebook
- Click on it to see the experiment and all runs
- Or go to Machine Learning → Experiments and look for experiments related to your notebook

### Want to use a custom experiment location?

```python
# Set custom experiment (optional)
mla.set_experiment_name('/Users/your.email@company.com/my_custom_experiment')
```

**For Agents:**
```python
from mlflowlite import Agent

agent = Agent(
    name="support_bot",
    model="claude-3-5-sonnet",
    experiment_name='/Users/your.email@company.com/my_agent_experiment'
)
```

## Benefits of Databricks Autolog

✅ **Zero configuration** - Just start using `mla.query()`  
✅ **Automatic experiment management** - No paths or names to set  
✅ **Notebook integration** - Experiments appear in sidebar  
✅ **Full traceability** - Complete input/output in artifacts  
✅ **Cost tracking** - Automatic cost estimation per query  
✅ **Quality metrics** - Helpfulness, speed, conciseness scores

---

**Just install and use. Databricks handles the rest!** 🎉
