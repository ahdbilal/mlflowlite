# Too Many Experiments? Here's the Fix

## Problem

If you're seeing many experiments like:
- `mlflowlite`
- `mlflowlite_1729712345`
- `agent_support_bot`
- `agent_support_bot_1729712346`
- `ai_gateway_queries`
- etc.

This happens when experiments get deleted and recreated with timestamps.

## Quick Fix

### Option 1: Clean Up (Recommended)

```python
import mlflow

# List all experiments
exps = mlflow.search_experiments()
for exp in exps:
    print(f"{exp.name}: {exp.lifecycle_stage}")

# Delete old timestamped ones (keep only the main ones)
# Only delete experiments with timestamps in the name
for exp in exps:
    if any(char.isdigit() for char in exp.name.split('_')[-1]):  # Has timestamp
        if exp.lifecycle_stage == "active":
            mlflow.delete_experiment(exp.experiment_id)
        
# Permanently delete from trash
import shutil
import os
trash_dir = "./mlruns/.trash"
if os.path.exists(trash_dir):
    shutil.rmtree(trash_dir)
    print("✅ Cleaned up trash")
```

### Option 2: Use One Experiment for Everything

```python
import mlflowlite as mla

# Set one experiment name for all queries
mla.set_experiment_name('my_llm_queries')

# Now all queries use the same experiment
response1 = mla.query(model='claude-3-5-sonnet', prompt='Query 1')
response2 = mla.query(model='claude-3-5-sonnet', prompt='Query 2')
# Both appear in 'my_llm_queries' experiment
```

### Option 3: Fresh Start

```bash
# Delete all MLflow data and start fresh
rm -rf mlruns/
```

Then reinstall:
```python
%pip install git+https://github.com/ahdbilal/mlflowlite.git --upgrade
```

## In Databricks

Databricks autolog handles this automatically. You won't see multiple experiments because:
- All queries go to the notebook's experiment
- No manual experiment creation

Just make sure you've updated to the latest version:
```python
%pip install git+https://github.com/ahdbilal/mlflowlite.git --upgrade
dbutils.library.restartPython()
```

## Why It Happens

The multiple experiments occur when:
1. An experiment is deleted
2. MLflow won't let you recreate it (exists in trash)
3. The code creates a new one with a timestamp
4. This repeats if you delete and run again

The latest version automatically restores deleted experiments instead of creating new ones.

## What About Prompts Tab?

MLflow's "Prompts" tab is for registered prompts (like a prompt registry). To use it:

```python
from mlflow import MlflowClient

client = MlflowClient()

# Create a prompt
client.create_registered_model("my_prompt")

# Log a prompt version
with mlflow.start_run():
    mlflow.log_param("prompt", "You are a helpful assistant")
    mlflow.log_text("You are a helpful assistant", "system_prompt.txt")
```

**However**, for mlflowlite, prompts are automatically logged as:
- **Artifacts**: `prompt.txt` and `response.txt` in each run
- **Parameters**: Model, temperature, etc.
- **Traces**: Full input/output in trace spans

You don't need the Prompts tab - everything is in the Experiments!

## Best Practice

**For all queries:** Use one experiment
```python
import mlflowlite as mla
mla.set_experiment_name('llm_queries')  # One experiment for everything
```

**For agents:** Let them auto-create their own
```python
from mlflowlite import Agent

agent = Agent(name="bot", model="claude-3-5-sonnet")
# Creates 'agent_bot' experiment automatically
```

This gives you:
- ✅ One experiment for all queries
- ✅ Separate experiments for each agent
- ✅ Clean and organized

---

**Bottom line:** Update to latest version and optionally clean up old experiments. The new code won't create duplicates.

