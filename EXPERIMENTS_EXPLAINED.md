# Experiments Structure - SIMPLIFIED

## The Fix

**Before:** Multiple confusing experiments
- `mlflowlite` (queries)
- `agent_support_bot` (agent runs)
- `agent_support_bot_74ccd1c6` (UUID versions)

**After (Latest Version):** ONE clean experiment
- `mlflowlite` - Everything goes here!
  - Query runs
  - Agent runs
  - All traces

## How It Works Now

### All in One Place

```python
import mlflowlite as mla
from mlflowlite import Agent

# Both use the SAME experiment
response = mla.query(model='claude-3-5-sonnet', prompt='Hello')
agent = Agent(name="bot", model="claude-3-5-sonnet")
result = agent.run("Hi")

# Both show up in: mlflowlite experiment
```

### What You See in MLflow UI

**Experiments Tab:**
```
mlflowlite/
├── Run: claude-3-5-sonnet_1729... (query)
├── Run: bot_abc123 (agent)
├── Run: claude-3-5-sonnet_1730... (query)
└── Run: bot_def456 (agent)
```

**Each Run Contains:**
- Parameters: model, temperature, etc.
- Metrics: cost, tokens, latency, scores
- Artifacts: prompt.txt, response.txt
- Trace: Full input/output with spans

## Why This Is Better

### Before (Multiple Experiments)
❌ Hard to compare queries vs agent runs
❌ Scattered data across experiments
❌ Confusing UUID suffixes
❌ Two places to look for data

### After (One Experiment)
✅ All LLM calls in one place
✅ Easy to compare everything
✅ Clean naming
✅ One view for all data

## Databricks

In Databricks, it's even simpler:
- Autolog manages everything
- All runs in your notebook's experiment
- No manual experiment setup needed

## If You See Multiple Experiments

That's from old code. To fix:

```python
# Clean start
import shutil
import os
if os.path.exists('mlruns'):
    shutil.rmtree('mlruns')
    
# Update code
%pip install git+https://github.com/ahdbilal/mlflowlite.git --upgrade

# Now everything uses mlflowlite experiment
```

## Traces with No Data?

If you see empty traces, it means:
1. Old version of code (update!)
2. Runs created before traces were properly integrated

**Fix:** Update and rerun your notebook. New runs will have full trace data!

---

**Bottom line:** Update to latest version → One clean experiment → All data in one place! 🎯

