# Troubleshooting mlflowlite

## Common Issues

### 1. "Experiment already exists in deleted state"

**Error:**
```
RuntimeError: All models failed after retries. Last error: Experiment 'mlflowlite' already exists in deleted state.
```

**Cause:** You previously deleted the MLflow experiment, and MLflow keeps it in a trash state.

**Solution 1: Restore the experiment (Recommended)**
```python
import mlflow

# Restore deleted experiment
exp = mlflow.get_experiment_by_name('mlflowlite')
if exp and exp.lifecycle_stage == 'deleted':
    mlflow.tracking.MlflowClient().restore_experiment(exp.experiment_id)
    print("✅ Experiment restored!")
```

**Solution 2: Use a different experiment name**
```python
import mlflowlite as mla

# Set a new experiment name
mla.set_experiment_name('mlflowlite_v2')

# Now run your queries
response = mla.query(model='claude-3-5-sonnet', prompt='Hello')
```

**Solution 3: Permanently delete from trash**
```bash
# Find MLflow tracking directory
# Usually: ./mlruns or ~/mlruns

# Remove the experiment folder from .trash
rm -rf mlruns/.trash/<experiment_id>
```

**Latest version handles this automatically** - Update to get automatic restoration:
```python
%pip install git+https://github.com/ahdbilal/mlflowlite.git --upgrade
```

---

### 2. Traces not showing up

**Symptom:** Can't find traces in the Traces tab

**Solutions:**

1. **Check MLflow version** (need >= 2.8.0 for Traces UI):
```python
import mlflow
print(f"MLflow version: {mlflow.__version__}")
```

2. **Verify tracing is enabled**:
```python
import mlflowlite as mla
print(f"Tracing enabled: {mla.litellm_style_api._mlflow_enabled}")

# Enable if False
mla.set_mlflow_tracking(True)
```

3. **Check the right place**:
   - In Databricks: Look for **"Traces"** in notebook sidebar (not Experiments)
   - Or: **Machine Learning → Tracing** in left sidebar

4. **Update to latest version**:
```python
%pip install git+https://github.com/ahdbilal/mlflowlite.git --upgrade
dbutils.library.restartPython()  # In Databricks
```

---

### 3. Authentication Errors

**Error:**
```
litellm.AuthenticationError: API key not found
```

**Solution:**
```python
import os

# Option 1: Set environment variable
os.environ['ANTHROPIC_API_KEY'] = 'your-key-here'

# Option 2: Pass directly to query
response = mla.query(
    model='claude-3-5-sonnet',
    prompt='Hello',
    api_key='your-key-here'
)

# Option 3: Use Databricks secrets (recommended)
os.environ['ANTHROPIC_API_KEY'] = dbutils.secrets.get(scope='keys', key='anthropic')
```

**Check which provider needs which key:**
- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Google: `GOOGLE_API_KEY`
- Mistral: `MISTRAL_API_KEY`

---

### 4. Timeout Errors

**Error:**
```
TimeoutError: Request timed out after 60s
```

**Solution 1: Increase timeout**
```python
# Per-request
response = mla.query(
    model='claude-3-5-sonnet',
    prompt='Complex task...',
    timeout=120  # 2 minutes
)

# Global default
mla.set_timeout(120)
```

**Solution 2: Use a faster model**
```python
# Instead of claude-3-5-sonnet, try:
response = mla.query(model='claude-3-haiku', prompt='...')  # Faster
response = mla.query(model='gpt-3.5-turbo', prompt='...')   # Faster
```

---

### 5. High Costs

**Symptom:** Unexpectedly high `cost_usd` in responses

**Solutions:**

1. **Use cheaper models**:
```python
# Expensive: claude-3-5-sonnet, gpt-4o
# Cheap: claude-3-haiku, gpt-3.5-turbo

response = mla.query(model='claude-3-haiku', prompt='...')
```

2. **Limit output tokens**:
```python
response = mla.query(
    model='claude-3-5-sonnet',
    prompt='Summarize briefly',
    max_tokens=100  # Limit response length
)
```

3. **Use smart routing** (automatically picks cheaper models for simple queries):
```python
from mlflowlite import smart_query

decision, response = smart_query('What is 2+2?')  # Uses cheap model
print(f"Used {decision.model}: {decision.reason}")
```

---

### 6. Module Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'mlflowlite'
```

**Solution:**
```python
# Install
%pip install git+https://github.com/ahdbilal/mlflowlite.git

# In Databricks, restart Python after install
dbutils.library.restartPython()
```

**Error:**
```
ModuleNotFoundError: No module named 'litellm'
```

**Solution:** Reinstall with dependencies:
```python
%pip install git+https://github.com/ahdbilal/mlflowlite.git --upgrade --force-reinstall
```

---

### 7. Databricks-Specific Issues

**Issue: Can't find experiments**

**Solution:** Experiments are auto-managed. Check:
1. Notebook sidebar → "Experiment" or "Traces" section
2. **Machine Learning** → **Experiments** or **Tracing**

**Issue: Permission errors**

**Solution:** Ensure you have write access to workspace:
```python
# Use your personal folder
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mla.set_experiment_name(f'/Users/{username}/my_experiment')
```

---

### 8. Retry/Fallback Not Working

**Issue:** Requests still fail even with retries configured

**Check:**
```python
# Verify configuration
import mlflowlite as mla

# Set retries
mla.set_max_retries(5)
mla.set_fallback_models(['gpt-4o', 'gpt-3.5-turbo'])

# Or per-request
response = mla.query(
    model='claude-3-5-sonnet',
    prompt='...',
    max_retries=5,
    fallback_models=['gpt-4o', 'gpt-3.5-turbo']
)
```

**Note:** Fallback only works if you have API keys for the fallback models!

---

## Getting Help

1. **Check trace details** - The trace ID shows what went wrong:
```python
print(f"Trace ID: {response.trace_id}")
# Look up this ID in Traces UI for detailed error info
```

2. **Enable verbose errors**:
```python
import mlflow
mlflow.autolog(log_traces=True, silent=False)
```

3. **Check documentation**:
   - `README.md` - Quick start
   - `DATABRICKS_SETUP.md` - Databricks-specific guide
   - `TRACE_VIEWING_GUIDE.md` - How to view traces

4. **Test with a simple query**:
```python
import mlflowlite as mla
import os

os.environ['ANTHROPIC_API_KEY'] = 'your-key'

response = mla.query(model='claude-3-haiku', prompt='Say hi')
print(response.content)
```

If this works, the issue is with your specific configuration, not the library.

---

## Quick Health Check

Run this to verify everything is working:

```python
import mlflowlite as mla
import mlflow
import os

print("=== mlflowlite Health Check ===\n")

# 1. Version
print(f"✓ MLflow version: {mlflow.__version__}")

# 2. Tracing enabled
print(f"✓ Tracing enabled: {mla.litellm_style_api._mlflow_enabled}")

# 3. API key set
has_key = bool(os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('OPENAI_API_KEY'))
print(f"{'✓' if has_key else '✗'} API key configured: {has_key}")

# 4. Test query (if key exists)
if has_key:
    try:
        model = 'claude-3-haiku' if os.environ.get('ANTHROPIC_API_KEY') else 'gpt-3.5-turbo'
        response = mla.query(model=model, prompt='Say "OK"', max_tokens=10)
        print(f"✓ Test query successful: {response.content}")
        print(f"  Trace ID: {response.trace_id}")
    except Exception as e:
        print(f"✗ Test query failed: {e}")

print("\n=== Health Check Complete ===")
```

