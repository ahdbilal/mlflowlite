# Using mlflowlite in Databricks

mlflowlite automatically detects Databricks environments and configures experiment paths correctly.

## Automatic Detection

The library automatically detects Databricks and uses proper experiment paths:

```python
import mlflowlite as ml

# Automatically uses /Users/your.email@company.com/mlflowlite in Databricks
response = ml.query(model='claude-3-5-sonnet', prompt='Hello')
```

## Custom Experiment Path

If you want to use a custom experiment name in Databricks:

```python
import mlflowlite as ml

# Set your custom experiment path (must be absolute in Databricks)
ml.set_experiment_name('/Users/your.email@company.com/my_custom_experiment')

# Now all queries will use this experiment
response = ml.query(model='claude-3-5-sonnet', prompt='Hello')
```

## Installation in Databricks

### Method 1: Install from GitHub

```python
%pip install git+https://github.com/ahdbilal/mlflowlite.git
```

### Method 2: Install from local wheel

1. Build the package:
   ```bash
   python setup.py bdist_wheel
   ```

2. Upload `dist/mlflowlite-*.whl` to Databricks

3. Install:
   ```python
   %pip install /path/to/mlflowlite-*.whl
   ```

### Method 3: Editable install (for development)

```python
%pip install -e .
```

## Setting API Keys

Use Databricks secrets for API keys:

```python
# Store secrets in Databricks
# Settings → Secrets → Create Secret

# In your notebook
import os
from databricks import secrets

# Set API keys
os.environ['ANTHROPIC_API_KEY'] = dbutils.secrets.get(scope='your_scope', key='anthropic_api_key')
os.environ['OPENAI_API_KEY'] = dbutils.secrets.get(scope='your_scope', key='openai_api_key')

# Now use mlflowlite
import mlflowlite as ml
response = ml.query(model='claude-3-5-sonnet', prompt='Hello')
```

## Viewing Experiments

In Databricks, experiments are visible in:
1. **Sidebar** → Experiments
2. **Path**: `/Users/your.email@company.com/mlflowlite`
3. All traces, metrics, and costs are logged automatically

## Complete Example

```python
# Install
%pip install git+https://github.com/ahdbilal/mlflowlite.git

# Restart Python
dbutils.library.restartPython()

# Setup API key
import os
os.environ['ANTHROPIC_API_KEY'] = dbutils.secrets.get(scope='keys', key='anthropic')

# Use mlflowlite
import mlflowlite as ml

# Query (automatically uses /Users/your.email/mlflowlite)
response = ml.query(
    model='claude-3-5-sonnet',
    prompt='Explain machine learning in one sentence'
)

print(f"Response: {response.content}")
print(f"Cost: ${response.cost:.4f}")
print(f"Trace ID: {response.trace_id}")

# View in Databricks Experiments UI
```

## Troubleshooting

### Error: "Got an invalid experiment name"

**Old error:**
```
RestException: INVALID_PARAMETER_VALUE: Got an invalid experiment name 'mlflowlite'. 
An experiment name must be an absolute path within the Databricks workspace
```

**Solution:**
This is now automatically handled! The library detects Databricks and uses proper paths.

If you still see this error:
```python
# Manually set the experiment path
ml.set_experiment_name('/Users/your.email@company.com/mlflowlite')
```

### Error: "No module named mlflowlite"

**Solution:**
```python
%pip install git+https://github.com/ahdbilal/mlflowlite.git
dbutils.library.restartPython()
```

### Error: "API key not found"

**Solution:**
```python
import os
os.environ['ANTHROPIC_API_KEY'] = dbutils.secrets.get(scope='your_scope', key='your_key')
# OR
os.environ['ANTHROPIC_API_KEY'] = 'your-key-here'  # Not recommended for production
```

## Features in Databricks

All mlflowlite features work in Databricks:

✅ Automatic tracing  
✅ Prompt versioning  
✅ DSPy optimization  
✅ Reliability (retry, timeout, fallbacks)  
✅ Smart routing  
✅ A/B testing  

Everything is logged to your Databricks MLflow workspace automatically!

