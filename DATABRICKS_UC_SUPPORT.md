# Databricks Unity Catalog Support

## ✅ What's Fixed

mlflowlite now works seamlessly on **both Local and Databricks** environments with Unity Catalog support.

### Key Changes

1. **Automatic Environment Detection**
   - Detects if running on Databricks or locally
   - Uses appropriate prompt naming convention for each environment

2. **Unity Catalog Integration**
   - **Local**: Prompts stored as `supportbot_prompt`
   - **Databricks**: Prompts stored as `main.default.supportbot_prompt` (Unity Catalog format)

3. **Correct API Usage**
   - **Databricks**: Uses `mlflow.genai.register_prompt()` (Unity Catalog API)
   - **Local**: Uses `mlflow.register_prompt()` (legacy API)

4. **Name Sanitization**
   - Converts camelCase to lowercase (`supportBot` → `supportbot_prompt`)
   - Removes invalid characters (spaces, dots, hyphens → underscores)
   - Ensures Unity Catalog compatibility

---

## 🚀 How to Use

### On Databricks

```python
# Cell 1: Install
%pip install -e . --force-reinstall --no-deps --quiet

# Cell 2: Restart Python
dbutils.library.restartPython()

# Cell 3: Use it!
from mlflowlite import Agent

agent = Agent(
    name="supportBot",
    model="claude-3-5-sonnet-20240620", 
    system_prompt="You are a support bot."
)

# Prompt will be registered as: main.default.supportbot_prompt
```

###On Local

```python
# Install
%pip install -e . --force-reinstall --no-deps --quiet

# Use it!
from mlflowlite import Agent

agent = Agent(
    name="supportBot",
    model="claude-3-5-sonnet-20240620",
    system_prompt="You are a support bot."
)

# Prompt will be registered as: supportbot_prompt
```

---

## ⚙️ Configuration

### Change Unity Catalog Schema

By default, Databricks uses `main.default`. To customize:

```python
import os
os.environ['MLFLOW_PROMPT_REGISTRY_UC_SCHEMA'] = 'my_catalog.my_schema'
```

Or set as MLflow experiment tag:

```python
import mlflow
mlflow.set_experiment_tags({
    "mlflow.promptRegistryLocation": "my_catalog.my_schema"
})
```

---

## 📋 What Was Fixed

| Issue | Solution |
|-------|----------|
| `INVALID_PARAMETER_VALUE: name is not a valid name` | Added Unity Catalog format support (`catalog.schema.name`) |
| CamelCase names failing | Added lowercase conversion |
| Wrong API on Databricks | Use `mlflow.genai.register_prompt()` for Databricks |
| Works locally but not Databricks | Auto-detect environment and use correct API |

---

## 🧪 Tested

✅ **Local Environment**: Prompts register successfully  
✅ **Databricks Connect**: Prompts register to Unity Catalog  
✅ **Name Sanitization**: `supportBot` → `supportbot_prompt`  
✅ **Unity Catalog Format**: `main.default.supportbot_prompt`

---

## 📚 References

- [Databricks MLflow Prompt Registry](https://docs.databricks.com/aws/en/mlflow3/genai/prompt-version-mgmt/prompt-registry/create-and-edit-prompts)
- [Unity Catalog Requirements](https://docs.databricks.com/aws/en/data-governance/unity-catalog/index.html)

