# How to View Traces in Databricks

## Update First!

```python
%pip install git+https://github.com/ahdbilal/mlflowlite.git --upgrade
dbutils.library.restartPython()
```

## Where Are Traces?

In Databricks, there are **two places** to find your MLflow data:

### 1. **Traces Tab** (What you want!)
This shows detailed trace visualization with full I/O.

**How to access:**
1. In your notebook, look at the **right sidebar**
2. Click on **"Traces"** (may be next to "Experiments")
3. OR: Go to **Machine Learning → Tracing** in the left sidebar

**What you'll see:**
- Each `mla.query()` call as a separate trace
- Click on a trace to see:
  - **Inputs**: Full messages with expandable JSON
  - **Outputs**: Complete response
  - **Spans**: Breakdown of LLM call timing
  - **Attributes**: cost, tokens, latency, scores

### 2. **Experiments Tab** (Old style)
This shows run-based logging (less detailed).

**Note:** With the latest version, traces should appear in the **Traces tab**, not Experiments.

## Example

```python
import mlflowlite as mla
import os

# Setup
os.environ['ANTHROPIC_API_KEY'] = dbutils.secrets.get(scope='keys', key='anthropic')

# Make a query
response = mla.query(
    model='claude-3-5-sonnet',
    prompt='Summarize this support ticket in 2 sentences',
    input='Customer is having login issues...'
)

print(f"✅ Response: {response.content}")
print(f"🔍 Trace ID: {response.trace_id}")
print(f"\n📊 View trace:")
print(f"   1. Click 'Traces' in the right sidebar")
print(f"   2. Find trace with ID: {response.trace_id}")
print(f"   3. Click to see full input/output")
```

## Troubleshooting

### "I don't see a Traces tab"

**Possible causes:**
1. MLflow version in Databricks is too old
2. Traces feature not enabled in your workspace

**Solution:**
Check your MLflow version:
```python
import mlflow
print(f"MLflow version: {mlflow.__version__}")
# Need >= 2.8.0 for Traces UI
```

If too old, contact your Databricks admin or use Databricks Runtime 13.0+

### "Traces tab is empty"

**Possible causes:**
1. Tracing not enabled
2. Old version of mlflowlite

**Solution:**
```python
# Check if tracing is enabled
import mlflowlite as mla
print(f"Tracing enabled: {mla.litellm_style_api._mlflow_enabled}")

# If False, enable it:
mla.set_mlflow_tracking(True)
```

### "I only see Experiments, not Traces"

The old implementation logged to Experiments. With the latest version (after this update), everything goes to the **Traces tab**.

**To verify you have the latest:**
```python
import mlflowlite
print(mlflowlite.__version__)
# Should be 0.1.0 or higher
```

## What Gets Logged

For each `mla.query()` or `mla.completion()`:

**Trace-level:**
- Request ID (trace_id)
- Timestamp
- Overall duration

**Span-level (llm_call):**
- **Inputs**: Full messages array
- **Outputs**: Complete response + finish_reason
- **Attributes**:
  - `model`: Model used
  - `temperature`: Temperature setting
  - `provider`: Provider name
  - `latency_seconds`: Request duration
  - `total_tokens`, `prompt_tokens`, `completion_tokens`
  - `cost_usd`: Estimated cost

**Span-level (evaluation):**
- `helpfulness`, `conciseness`, `speed` scores

## Trace vs Experiment

| Feature | Traces Tab | Experiments Tab |
|---------|-----------|-----------------|
| **Visualization** | Structured tree view | Flat run list |
| **Input/Output** | Full, expandable JSON | Truncated in params |
| **Timing** | Detailed span breakdown | Single latency metric |
| **Best for** | Debugging LLM calls | Model comparison |

**Recommendation:** Use the **Traces tab** for day-to-day debugging and monitoring!

---

**Bottom line:** Update mlflowlite, run a query, then check the **Traces** tab (not Experiments)! 🎯

