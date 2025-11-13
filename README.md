# MLflowlite - Enterprise LLM Gateway

**Keep your GenAI stack at the frontier without breaking production.**

When new models (GPT-5, Claude Sonnet 4.5, Gemini 2.5) are released, platform teams need to **evaluate, compare, and gradually migrate** their apps — balancing quality, latency, cost, and governance. MLflowlite makes this seamless with a unified API and management UI.

## 🚀 New: Gateway UI

Centralized web interface for managing LLM endpoints, monitoring usage, and tracking costs across teams.

**Features:**
- 📊 **Real-time Dashboard** - Aggregate metrics, cost tracking, latency monitoring
- 🔌 **Endpoint Management** - Create, edit, and manage LLM provider endpoints
- 👥 **Team Usage Tracking** - Monitor consumption by team (ML, Product, Research)
- 📈 **Live Metrics** - Request rates, success rates, P95 latency (updates every 5s)
- 🔗 **MLflow Integration** - Direct links to traces and prompt registry
- 💰 **Cost Analytics** - Cost breakdown by provider and endpoint

**Quick Start:**
```bash
venv311/bin/python mlflowlite/ui/gateway_server.py
# Open http://localhost:5001/gateway
```

[📖 Complete Gateway UI Documentation](./GATEWAY_UI_FEATURES.md) | [🎯 Quick Start Guide](./RUN_GATEWAY_UI.md)

---

## Core Use Case: Model Migration Workflow

1. ✅ **Baseline Capture** - Collect outputs from your current model
2. 🔄 **New Model Evaluation** - Compare side-by-side with same prompts
3. 🎯 **Automatic Optimization** - Auto-rewrite prompts for new models
4. 📊 **Performance Comparison** - Quality, latency, cost analysis
5. 🚀 **Gradual Migration** - Risk-free A/B testing and rollout
6. ✨ **Production Deployment** - Zero downtime, zero code changes

**Key Benefit:** One API for all models. Switch providers, upgrade models, optimize costs — without rewriting code.

---

## Installation

```bash
git clone https://github.com/ahdbilal/mlflowlite.git
cd mlflowlite
python -m venv venv311
source venv311/bin/activate  # On Windows: venv311\Scripts\activate
pip install -e .
```

## Quick Start

### 1. Run Demo Notebook

```bash
./start_notebook.sh
# Open MLflowlite_Demo.ipynb
```

The demo showcases:
- Model migration (Sonnet → Haiku with cost optimization)
- Prompt optimization with MLflow
- A/B testing between model variants
- Automatic MLflow tracing

### 2. Start Gateway UI

```bash
venv311/bin/python mlflowlite/ui/gateway_server.py
```
Open http://localhost:5001/gateway to manage endpoints and view metrics.

### 3. Start MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open http://localhost:5000 to view traces, experiments, and prompts.

---

## Features

### 🎯 Automatic MLflow Tracing

Every LLM call is automatically traced to MLflow with full input/output capture.

```python
import mlflowlite as mla

response = mla.completion(
    model='claude-haiku-4-5-20251001',
    messages=[{"role": "user", "content": "Hello!"}]
)

print(f"Response: {response.content}")
print(f"Cost: ${response.cost:.6f}")      # From LiteLLM pricing database
print(f"Trace: {response.trace_url}")     # Click to view in MLflow UI
```

**What's Tracked:**
- ✅ Request inputs (messages, model, temperature, max_tokens)
- ✅ Response outputs (text, token usage, cost, latency)
- ✅ Metadata (model, provider, finish reason)
- ✅ Clickable trace URLs for easy debugging

### 💰 Cost Tracking

Accurate cost calculation using LiteLLM's built-in pricing database.

```python
response = mla.completion(
    model='claude-sonnet-4-5-20250929',
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)

print(f"💰 Cost: ${response.cost:.6f}")
print(f"📊 Tokens: {response.usage['total_tokens']}")
print(f"⚡ Latency: {response.latency:.2f}s")
```

**Supports all major providers:**
- OpenAI (GPT-4o, GPT-4 Turbo, GPT-3.5)
- Anthropic (Claude 4.5 Sonnet/Haiku, Claude 3)
- Azure OpenAI
- Google (Gemini)
- AWS Bedrock
- Cohere

### 🎨 Simplified Prompt Optimization

One-line prompt optimization powered by MLflow's GEPA algorithm.

```python
optimized_prompt = mla.optimize_prompt(
    prompt_template="Classify sentiment: {{text}}",
    dataset=dataset_df,
    model_from="claude-sonnet-4-5-20250929",  # Expensive
    model_to="claude-haiku-4-5-20251001",     # Cheap
    prompt_id="sentiment_classifier",
    max_iterations=5,
    save_optimized=True  # Auto-saves to MLflow
)

print(f"Optimized prompt:\n{optimized_prompt}")
```

**What it does:**
1. Tests your prompt on the dataset
2. Generates improved versions using an LLM
3. Evaluates each variant
4. Returns the best-performing prompt
5. Saves to MLflow prompt registry

### 📝 Prompt Versioning & Registry

MLflow-integrated prompt management with automatic versioning.

```python
# Register a prompt
response = mla.completion(
    model="claude-haiku-4-5-20251001",
    prompt_id="support_bot",
    prompt_template="You are a helpful support agent. Answer: {{question}}",
    prompt_variables={"question": "How do I reset my password?"}
)

# Prompts are automatically:
# ✅ Saved to MLflow prompt registry
# ✅ Versioned (v1, v2, v3...)
# ✅ Linked to traces
# ✅ Viewable in MLflow UI

# View in UI: http://localhost:5000/#/prompts
```

### 🔄 A/B Testing

Data-driven model and prompt optimization with built-in routing.

```python
from mlflowlite.routing import create_ab_test

# Create A/B test
ab_test = create_ab_test(
    name="model_test",
    variants={
        'prod': {'model': 'claude-sonnet-4-5-20250929'},
        'new':  {'model': 'claude-haiku-4-5-20251001'}
    },
    split=[0.8, 0.2]  # 80/20 split
)

# Run test
for text in test_inputs:
    variant, resp = ab_test.run(
        messages=[{"role": "user", "content": text}]
    )
    print(f"{variant}: ${resp.cost:.4f}")

# View results
ab_test.print_report()
```

**Output:**
```
🏆 Winners:
   • Best cost: new ($0.0001 vs $0.0027)
   • Best latency: new (0.8s vs 1.2s)
   • Best quality: prod (N/A)
```

### 📊 GenAI Evaluation

MLflow-style evaluation with built-in and custom scorers.

```python
from mlflowlite import evaluate, Correctness, Guidelines, scorer

# Define evaluation dataset
eval_dataset = [
    {
        "inputs": {"question": "What is the capital of France?"},
        "expectations": {"expected_response": "Paris"},
    },
]

# Define prediction function
def qa_predict_fn(question: str) -> str:
    return mla.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}]
    ).content

# Built-in scorers (LLM-as-a-Judge)
correctness_scorer = Correctness()
english_scorer = Guidelines(name="is_english", guidelines="Answer must be in English")

# Custom scorer
@scorer
def is_concise(outputs: str) -> bool:
    return len(outputs.split()) <= 20

# Run evaluation
results = evaluate(
    data=eval_dataset,
    predict_fn=qa_predict_fn,
    scorers=[correctness_scorer, english_scorer, is_concise],
)

# View results
print(results.aggregate_scores)
```

**Built-in Scorers:**
- `Correctness` - Check factual accuracy using LLM-as-a-Judge
- `Guidelines` - Check adherence to natural language rules
- `Relevance` - Check if response is on-topic
- `Faithfulness` - Check for hallucinations (RAG apps)
- `Conciseness` - Check response length

### 🛡️ Reliability Features

Built-in retry logic, timeouts, and fallback chains.

```python
response = mla.completion(
    model='claude-sonnet-4-5-20250929',
    messages=[...],
    timeout=30,                                      # seconds
    max_retries=5,                                   # retry attempts
    fallback_models=['claude-haiku-4-5-20251001', 'gpt-4o']  # fallback chain
)
```

---

## API Reference

### Main Functions

```python
import mlflowlite as mla

# LiteLLM-style completion (recommended)
response = mla.completion(
    model='claude-haiku-4-5-20251001',
    messages=[{"role": "user", "content": "Hello"}]
)

# With prompt template
response = mla.completion(
    model='claude-haiku-4-5-20251001',
    prompt_id="greeting_bot",
    prompt_template="Greet the user: {{name}}",
    prompt_variables={"name": "Alice"}
)

# Batch processing
responses = mla.batch_completion(
    model='claude-haiku-4-5-20251001',
    messages_list=[...]
)

# Optimize prompts
optimized = mla.optimize_prompt(
    prompt_template="...",
    dataset=df,
    model_from="expensive-model",
    model_to="cheap-model",
    max_iterations=5
)
```

### Response Object (OpenAI-Compatible)

```python
# OpenAI-compatible access
response.id                              # Unique completion ID
response.object                          # "chat.completion"
response.created                         # Unix timestamp
response.model                           # Model used
response.choices[0]["message"]["content"]  # Response text
response.usage["total_tokens"]           # Token usage

# Convenience attributes
response.content      # Direct access to response text
response.latency      # Response time in seconds
response.cost         # Cost from LiteLLM pricing (accurate!)
response.scores       # Quality scores dict
response.trace_id     # MLflow trace ID
response.trace_url    # Clickable MLflow UI link

# Convert to OpenAI JSON format
response.to_dict()    # Full OpenAI-compatible dictionary
```

### Configuration

```python
import mlflow

# MLflow is auto-configured on first call
# But you can customize:
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("my_experiment")

# Or set a unique experiment for demo purposes
from datetime import datetime
experiment_name = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
mlflow.set_experiment(experiment_name)
```

---

## Gateway UI Features

The Gateway UI provides a centralized interface for managing LLM endpoints:

### Dashboard View
- **Aggregate Metrics** - Total requests, active endpoints, costs, latency
- **Request Charts** - Time-series visualization with filters (1H/6H/24H/7D)
- **Cost Breakdown** - Pie chart by provider (OpenAI, Anthropic, Azure, Google)
- **Latency Distribution** - Bar chart showing performance buckets
- **Team Usage** - Consumption by ML Team, Product Team, Research Team

### Endpoints View
- **Endpoints List** - Table with metrics (requests, latency, cost, status)
- **Create Endpoint** - Modal form for adding new endpoints
- **Endpoint Details** - Click any row to see:
  - Real-time metrics (updates every 5s)
  - Usage metrics (6 cards: requests, success rate, latency, cost, tokens, errors)
  - Cost and latency trend charts
  - Direct links to MLflow traces and prompt registry

### Provider Support
- OpenAI (GPT-4o, GPT-4 Turbo, GPT-3.5)
- Anthropic (Claude 4.5 Sonnet/Haiku, Claude 3)
- Azure OpenAI
- Google Gemini (1.5 Pro/Flash)
- AWS Bedrock
- Cohere

[📖 See complete Gateway UI documentation](./GATEWAY_UI_FEATURES.md)

---

## Models Supported

| Provider | Models | Notes |
|----------|--------|-------|
| **OpenAI** | `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`, `gpt-4o-mini` | Most popular |
| **Anthropic** | `claude-sonnet-4-5-20250929`, `claude-haiku-4-5-20251001`, `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku` | Latest versions |
| **Azure OpenAI** | Same as OpenAI | Enterprise deployment |
| **Google** | `gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-pro` | Fast and cheap |
| **Mistral** | `mistral-large`, `mistral-medium`, `mistral-small` | European option |
| **AWS Bedrock** | Various | Enterprise AWS |
| **Cohere** | `command-r-plus`, `command-r`, `command` | RAG-optimized |

Powered by [LiteLLM](https://github.com/BerriAI/litellm) - supports 100+ models.

---

## Examples

### Notebooks
- **`MLflowlite_Demo.ipynb`** ⭐ - Start here! Complete demo with:
  - Model migration workflow
  - Prompt optimization
  - A/B testing
  - Automatic tracing

### Python Scripts
- `examples/quick_start.py` - Minimal example
- `examples/openai_compatibility_demo.py` - OpenAI API format
- `examples/reliability_demo.py` - Retry, timeout, fallbacks
- `examples/routing_demo.py` - Smart routing & A/B testing
- `examples/evaluation_demo.py` - GenAI evaluation
- `examples/model_migration_cost_reduction.py` - Cost optimization

### Gateway UI
```bash
# Start Gateway UI
venv311/bin/python mlflowlite/ui/gateway_server.py

# Open http://localhost:5001/gateway
```

---

## Troubleshooting

**Gateway UI Flask error?**
```bash
# Use the venv Python (not system Python)
venv311/bin/python mlflowlite/ui/gateway_server.py
```

**Traces not showing in MLflow UI?**
- Check MLflow UI is running: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
- Verify tracking URI: MLflow auto-configures to `sqlite:///mlflow.db`
- Look in the correct experiment (MLflow creates one automatically)

**Cost seems wrong?**
- Cost is calculated using LiteLLM's pricing database
- Updated regularly with latest model prices
- Fallback to estimation if model not found

**API errors?**
```python
import os
os.environ['ANTHROPIC_API_KEY'] = 'your-key-here'
os.environ['OPENAI_API_KEY'] = 'your-key-here'
```

---

## Structure

```
mlflowlite/
├── __init__.py
├── litellm_style_api.py    # Main API
├── agent.py                # Advanced agent
├── llm/                    # LLM providers
├── tools/                  # Tool framework
├── tracing/                # MLflow integration
├── prompts/                # Prompt versioning & registry
├── evaluation/             # Scoring & evaluation
├── optimization/           # Prompt optimization
├── routing.py              # A/B testing & routing
└── ui/                     # Gateway UI
    ├── gateway/
    │   ├── index.html          # Original UI
    │   └── index_enhanced.html # Enhanced UI with dashboard
    ├── gateway_server.py       # Flask server
    └── README.md               # UI documentation

examples/                   # Example scripts
tests/                      # Test suite
MLflowlite_Demo.ipynb      # Interactive demo notebook
```

---

## Key Improvements in Latest Version

### ✅ MLflow Tracing
- **Automatic tracing** - No setup required
- **Full input/output capture** - See exact requests/responses
- **Clickable trace URLs** - Easy debugging

### ✅ Cost Tracking
- **LiteLLM integration** - Accurate pricing from provider database
- **Per-request costs** - Track spending precisely
- **Cost analytics** - Dashboard view in Gateway UI

### ✅ Prompt Optimization
- **Simplified API** - One function call (`mla.optimize_prompt()`)
- **Automatic saving** - Prompts saved to MLflow registry
- **Iteration control** - Set max iterations for quick tests

### ✅ Gateway UI
- **Endpoint management** - Create, edit, delete endpoints
- **Real-time metrics** - Live dashboard with updates
- **Team tracking** - Monitor usage by team
- **MLflow integration** - Direct links to traces and prompts

---

## Contributing

Contributions welcome! Please open an issue or PR.

## License

Apache 2.0

---

## Links

- **GitHub**: https://github.com/ahdbilal/mlflowlite
- **MLflow**: https://mlflow.org
- **LiteLLM**: https://github.com/BerriAI/litellm

---

**Built with ❤️ for teams managing production LLM applications**
