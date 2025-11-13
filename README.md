# MLflow AI Gateway

**Keep your GenAI stack at the frontier without breaking production.**

When new models (GPT-5, Claude Sonnet 4.5, Gemini 2.5) are released, platform teams need to **evaluate, compare, and gradually migrate** their apps — balancing quality, latency, cost, and governance. MLflow AI Gateway makes this seamless with a single unified API.

## Core Use Case: Model Migration Workflow

1. ✅ **Baseline Capture** - Collect outputs from your current model
2. 🔄 **New Model Evaluation** - Compare side-by-side with same prompts
3. 🎯 **Automatic Optimization** - Auto-rewrite prompts for new models
4. 📊 **Performance Comparison** - Quality, latency, cost analysis
5. 🚀 **Gradual Migration** - Risk-free A/B testing and rollout
6. ✨ **Production Deployment** - Zero downtime, zero code changes

**Key Benefit:** One API for all models. Switch providers, upgrade models, optimize costs — without rewriting code.

## Installation

```bash
git clone https://github.com/ahdbilal/mlflowlite.git
cd mlflowlite
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

## Setup

```bash
cp .env.example .env
# Add your API key to .env
```

## Quick Start

```bash
./start_notebook.sh
# Open MLflowlite_Demo.ipynb
```

**Note:** The demo notebook uses SQLite for MLflow tracking (required for `mlflow.genai.optimize_prompts()`). The database file `mlflow.db` is created automatically.

## Features

### 1. Automatic Tracing
Every LLM call logged to MLflow. Zero config.

```python
import mlflowlite as ml

response = ml.query(model='claude-3-5-sonnet', prompt='Hello')
print(f"Cost: ${response.cost:.4f}")
print(f"Latency: {response.latency:.2f}s")
```

### 2. Prompt Versioning
Git-like version control for prompts.

```python
from mlflowlite import Agent

agent = Agent(name="bot", model="claude-3-5-sonnet")
result_v1 = agent.run("Query")

agent.prompt_registry.add_version(
    system_prompt="Improved prompt",
    user_template="{query}",
    examples=[],
    metadata={"change": "More concise"}
)
result_v2 = agent.run("Query")

# Compare
print(f"v1: {result_v1.trace.total_tokens} tokens")
print(f"v2: {result_v2.trace.total_tokens} tokens")
```

### 3. AI-Powered Optimization
Get specific improvement suggestions.

```python
ml.set_suggestion_provider("claude-3-5-sonnet")
ml.print_suggestions(response)
```

### 4. Smart Routing
Automatically select the best model based on query complexity.

```python
# Analyzes complexity and picks the right model
decision, response = ml.smart_query("Explain quantum computing")
print(f"Selected {decision.model}: {decision.reason}")
```

### 5. A/B Testing
Data-driven model and prompt optimization.

```python
test = ml.create_ab_test(
    name="model_test",
    variants={
        'A': {'model': 'gpt-4o'},
        'B': {'model': 'claude-3-5-sonnet'}
    }
)
variant, response = test.run(messages=[...])
test.print_report()  # See which performs better
```

### 6. GenAI Evaluation
MLflow-style evaluation with built-in and custom scorers. Systematically measure quality at scale.

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
    return ml.completion(
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
df = results.to_dataframe()
```

**Built-in Scorers:**
- `Correctness` - Check factual accuracy using LLM-as-a-Judge
- `Guidelines` - Check adherence to natural language rules
- `Relevance` - Check if response is on-topic
- `Faithfulness` - Check for hallucinations (RAG apps)
- `Conciseness` - Check response length

**Evaluate Traces:** Retroactively evaluate captured traces
```python
from mlflowlite import evaluate_with_traces

results = evaluate_with_traces(
    traces=agent_traces,
    scorers=[correctness_scorer, english_scorer],
)
```

## API

### Main Functions

```python
import mlflowlite as ml

# Simple query
response = ml.query(model='claude-3-5-sonnet', prompt='...', input='...')

# LiteLLM-style completion
response = ml.completion(model='claude-3-5-sonnet', messages=[...])

# With reliability features
response = ml.completion(
    model='claude-3-5-sonnet',
    messages=[...],
    timeout=30,                              # seconds
    max_retries=5,                           # retry attempts
    fallback_models=['gpt-4o', 'gpt-3.5-turbo']  # fallback chain
)

# Batch processing
responses = ml.batch_completion(model='claude-3-5-sonnet', messages_list=[...])

# Suggestions
ml.print_suggestions(response, use_llm=False)  # heuristic
ml.print_suggestions(response, use_llm=True)   # LLM-powered

# Configuration
ml.set_mlflow_tracking(enabled=True)
ml.set_suggestion_provider("claude-3-5-sonnet")
ml.set_timeout(60)                           # default timeout
ml.set_max_retries(3)                        # default retries
ml.set_fallback_models(['gpt-4o', 'gpt-3.5-turbo'])  # default fallbacks
```

### Response Object (OpenAI-Compatible)

The Response object is fully compatible with OpenAI's API format, making it a drop-in replacement:

```python
# OpenAI-compatible access
response.id                              # Unique completion ID
response.object                          # "chat.completion"
response.created                         # Unix timestamp
response.model                           # Model used
response.choices[0]["message"]["content"]  # Response text (OpenAI format)
response.usage["total_tokens"]           # Token usage (OpenAI format)

# Convenience attributes (same data, easier access)
response.content      # Direct access to response text
response.latency      # Response time in seconds
response.cost         # Estimated cost in USD
response.scores       # Quality scores dict
response.trace_id     # MLflow trace ID

# Convert to OpenAI JSON format
response.to_dict()    # Full OpenAI-compatible dictionary
```

**Drop-in replacement example:**
```python
# Code expecting OpenAI SDK format works unchanged!
def process_openai_response(response):
    content = response.choices[0]["message"]["content"]
    tokens = response.usage["total_tokens"]
    return {"content": content, "tokens": tokens}

# Works with mlflowlite responses!
response = ml.completion(model="gpt-4", messages=[...])
result = process_openai_response(response)  # ✅ Works!
```

### Agent Class

```python
from mlflowlite import Agent

agent = Agent(
    name="my_agent",
    model="claude-3-5-sonnet",
    system_prompt="You are a helpful assistant",
    tools=[]  # optional: ['calculator', 'search']
)

result = agent.run("Your query", evaluate=True)
print(result.trace.summary())
```

## View Traces

```bash
mlflow ui
# Open http://localhost:5000
```

## Models Supported

- OpenAI: `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`
- Anthropic: `claude-3-5-sonnet`, `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`
- Google: `gemini-pro`, `gemini-1.5-pro`
- Mistral: `mistral-large`, `mistral-medium`, `mistral-small`

## Examples

- **`MLflowlite_Demo.ipynb`** - Start here! Interactive demo with all features
- `complete_demo.py` - Full Python demo
- `examples/quick_start.py` - Minimal example
- `examples/openai_compatibility_demo.py` - OpenAI API format compatibility
- `examples/reliability_demo.py` - Retry, timeout, fallbacks
- `examples/routing_demo.py` - Smart routing & A/B testing
- `examples/evaluation_demo.py` - GenAI evaluation with scorers

## Troubleshooting

**Multiple experiments?** Update to latest version - everything now uses one `llm_workspace` experiment.

**Traces not showing?** Check: Machine Learning → Tracing (or notebook sidebar in Databricks).

**API errors?** Make sure your API key is set: `os.environ['ANTHROPIC_API_KEY'] = 'your-key'`

**Custom experiment name?** Use `ml.set_experiment_name('your_experiment_name')`

## Structure

```
mlflowlite/
├── __init__.py
├── litellm_style_api.py    # Main API
├── agent.py                # Advanced agent
├── llm/                    # LLM providers
├── tools/                  # Tool framework
├── tracing/                # MLflow integration
├── prompts/                # Prompt versioning
├── evaluation/             # Scoring
└── optimization/           # DSPy
```

## License

Apache 2.0
