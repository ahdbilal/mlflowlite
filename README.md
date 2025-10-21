# mlflowlite

Easy LLM observability with automatic MLflow tracing.

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
# Fast heuristic
ml.print_suggestions(response)

# LLM-powered (smarter)
ml.print_suggestions(response, use_llm=True)
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

### Response Object

```python
response.content      # Response text
response.latency      # Seconds
response.cost         # USD
response.usage        # Token counts dict
response.scores       # Quality scores dict
response.trace_id     # MLflow trace ID
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

- `MLflowlite_Demo.ipynb` - Interactive demo
- `complete_demo.py` - Full Python demo
- `examples/quick_start.py` - Minimal example

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
