# ✨ Simplified API

We've removed the redundant `mla.` prefix! Now it's even simpler.

## Before (Redundant):
```python
import mlflowlite as mla

response = mla.query(model='claude-3-5-sonnet', prompt='Hello')
mla.print_response(response)
mla.print_suggestions(response)
```

## After (Clean):
```python
from mlflowlite import query, print_response, print_suggestions

response = query(model='claude-3-5-sonnet', prompt='Hello')
print_response(response)
print_suggestions(response)
```

## All You Need

```python
from mlflowlite import (
    query,                    # Make LLM calls
    print_response,           # Pretty print results
    print_suggestions,        # Get optimization advice
    set_suggestion_provider,  # Configure AI-powered suggestions
    set_timeout,              # Set global timeout
    set_max_retries,          # Set global retries
    set_fallback_models,      # Set fallback chain
    smart_query,              # Auto-select best model
    create_ab_test,           # A/B testing
    Agent,                    # For prompt versioning
)
```

## Why This Is Better

1. **Fewer characters**: `query()` vs `mla.query()`
2. **More Pythonic**: Direct imports are standard
3. **Clearer**: You see exactly what you're using
4. **Simpler**: One less thing to remember

## Quick Start

```python
from mlflowlite import query

response = query(
    model='claude-3-5-sonnet',
    prompt='Explain quantum computing'
)

print(response.content)
response.print_links()  # See where it's logged in MLflow UI
```

**That's it!** 🎉

