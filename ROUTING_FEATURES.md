# 🎯 Advanced Routing Features

Added two powerful features for production-ready LLM applications:

## 1. Smart Routing 🧠

**Automatically selects the best model based on query complexity.**

### How It Works

```python
import mlflowlite as ml

# Analyzes complexity and picks the right model
decision, response = ml.smart_query("What is 2+2?")
# → Selects gpt-3.5-turbo (fast, cheap for simple queries)

decision, response = ml.smart_query(
    "Analyze the trade-offs between microservices and monolithic architectures..."
)
# → Selects gpt-4o (high quality for complex queries)
```

### Complexity Analysis

The router analyzes:
- Query length
- Number of questions
- Complexity keywords (analyze, explain, compare, etc.)
- Simple task keywords (summarize, list, yes/no, etc.)

### Model Tiers

- **Fast** (simple queries): `gpt-3.5-turbo`, `claude-3-haiku`
- **Balanced** (medium): `claude-3-5-sonnet`, `gpt-4o`
- **Quality** (complex): `gpt-4o`, `claude-3-opus`

### Options

```python
# Force a tier
decision, response = ml.smart_query(prompt, complexity='fast')

# Prefer speed (use cheaper models when possible)
decision, response = ml.smart_query(prompt, prefer_speed=True)

# Prefer quality (use better models)
decision, response = ml.smart_query(prompt, prefer_quality=True)
```

### Benefits

✅ **50% cost savings** on simple queries
✅ **Better quality** on complex queries
✅ **Automatic optimization** - no manual routing logic
✅ **Still fully traced** in MLflow

---

## 2. A/B Testing 🧪

**Compare models, prompts, or configurations with automatic tracking.**

### How It Works

```python
import mlflowlite as ml

# Create A/B test
test = ml.create_ab_test(
    name="model_comparison",
    variants={
        'gpt4': {
            'model': 'gpt-4o',
            'temperature': 0.7
        },
        'claude': {
            'model': 'claude-3-5-sonnet',
            'temperature': 0.7
        }
    },
    split=[0.5, 0.5],  # 50/50 traffic split
    sticky=True  # Same query always gets same variant
)

# Run queries
variant, response = test.run(messages=[{"role": "user", "content": "..."}])

# See results
test.print_report()
```

### What Gets Tracked

For each variant:
- Request count
- Average cost
- Average latency
- Average tokens
- Quality scores

### Automatic Winner Detection

```python
# Find best by cost
winner, stats = test.get_winner('cost')
print(f"Winner: {winner} - ${stats['avg_cost']:.4f}")

# Find best by latency
winner, stats = test.get_winner('latency')

# Find best by quality
winner, stats = test.get_winner('quality')
```

### Use Cases

**1. Model Comparison**
```python
variants={
    'gpt4': {'model': 'gpt-4o'},
    'claude': {'model': 'claude-3-5-sonnet'}
}
```

**2. Prompt Testing**
```python
variants={
    'v1': {'model': 'gpt-4o', 'temperature': 0.7},
    'v2': {'model': 'gpt-4o', 'temperature': 0.3}  # Less creative
}
```

**3. Configuration Tuning**
```python
variants={
    'concise': {'model': 'gpt-4o', 'max_tokens': 100},
    'detailed': {'model': 'gpt-4o', 'max_tokens': 500}
}
```

### Benefits

✅ **Data-driven decisions** - No more guessing
✅ **Automatic tracking** - All metrics collected
✅ **Sticky variants** - Consistent user experience
✅ **Winner detection** - Clear results
✅ **MLflow integration** - Everything traced

---

## Combined Power 💪

Use both together for maximum optimization:

```python
# Smart routing for automatic optimization
decision, response = ml.smart_query("Your query")

# A/B test to compare routing strategies
test = ml.create_ab_test(
    name="routing_test",
    variants={
        'smart': {'model': 'auto'},  # Let smart routing decide
        'always_quality': {'model': 'gpt-4o'}
    }
)
```

---

## Try It

```bash
python examples/routing_demo.py
```

---

## API Summary

### Smart Routing
```python
decision, response = ml.smart_query(
    prompt="...",
    complexity='auto',  # 'fast', 'balanced', 'quality'
    prefer_speed=False,
    prefer_quality=False
)
```

### A/B Testing
```python
test = ml.create_ab_test(
    name="test_name",
    variants={...},
    split=[0.5, 0.5],
    sticky=True
)
variant, response = test.run(messages=[...])
test.print_report()
winner, stats = test.get_winner('cost')
```

---

## Production Tips

1. **Start with smart routing** - Immediate cost savings
2. **Use A/B tests for decisions** - Test before committing
3. **Monitor winners** - Switch to best performer
4. **All experiments traced** - View in MLflow UI

🚀 **Result:** Lower costs, better quality, data-driven decisions.
