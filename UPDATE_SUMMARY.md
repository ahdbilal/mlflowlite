# 🎉 Major Update: Unified Interface

## What Changed

We've simplified mlflowlite to **ONE concept**: `Agent`

### Before (Confusing):
```python
# Two different ways to do the same thing?
response = query(model='...', prompt='...')  # Simple
agent = Agent(name='...', model='...')       # Advanced
result = agent.run('...')
```

### After (Clear):
```python
# ONE interface for everything!
agent = Agent(model='claude-3-5-sonnet')

# Simple usage (replaces query)
response = agent("What is 2+2?")
response.print_links()

# Advanced usage (when you need versioning)
agent = Agent(model='...', name="bot")  # name is optional!
result = agent.run("Complex query")
```

## Key Changes

1. **`Agent` is now callable**: `agent("prompt")` works!
2. **`name` is optional**: Only needed for prompt versioning
3. **Progressive disclosure**: Start simple, add features when needed
4. **Backward compatible**: `query()` still works (creates Agent internally)

## Updated Files

✅ `mlflowlite/agent.py` - Added `__call__` method
✅ `MLflowlite_Demo.ipynb` - Shows unified interface
✅ `UNIFIED_INTERFACE.md` - Documentation
✅ All examples still work

## Migration Path

### No breaking changes!

```python
# Old way still works
from mlflowlite import query
response = query(model='...', prompt='...')

# New way (recommended)
from mlflowlite import Agent
agent = Agent(model='...')
response = agent("...")
```

## Why This Is Better

1. **One concept to learn**: Just `Agent`
2. **More intuitive**: Create once, use many times
3. **Clearer progression**: Simple → Advanced is obvious
4. **More Pythonic**: Callable objects are standard

## Try It Now

```bash
cd /Users/ahmed.bilal/Desktop/gateway-oss
git pull origin main
./start_notebook.sh
```

Open `MLflowlite_Demo.ipynb` and see the new unified interface!

**One interface. Everything you need.** 🎯

