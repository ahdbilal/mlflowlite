# ✨ Unified Interface: One Concept for Everything

We've unified `query()` and `Agent` into **one single concept**: `Agent`

## The Problem Before

```python
# Simple queries
response = query(model='claude-3-5-sonnet', prompt='Hello')

# Advanced workflows  
agent = Agent(name="bot", model='claude-3-5-sonnet')
result = agent.run("Hello")
```

**Why have two different concepts?** Confusing!

## The Solution: Just Use `Agent`

### Simple Queries (replaces `query()`)

```python
# Create once
agent = Agent(model='claude-3-5-sonnet')

# Use like a function
response = agent("What is 2+2?")
print(response.content)
response.print_links()  # See MLflow UI
```

### Advanced Workflows (prompt versioning + tools)

```python
# Add a name for versioning
agent = Agent(
    model='claude-3-5-sonnet',
    name="support_bot",  # Optional - only for versioning
    system_prompt="You are helpful",
    tools=["search"]
)

# Use run() for full features
result = agent.run("Help me troubleshoot")
print(result.response)

# Version your prompts
agent.prompt_registry.add_version(
    system_prompt="You are VERY helpful",
    metadata={"change": "More enthusiastic"}
)
result_v2 = agent.run("Help me troubleshoot")
```

## Why This Is Better

1. **One Concept**: Just learn `Agent`
2. **Flexible**: Simple or advanced, same interface
3. **Callable**: `agent(prompt)` - feels natural
4. **Progressive**: Start simple, add features when needed

## Migration Guide

### Before:
```python
from mlflowlite import query

response = query(model='claude-3-5-sonnet', prompt='Hello')
```

### After (Option 1 - Direct replacement):
```python
from mlflowlite import Agent

agent = Agent(model='claude-3-5-sonnet')
response = agent("Hello")  # Same Response object!
```

### After (Option 2 - Keep query() as syntactic sugar):
```python
from mlflowlite import query  # Still works!

response = query(model='claude-3-5-sonnet', prompt='Hello')
# query() now creates an Agent under the hood
```

## The Big Picture

**One unified concept that scales:**

```python
from mlflowlite import Agent

# Level 1: Quick query
agent = Agent(model='claude-3-5-sonnet')
response = agent("Hello")

# Level 2: Add system prompt
agent = Agent(
    model='claude-3-5-sonnet',
    system_prompt="You are helpful"
)
response = agent("Hello")

# Level 3: Add versioning
agent = Agent(
    model='claude-3-5-sonnet',
    name="my_bot",  # Now it tracks versions!
    system_prompt="You are helpful"
)
result = agent.run("Hello")
agent.prompt_registry.add_version(...)

# Level 4: Add tools
agent = Agent(
    model='claude-3-5-sonnet',
    name="my_bot",
    system_prompt="You are helpful",
    tools=["search", "calculator"]
)
result = agent.run("What's 2+2 and search for it")
```

**Same interface, growing capabilities!** 🎉

