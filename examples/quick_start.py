"""
mlflowlite Quick Start
Minimal example showing basic usage.
"""

from dotenv import load_dotenv
import mlflowlite as mla

load_dotenv()

print("mlflowlite Quick Start")

# Example 1: Simple completion (LiteLLM-style)
print("\n1️⃣ Simple Completion")
print("-" * 60)

response = mla.completion(
    model="claude-3-5-sonnet",
    messages=[{"role": "user", "content": "What is 2 + 2?"}]
)

print(f"Response: {response.content}")
print(f"Cost: ${response.cost:.4f} | Latency: {response.latency:.2f}s")
print(f"Scores: {response.scores}")

# Example 2: Even simpler with query()
print("\n2️⃣ Simplified Query")
print("-" * 60)

response = mla.query(
    model="claude-3-5-sonnet",
    prompt="Summarize in one sentence",
    input="MLflow Agents provides automatic tracing and evaluation for LLM calls."
)

mla.print_response(response)

# Example 3: Advanced agent with tools
print("\n3️⃣ Agent with Tools")
print("-" * 60)

from mlflowlite import Agent

agent = Agent(
    name="calculator_agent",
    model="claude-3-5-sonnet",
    tools=["calculator"]
)

result = agent.run("What is 123 * 456?")
print(f"Response: {result.response}")

print("\n" + "=" * 60)
print("✅ All examples complete!")
print("📊 View traces: mlflow ui")
print("📓 Learn more: MLflow_Agents_Demo.ipynb")
print("=" * 60)
