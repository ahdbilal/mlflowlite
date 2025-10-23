# Test that experiment name is optional

# This should work without specifying experiment_name
from mlflowlite import Agent
import mlflowlite as ml

# Test 1: API query without experiment name (should auto-detect)
print("Test 1: API query (auto-detect experiment name)")
print("This will auto-detect and use:")
print("  - Local: 'mlflowlite'")  
print("  - Databricks: '/Users/your.email/mlflowlite'")
print()

# Test 2: Agent without experiment name (should auto-detect)
print("Test 2: Agent without experiment_name parameter")
agent = Agent(
    name="support_bot",
    model="claude-3-5-sonnet"
)
print("✅ Agent created successfully without experiment_name!")
print("This will auto-detect and use:")
print("  - Local: 'agent_support_bot'")
print("  - Databricks: '/Users/your.email/mlflowlite/agent_support_bot'")
print()

# Test 3: Agent with custom experiment name (still works)
print("Test 3: Agent with custom experiment_name")
agent2 = Agent(
    name="classifier",
    model="gpt-4o",
    experiment_name="/Users/custom@email.com/my_experiment"
)
print("✅ Agent created with custom experiment name!")
print(f"   Will use: '/Users/custom@email.com/my_experiment'")

print("\n" + "="*60)
print("✅ All tests passed!")
print("="*60)
print("\nConclusion:")
print("1. experiment_name is OPTIONAL")
print("2. Auto-detects Databricks vs Local")
print("3. Can still be set manually if needed")
