#!/usr/bin/env python3
"""
Test script to debug optimize_prompts with Anthropic
"""
import os
import sys
import warnings
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('LiteLLM').setLevel(logging.ERROR)
logging.getLogger('mlflow').setLevel(logging.ERROR)
logging.getLogger('alembic').setLevel(logging.ERROR)

# Force local mode
os.environ['MLFLOW_ENABLE_UC_PROMPTS'] = 'false'

# Set API key from environment or use placeholder
if 'ANTHROPIC_API_KEY' not in os.environ:
    print("⚠️  ANTHROPIC_API_KEY not set. Please set it:")
    print("   export ANTHROPIC_API_KEY='your-key-here'")
    print("\nContinuing with placeholder (will fail at API call)...\n")
    os.environ['ANTHROPIC_API_KEY'] = 'placeholder'

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Reload mlflowlite
if 'mlflowlite' in sys.modules:
    del sys.modules['mlflowlite']

from mlflowlite import Agent

print("="*80)
print("TESTING OPTIMIZE_PROMPTS WITH ANTHROPIC")
print("="*80)

# Step 1: Create baseline data (simplified - just 3 examples)
print("\n1️⃣  Creating baseline data with production model...\n")

test_cases = [
    "User cannot login, getting 403 error",
    "Feature request: add dark mode to dashboard",
    "Database timeout when running large reports"
]

# Production agent (Sonnet 4.0)
prod_prompt = """You are a support ticket analyzer. Analyze each ticket and respond with:

**PRIORITY**: P0-Critical, P1-High, P2-Medium, or P3-Low
**CATEGORY**: Authentication, Database, API, UI, Performance, or Other
**TEAM**: Backend, Frontend, DevOps, or Security
**SUMMARY**: One-line summary
**ACTION**: Immediate next step"""

prod_agent = Agent(
    model='claude-sonnet-4-20250514',
    prompt=prod_prompt,
    prompt_name='ticket_router'
)

baseline_outputs = []
print("   Running production model...")
for i, ticket in enumerate(test_cases, 1):
    try:
        result = prod_agent(ticket=ticket)
        baseline_outputs.append(result.content)
        print(f"   ✅ {i}/3 complete")
    except Exception as e:
        print(f"   ❌ Error on case {i}: {e}")
        sys.exit(1)

print(f"\n   ✅ Collected {len(baseline_outputs)} baseline outputs")

# Step 2: Create dataset with expectations
print("\n2️⃣  Creating MLflow dataset with expectations...\n")

from mlflow.genai.datasets import create_dataset

dataset = create_dataset(name="ticket_router_test")

records = []
for i, ticket in enumerate(test_cases):
    records.append({
        "inputs": {"ticket": ticket},
        "outputs": baseline_outputs[i],
        "expectations": {
            "expected_response": baseline_outputs[i]  # For Equivalence scorer
        }
    })

dataset.merge_records(records)
print(f"   ✅ Dataset created with {len(records)} records (with expectations)")

# Step 3: Register prompt as template
print("\n3️⃣  Registering prompt with {{ticket}} placeholder...\n")

# Add {{ticket}} placeholder for MLflow template
prompt_template = prod_prompt + "\n\nAnalyze this ticket: {{ticket}}"

registered_prompt = mlflow.genai.register_prompt(
    name="ticket_router",
    template=prompt_template,
    commit_message="Test prompt"
)

print(f"   ✅ Registered: {registered_prompt.uri}")
print(f"   ✅ Template includes {{{{ticket}}}} placeholder")

# Step 4: Define prediction function following MLflow pattern
print("\n4️⃣  Defining prediction function...\n")

@mlflow.trace
def predict_fn(ticket, prompts=None):
    """Prediction function following MLflow docs pattern"""
    # Get prompt (base or optimized from MLflow)
    prompt = prompts.get('ticket_router') if prompts else registered_prompt
    
    # CRITICAL: Call .format() so MLflow tracks prompt usage!
    formatted_prompt = prompt.format(ticket=ticket)
    
    # Create agent with formatted prompt and call it
    # Since prompt is fully formatted, pass it as the query
    agent = Agent(
        model='claude-3-5-haiku-20241022'
    )
    
    # Call agent with the formatted prompt as input
    result = agent(prompt=formatted_prompt)
    return result.content

print("   ✅ predict_fn defined")
print("   ✅ Calls prompt.format(ticket=ticket) - key for MLflow tracking!")

# Step 5: Test prediction function
print("\n5️⃣  Testing prediction function before optimization...\n")

try:
    test_result = predict_fn(ticket=test_cases[0])
    print(f"   ✅ Test passed! Output: {test_result[:50]}...")
except Exception as e:
    print(f"   ❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Run optimize_prompts with Anthropic
print("\n6️⃣  Running optimize_prompts with Anthropic Claude as judge...\n")

from mlflow.genai.optimize import GepaPromptOptimizer
from mlflow.genai.scorers import Equivalence

print("   Model: Claude 3.5 Haiku (target)")
print("   Judge: Claude Sonnet 4.0 (Anthropic)")
print("   Scorer: Equivalence (semantic similarity)")
print("   All-Anthropic stack!")
print()

try:
    print("   🔍 Debug: Dataset structure check...")
    print(f"      Records: {len(records)}")
    print(f"      First record keys: {list(records[0].keys())}")
    print(f"      First record inputs: {list(records[0]['inputs'].keys())}")
    print(f"      First record expectations: {list(records[0]['expectations'].keys())}")
    print()
    
    optimization_result = mlflow.genai.optimize_prompts(
        predict_fn=predict_fn,
        train_data=dataset,
        prompt_uris=[registered_prompt.uri],
        optimizer=GepaPromptOptimizer(
            reflection_model="anthropic:/claude-sonnet-4-20250514"
        ),
        scorers=[
            Equivalence(model="anthropic:/claude-sonnet-4-20250514")
        ]
    )
    
    print("\n" + "="*80)
    print("✅ OPTIMIZATION SUCCESSFUL!")
    print("="*80)
    
    optimized_prompt = optimization_result.optimized_prompts[0]
    print(f"\n📝 Original prompt (first 150 chars):")
    print(f"   {prompt_template[:150]}...")
    print(f"\n📝 Optimized prompt (first 150 chars):")
    print(f"   {optimized_prompt.template[:150]}...")
    
except Exception as e:
    print("\n" + "="*80)
    print("❌ OPTIMIZATION FAILED!")
    print("="*80)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("ALL TESTS PASSED! ✅")
print("="*80)

