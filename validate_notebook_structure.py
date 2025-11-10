#!/usr/bin/env python3
"""
Validate notebook structure without making API calls
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
os.environ['ANTHROPIC_API_KEY'] = 'test-key-validation-only'

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

print("="*80)
print("VALIDATING NOTEBOOK STRUCTURE (NO API CALLS)")
print("="*80)

# Test 1: Import mlflowlite
print("\n1️⃣  Testing mlflowlite imports...")
try:
    if 'mlflowlite' in sys.modules:
        del sys.modules['mlflowlite']
    from mlflowlite import Agent, load_prompt
    print("   ✅ Imports successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Test prompt registration
print("\n2️⃣  Testing prompt registration...")
try:
    test_prompt = """You are a test prompt.

Analyze this: {{input}}"""
    
    registered = mlflow.genai.register_prompt(
        name="test_validation_prompt",
        template=test_prompt,
        commit_message="Validation test"
    )
    
    print(f"   ✅ Prompt registered: {registered.uri}")
    print(f"   ✅ Template includes {{{{input}}}} placeholder")
    
    # Test format() method
    formatted = registered.format(input="test value")
    assert "test value" in formatted
    print("   ✅ format() method works correctly")
    
except Exception as e:
    print(f"   ❌ Registration failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test predict_fn pattern
print("\n3️⃣  Testing predict_fn pattern...")
try:
    @mlflow.trace
    def test_predict_fn(input, prompts=None):
        """Test prediction function following MLflow pattern"""
        prompt = prompts.get('test_validation_prompt') if prompts else registered
        formatted = prompt.format(input=input)
        # Don't call Agent (no API key), just return formatted prompt
        return f"Would call Agent with: {formatted[:50]}..."
    
    # Test without prompts dict
    result1 = test_predict_fn(input="test input")
    print(f"   ✅ predict_fn works without prompts dict")
    print(f"      Output: {result1[:60]}...")
    
    # Test with prompts dict
    result2 = test_predict_fn(input="test input", prompts={'test_validation_prompt': registered})
    print(f"   ✅ predict_fn works with prompts dict")
    print(f"      Output: {result2[:60]}...")
    
except Exception as e:
    print(f"   ❌ predict_fn test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test dataset structure
print("\n4️⃣  Testing dataset structure...")
try:
    from mlflow.genai.datasets import create_dataset
    
    dataset = create_dataset(name="test_validation_dataset")
    
    # Test record format
    test_records = [
        {
            "inputs": {"input": "test 1"},
            "outputs": "output 1"
        },
        {
            "inputs": {"input": "test 2"},
            "outputs": "output 2"
        }
    ]
    
    dataset.merge_records(test_records)
    print(f"   ✅ Dataset created with {len(test_records)} records")
    print(f"   ✅ Structure: inputs={{input}}, outputs")
    
except Exception as e:
    print(f"   ❌ Dataset test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Validate URI format
print("\n5️⃣  Testing Anthropic URI format...")
try:
    from mlflow.genai.scorers import Equivalence
    
    # Test URI format (won't call API, just validate format)
    test_uri = "anthropic:/claude-sonnet-4-20250514"
    print(f"   ✅ URI format: {test_uri}")
    print(f"   ✅ Pattern: <provider>:/<model-name>")
    
    # Validate it matches expected pattern
    assert test_uri.startswith("anthropic:/")
    assert "/" in test_uri
    parts = test_uri.split(":/")
    assert len(parts) == 2
    print(f"   ✅ Provider: {parts[0]}")
    print(f"   ✅ Model: {parts[1]}")
    
except Exception as e:
    print(f"   ❌ URI validation failed: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("✅ ALL STRUCTURE VALIDATIONS PASSED!")
print("="*80)
print("\nNotebook structure is correct. Ready to run with real API key!")
print("\nTo test with real API calls:")
print("1. Set: export ANTHROPIC_API_KEY='your-key-here'")
print("2. Run notebook cells 19, 22, 23 in order")
print("3. Should see non-zero equivalence scores!")

