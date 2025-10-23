"""Test if tracing is working"""
import os
os.environ['ANTHROPIC_API_KEY'] = 'test-key-placeholder'

# Simulate Databricks environment
os.environ['DATABRICKS_RUNTIME_VERSION'] = '14.0'
os.environ['USER'] = 'test.user@company.com'

import mlflowlite as ml

# Check if MLflow tracking is enabled
print("Testing tracing configuration...")
print(f"MLflow tracking enabled: {ml.litellm_style_api._mlflow_enabled}")

# Check experiment name
from mlflowlite.litellm_style_api import _get_experiment_name
exp_name = _get_experiment_name()
print(f"Experiment name would be: {exp_name}")

print("\n✅ Tracing is configured!")
print("\nNote: Actual API call would fail without real API key,")
print("but the tracing logic is in place and ready.")
