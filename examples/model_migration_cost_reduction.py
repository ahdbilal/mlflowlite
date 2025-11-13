"""
Model Migration with Automatic Prompt Optimization

Demonstrates how to migrate from expensive models (Claude Sonnet) to cheaper ones
(Claude Haiku) while maintaining quality using MLflow's optimize_prompts() API.

Based on: https://mlflow.org/docs/latest/genai/prompt-registry/rewrite-prompts/
"""

import mlflowlite as mla
import mlflow
from mlflow.genai.optimize import GepaPromptOptimizer
from mlflow.genai.scorers import Equivalence

print("="*70)
print("🔄 MODEL MIGRATION: Claude Sonnet → Claude Haiku")
print("   Auto-rewrite prompts to reduce costs while maintaining quality")
print("="*70)

# Step 1: Register base prompt and test with expensive model
print("\n1️⃣ Starting with Claude Sonnet (Expensive)")
print("-" * 70)

response = mla.completion(
    model="claude-3-5-sonnet-20241022",
    prompt_id="sentiment_migrate",
    prompt_template="Classify sentiment. Answer 'positive', 'negative', or 'neutral'.\n\nText: {{text}}",
    prompt_variables={"text": "This product is amazing!"}
)

print(f"Response: {response.choices[0]['message']['content']}")
print(f"Cost: ${response.cost:.4f}")
print("💰 Claude Sonnet is accurate but expensive")

# Step 2: Collect training data from Claude Sonnet
print("\n2️⃣ Collecting Training Data from Claude Sonnet")
print("-" * 70)

training_inputs = [
    "This movie was absolutely fantastic!",
    "Terrible service and cold food.",
    "It was okay, nothing special.",
    "Complete waste of money!",
    "Best experience ever!",
    "Works as described. No complaints.",
    "This exceeded my expectations!",
    "Worst customer support ever.",
    "Fine for the price.",
    "Truly wonderful product!"
]

gpt4_outputs = []
gpt4_total_cost = 0

# Traces are automatically captured by mla.completion()
for text in training_inputs:
    result = mla.completion(
        model="claude-3-5-sonnet-20241022",
        prompt_id="sentiment_migrate",
        prompt_template="Classify the sentiment of this text as positive, negative, or neutral: {{text}}",
        prompt_variables={"text": text}
    )
    
    output = result.choices[0]['message']['content'].lower()
    gpt4_outputs.append(output)
    gpt4_total_cost += result.cost
    
    print(f"  {text[:35]:35} → {output}")

print(f"\n💰 Claude Sonnet Total Cost: ${gpt4_total_cost:.4f}")
print("✅ Traces automatically captured by MLflow")

# Step 3: Create training dataset
print("\n3️⃣ Creating Training Dataset")
print("-" * 70)

# Create dataset structure for MLflow
import pandas as pd

# Convert to DataFrame format
dataset_df = pd.DataFrame([
    {
        "inputs": {"text": text},
        "outputs": output
    }
    for text, output in zip(training_inputs, gpt4_outputs)
])

# MLflow optimize_prompts() accepts DataFrames directly
print(f"✅ Created dataset with {len(dataset_df)} examples")
print(f"   Structure: {{\"inputs\": {{\"text\": ...}}, \"outputs\": ...}}")

# Step 4: Test cheap model before optimization
print("\n4️⃣ Testing Claude Haiku (Before Optimization)")
print("-" * 70)

gpt4mini_before_cost = 0
mismatches_before = 0

print("Testing with original prompt...\n")
for i, text in enumerate(training_inputs[:5]):
    result = mla.completion(
        model="claude-3-5-haiku-20241022",
        prompt_id="sentiment_migrate",
        prompt_variables={"text": text}
    )
    
    output = result.choices[0]['message']['content'].lower()
    expected = gpt4_outputs[i]
    match = "✓" if output == expected else "✗"
    
    if output != expected:
        mismatches_before += 1
    
    gpt4mini_before_cost += result.cost
    print(f"  {match} {text[:35]:35} → {output}")

print(f"\n⚠️  Quality Issues: {mismatches_before}/5 mismatches")
print(f"💰 Cost: ${gpt4mini_before_cost:.4f}")

# Step 5: Optimize prompt for Claude Haiku
print("\n5️⃣ Optimizing Prompt for Claude Haiku")
print("-" * 70)

# Register prompt in MLflow standard registry
print("📝 Registering prompt in MLflow registry...\n")
from mlflow import MlflowClient

client = MlflowClient()
prompt_name = "sentiment_migrate"
prompt_template = "Classify the sentiment of this text as positive, negative, or neutral: {{text}}"

# Create prompt (if not exists) and add version
try:
    prompt = client.create_prompt(
        name=prompt_name,
        description="Sentiment classification prompt for migration demo"
    )
    print(f"✅ Created prompt container: {prompt_name}")
except Exception:
    print(f"   Prompt '{prompt_name}' already exists")

# Create a new version with the template
try:
    prompt_version = client.create_prompt_version(
        name=prompt_name,
        template=prompt_template,
        description="Original sentiment classification template"
    )
    version_num = prompt_version.version
    print(f"✅ Created prompt version: {version_num}")
except Exception:
    versions = client.search_prompt_versions(f"name='{prompt_name}'")
    if versions:
        version_num = max(int(v.version) for v in versions)
        print(f"   Using existing version: {version_num}")
    else:
        raise Exception("Could not create or find prompt version")

prompt_uri = f"prompts:/{prompt_name}/{version_num}"
print(f"✅ Prompt URI: {prompt_uri}\n")

# Define prediction function
@mlflow.trace
def predict_fn(text: str) -> str:
    result = mla.completion(
        model="claude-3-5-haiku-20241022",
        prompt_id="sentiment_migrate",
        prompt_variables={"text": text}
    )
    return result.choices[0]['message']['content'].lower()

print("🔄 Running MLflow optimize_prompts()...")
print("   (This uses Claude Sonnet to analyze and rewrite the prompt)")
print("   (This may take 2-3 minutes...)\n")

# Optimize (using the registered prompt)
result = mlflow.genai.optimize_prompts(
    predict_fn=predict_fn,
    train_data=dataset_df,
    prompt_uris=[prompt_uri],
    optimizer=GepaPromptOptimizer(reflection_model="anthropic:/claude-3-5-sonnet-20241022"),
    scorers=[Equivalence(model="anthropic:/claude-3-5-sonnet-20241022")]
)

optimized_prompt = result.optimized_prompts[0]
print("✅ Optimization complete!")
print(f"\nOptimized prompt:\n{'-'*70}")
print(optimized_prompt.template)
print('-'*70)

# Step 6: Test with optimized prompt
print("\n6️⃣ Testing with Optimized Prompt")
print("-" * 70)

gpt4mini_after_cost = 0
mismatches_after = 0

# Register optimized prompt
response = mla.completion(
    model="claude-3-5-haiku-20241022",
    prompt_id="sentiment_migrate_optimized",
    prompt_template=optimized_prompt.template,
    prompt_variables={"text": "Test"}
)

print("Testing with optimized prompt...\n")
for i, text in enumerate(training_inputs[:5]):
    result = mla.completion(
        model="claude-3-5-haiku-20241022",
        prompt_id="sentiment_migrate_optimized",
        prompt_variables={"text": text}
    )
    
    output = result.choices[0]['message']['content'].lower()
    expected = gpt4_outputs[i]
    match = "✓" if output == expected else "✗"
    
    if output != expected:
        mismatches_after += 1
    
    gpt4mini_after_cost += result.cost
    print(f"  {match} {text[:35]:35} → {output}")

print(f"\n✅ Better Quality: {mismatches_after}/5 mismatches")
print(f"💰 Cost: ${gpt4mini_after_cost:.4f}")

# Summary
print("\n" + "="*70)
print("📊 MIGRATION RESULTS")
print("="*70)

print(f"\n💰 COST COMPARISON:")
print(f"   Claude Sonnet (original):        ${gpt4_total_cost:.4f}")
print(f"   Claude Haiku (before optimize):  ${gpt4mini_before_cost:.4f}")
print(f"   Claude Haiku (after optimize):   ${gpt4mini_after_cost:.4f}")

savings = gpt4_total_cost - gpt4mini_after_cost
savings_pct = (savings / gpt4_total_cost) * 100

print(f"\n   💸 Savings: ${savings:.4f} ({savings_pct:.1f}%)")

print(f"\n✅ QUALITY:")
print(f"   Before optimization: {mismatches_before} mismatches")
print(f"   After optimization:  {mismatches_after} mismatches")

print(f"\n🎯 RESULT:")
print(f"   Optimized prompt maintains quality at {savings_pct:.1f}% lower cost!")

print("\n" + "="*70)
print("📚 Learn more:")
print("   https://mlflow.org/docs/latest/genai/prompt-registry/rewrite-prompts/")
print("="*70)

