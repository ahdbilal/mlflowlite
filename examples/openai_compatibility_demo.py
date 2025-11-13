"""
Demo: OpenAI-Compatible API Format

Shows how mlflowlite.completion() returns OpenAI-compatible responses.
This makes it a drop-in replacement for OpenAI/LiteLLM APIs.
"""

import mlflowlite as mla
import json

print("=" * 70)
print("🔗 OpenAI-Compatible API Demo")
print("=" * 70)

# Example 1: Basic completion with OpenAI-compatible response
print("\n1️⃣  Basic Completion\n")

response = mla.completion(
    model="claude-3-5-sonnet",
    messages=[
        {"role": "user", "content": "Say hello in 3 words"}
    ]
)

print("✅ Response received!")
print(f"   ID: {response.id}")
print(f"   Model: {response.model}")
print(f"   Created: {response.created}")
print(f"   Object: {response.object}")

# Example 2: Access content the OpenAI way
print("\n2️⃣  OpenAI-Style Access\n")

# OpenAI format: response.choices[0].message.content
openai_content = response.choices[0]["message"]["content"]
print(f"OpenAI format: {openai_content}")

# Convenience format: response.content
convenience_content = response.content
print(f"Convenience:   {convenience_content}")

print(f"\n✅ Both return the same content: {openai_content == convenience_content}")

# Example 3: Usage statistics
print("\n3️⃣  Token Usage (OpenAI Format)\n")

print(f"Prompt tokens:     {response.usage['prompt_tokens']}")
print(f"Completion tokens: {response.usage['completion_tokens']}")
print(f"Total tokens:      {response.usage['total_tokens']}")

# Example 4: Serialize to JSON (OpenAI format)
print("\n4️⃣  JSON Serialization (OpenAI-Compatible)\n")

# Convert to dict (OpenAI format)
response_dict = response.to_dict()

print("OpenAI-compatible JSON:")
print(json.dumps({
    "id": response_dict["id"],
    "object": response_dict["object"],
    "created": response_dict["created"],
    "model": response_dict["model"],
    "choices": response_dict["choices"],
    "usage": response_dict["usage"]
}, indent=2))

# Example 5: MLflow enhancements (bonus!)
print("\n5️⃣  MLflow Enhancements (Bonus!)\n")

print(f"Latency:  {response.latency:.3f}s")
print(f"Cost:     ${response.cost:.4f}")
print(f"Trace ID: {response.trace_id}")

if response.scores:
    print("\nEvaluation Scores:")
    for metric, score in response.scores.items():
        print(f"  {metric}: {score:.2f}")

# Example 6: Drop-in replacement for OpenAI SDK
print("\n6️⃣  Drop-in Replacement Pattern\n")

def process_with_openai_format(response):
    """
    Function that expects OpenAI response format.
    Works with mlflowlite responses!
    """
    # Access using OpenAI's structure
    choice = response.choices[0]
    message = choice["message"]
    content = message["content"]
    finish_reason = choice["finish_reason"]
    
    return {
        "content": content,
        "finish_reason": finish_reason,
        "tokens": response.usage["total_tokens"]
    }

result = process_with_openai_format(response)
print(f"✅ Function expecting OpenAI format works!")
print(f"   Content: {result['content'][:50]}...")
print(f"   Tokens:  {result['tokens']}")
print(f"   Status:  {result['finish_reason']}")

# Example 7: Full OpenAI JSON structure
print("\n7️⃣  Complete OpenAI JSON Structure\n")

print("Full response (OpenAI + MLflow extensions):")
full_json = json.dumps(response.to_dict(), indent=2, default=str)
print(full_json[:500] + "..." if len(full_json) > 500 else full_json)

print("\n" + "=" * 70)
print("✅ mlflowlite is a drop-in replacement for OpenAI/LiteLLM!")
print("   • Same response format")
print("   • Plus MLflow tracing")
print("   • Plus automatic evaluation")
print("   • Plus cost tracking")
print("=" * 70)


