"""
mlflowlite Reliability Features Demo
Shows: Retry | Timeout | Fallbacks
"""

from dotenv import load_dotenv
import mlflowlite as ml

load_dotenv()

print("=" * 60)
print("mlflowlite - Reliability Features Demo")
print("=" * 60)

# Example 1: Basic usage with default settings
print("\n1️⃣ Basic usage (default: 60s timeout, 3 retries)")
print("-" * 60)

response = ml.query(
    model="claude-3-5-sonnet",
    prompt="Explain retry patterns in one sentence"
)
print(f"✅ Response: {response.content}")
print(f"   Latency: {response.latency:.2f}s | Cost: ${response.cost:.4f}")

# Example 2: Custom timeout
print("\n\n2️⃣ With custom timeout (30 seconds)")
print("-" * 60)

try:
    response = ml.query(
        model="claude-3-5-sonnet",
        prompt="What is exponential backoff?",
        timeout=30  # 30 second timeout
    )
    print(f"✅ Response: {response.content[:100]}...")
    print(f"   Latency: {response.latency:.2f}s")
except TimeoutError as e:
    print(f"⏱️  Timeout: {e}")

# Example 3: With retries
print("\n\n3️⃣ With custom retry logic (5 attempts)")
print("-" * 60)

response = ml.query(
    model="claude-3-5-sonnet",
    prompt="Explain circuit breaker pattern briefly",
    max_retries=5,  # Will retry up to 5 times on failure
    timeout=20
)
print(f"✅ Response: {response.content[:100]}...")

# Example 4: With fallback models
print("\n\n4️⃣ With fallback models (automatic failover)")
print("-" * 60)

try:
    response = ml.completion(
        model="claude-3-5-sonnet",  # Primary model
        messages=[{"role": "user", "content": "What is graceful degradation?"}],
        fallback_models=["gpt-4o", "gpt-3.5-turbo"],  # Fallbacks if primary fails
        timeout=15,
        max_retries=2
    )
    print(f"✅ Used model: {response.model}")
    print(f"   Response: {response.content[:100]}...")
    print(f"   Metadata: {response.metadata}")
except Exception as e:
    print(f"❌ All models failed: {e}")

# Example 5: Set global defaults
print("\n\n5️⃣ Setting global reliability defaults")
print("-" * 60)

# Configure defaults for all subsequent calls
ml.set_timeout(45)
ml.set_max_retries(4)
ml.set_fallback_models(["claude-3-5-sonnet", "gpt-4o", "gpt-3.5-turbo"])

print("✅ Global settings configured:")
print("   • Timeout: 45s")
print("   • Max retries: 4")
print("   • Fallback chain: claude-3-5-sonnet → gpt-4o → gpt-3.5-turbo")

response = ml.query(
    model="claude-3-5-sonnet",
    prompt="What is fault tolerance?"
)
print(f"\n   Response: {response.content[:100]}...")

# Example 6: Reliability with tracing
print("\n\n6️⃣ All reliability attempts are traced in MLflow")
print("-" * 60)
print("✅ Every retry, timeout, and fallback is logged")
print("   View traces: mlflow ui (http://localhost:5000)")

print("\n" + "=" * 60)
print("💡 Key Benefits:")
print("   • Automatic retry with exponential backoff")
print("   • Configurable timeouts per request or globally")
print("   • Fallback chains for high availability")
print("   • All attempts traced in MLflow for debugging")
print("=" * 60)

