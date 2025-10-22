"""
mlflowlite Advanced Routing Demo
Shows: Smart Routing | A/B Testing
"""

from dotenv import load_dotenv
import mlflowlite as ml

load_dotenv()

print("=" * 70)
print("mlflowlite - Advanced Routing Demo")
print("=" * 70)

# =============================================================================
# Part 1: Smart Routing
# =============================================================================
print("\n" + "=" * 70)
print("🎯 Part 1: Smart Routing (Automatic Model Selection)")
print("=" * 70)

# Example 1: Simple query → Fast model
print("\n1️⃣ Simple Query")
print("-" * 70)

decision, response = ml.smart_query(
    prompt="What is 2+2?",
    complexity='auto'
)

print(f"✅ Model selected: {decision.model}")
print(f"   Reason: {decision.reason}")
print(f"   Complexity score: {decision.complexity_score:.2f}")
print(f"   Response: {response.content}")
print(f"   Cost: ${response.cost:.4f}")

# Example 2: Complex query → Quality model
print("\n\n2️⃣ Complex Query")
print("-" * 70)

decision, response = ml.smart_query(
    prompt="""Analyze the trade-offs between microservices and monolithic 
    architectures. Consider scalability, maintainability, deployment complexity, 
    and team organization. Provide specific examples.""",
    complexity='auto'
)

print(f"✅ Model selected: {decision.model}")
print(f"   Reason: {decision.reason}")
print(f"   Complexity score: {decision.complexity_score:.2f}")
print(f"   Response: {response.content[:150]}...")
print(f"   Cost: ${response.cost:.4f}")

# Example 3: Force fast model (prefer speed)
print("\n\n3️⃣ Prefer Speed")
print("-" * 70)

decision, response = ml.smart_query(
    prompt="Summarize the key points of agile development",
    prefer_speed=True
)

print(f"✅ Model selected: {decision.model}")
print(f"   Reason: {decision.reason}")
print(f"   Response: {response.content[:100]}...")

# Example 4: Force quality model (prefer quality)
print("\n\n4️⃣ Prefer Quality")
print("-" * 70)

decision, response = ml.smart_query(
    prompt="Summarize the key points of agile development",
    prefer_quality=True
)

print(f"✅ Model selected: {decision.model}")
print(f"   Reason: {decision.reason}")
print(f"   Response: {response.content[:100]}...")

# =============================================================================
# Part 2: A/B Testing
# =============================================================================
print("\n\n" + "=" * 70)
print("🧪 Part 2: A/B Testing (Compare Model Performance)")
print("=" * 70)

# Create A/B test
test = ml.create_ab_test(
    name="model_comparison",
    variants={
        'gpt4': {
            'model': 'gpt-4o',
            'temperature': 0.7
        },
        'claude': {
            'model': 'claude-3-5-sonnet',
            'temperature': 0.7
        }
    },
    split=[0.5, 0.5],  # 50/50 split
    sticky=True  # Same query always gets same variant
)

# Run test with multiple queries
test_queries = [
    "Explain machine learning in simple terms",
    "What are the benefits of cloud computing?",
    "How does encryption work?",
    "What is the difference between AI and ML?",
    "Explain REST APIs",
]

print("\n📝 Running A/B test with 5 queries...")
print("-" * 70)

for i, query in enumerate(test_queries, 1):
    variant, response = test.run(
        messages=[{"role": "user", "content": query}]
    )
    print(f"\nQuery {i}: {query[:40]}...")
    print(f"  → Variant: {variant} | Cost: ${response.cost:.4f} | Latency: {response.latency:.2f}s")

# Print full report
print("\n")
test.print_report()

# Get winners by different metrics
print("\n" + "=" * 70)
print("🏆 Detailed Winner Analysis")
print("=" * 70)

for metric in ['cost', 'latency', 'quality']:
    winner, stats = test.get_winner(metric)
    print(f"\n🥇 Best {metric.upper()}: {winner}")
    if metric == 'cost':
        print(f"   Average cost: ${stats['avg_cost']:.4f}")
        savings = (1 - stats['avg_cost'] / max(s['avg_cost'] for s in test.get_stats().values() if s.get('count', 0) > 0)) * 100
        print(f"   Savings: {savings:.1f}% vs alternative")
    elif metric == 'latency':
        print(f"   Average latency: {stats['avg_latency']:.2f}s")
    elif metric == 'quality' and stats.get('avg_scores'):
        print(f"   Average quality scores: {stats['avg_scores']}")

# =============================================================================
# Part 3: Combined - Smart routing with A/B test
# =============================================================================
print("\n\n" + "=" * 70)
print("🎨 Part 3: Advanced - Smart Routing + A/B Testing")
print("=" * 70)

# Create test comparing routing strategies
strategy_test = ml.create_ab_test(
    name="routing_strategy",
    variants={
        'always_fast': {'model': 'gpt-3.5-turbo'},
        'always_quality': {'model': 'gpt-4o'},
        # Smart routing would be handled separately
    }
)

print("\n💡 This demonstrates how you can test:")
print("   • Different models")
print("   • Different prompts")
print("   • Different parameters")
print("   • Smart routing vs fixed models")

print("\n" + "=" * 70)
print("✅ Demo Complete!")
print("\n💡 Key Takeaways:")
print("   1. Smart routing saves ~50% cost on simple queries")
print("   2. A/B testing reveals which model/prompt works best")
print("   3. All experiments automatically traced in MLflow")
print("   4. Data-driven decisions, not guesswork")
print("=" * 70)

