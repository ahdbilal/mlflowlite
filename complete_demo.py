"""
mlflowlite Complete Demo
Demonstrates: Tracing | Prompt Versioning | DSPy Optimization
"""

import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

import mlflowlite as mla
from mlflowlite import Agent
from mlflowlite.prompts.registry import PromptRegistry
import mlflow

print("=" * 80)
print("mlflowlite - Complete Demo")
print("=" * 80)

# Sample data
support_ticket = """
Subject: Cannot access dashboard after login

Hi Support,
I successfully log into the application, but when I click on 'Dashboard',
I get a blank screen. This started happening after yesterday's update.
I've tried clearing my browser cache and using a different browser,
but the issue persists.

Browser: Chrome 120
OS: macOS Ventura
Account: premium_user_12345
"""

# ============================================================================
# PART 1: AUTOMATIC TRACING
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: AUTOMATIC MLflow TRACING 📊")
print("=" * 80)

print("\n🔍 Every completion is automatically traced to MLflow!")
print("No manual instrumentation needed.\n")

# Make a simple call - automatically traced!
response1 = mla.query(
    model='claude-3-5-sonnet',
    prompt='Summarize this support ticket in 2 sentences',
    input=support_ticket
)

print(f"✅ Response: {response1.content}\n")
print(f"📊 Automatic Metrics Logged:")
print(f"   • Trace ID: {response1.trace_id}")
print(f"   • Latency: {response1.latency:.2f}s")
print(f"   • Cost: ${response1.cost:.4f}")
print(f"   • Tokens: {response1.usage.get('total_tokens', 0)}")
print(f"   • Model: {response1.model}")
print(f"   • Scores: {response1.scores}")

print(f"\n💡 View this trace:")
print(f"   mlflow ui")
print(f"   Then open: http://localhost:5000")
print(f"   Look for experiment: 'mlflowlite'")
print(f"   Find run: {response1.trace_id}")

# ============================================================================
# PART 2: PROMPT MANAGEMENT & VERSIONING
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: PROMPT MANAGEMENT & VERSIONING 📝")
print("=" * 80)

print("\n🎯 Using Prompt Registry for version control\n")

# Create an agent with prompt management
agent = Agent(
    name="support_bot",
    model="claude-3-5-sonnet",
    system_prompt="""You are a helpful support bot. Analyze support tickets and provide:
1. Quick summary
2. Root cause analysis
3. Recommended actions

Be concise and actionable.""",
    tools=[],  # No tools for now
)

print(f"✅ Created agent: {agent.name}")
print(f"📝 Prompt registry initialized")
print(f"   Version: {agent.prompt_registry.get_latest().version}")

# Run with version 1
result1 = agent.run(
    f"Analyze this ticket:\n\n{support_ticket}",
    evaluate=True
)

print(f"\n✅ Run 1 with prompt v{agent.prompt_registry.get_latest().version}:")
print(f"   Response: {result1.response[:150]}...")
print(f"   Tokens: {result1.trace.total_tokens}")
print(f"   Cost: ${result1.trace.total_cost:.4f}")

# Update the prompt (new version!)
print("\n📝 Updating prompt (creating new version)...")
agent.prompt_registry.add_version(
    system_prompt="""You are a support bot. For each ticket provide:
1. Issue summary (1 line)
2. Root cause (1 line)  
3. Fix (1-2 lines)

Be extremely concise.""",
    user_template="{query}",  # Agent uses 'query' not 'input'
    examples=[],
    metadata={"change": "Made more concise", "reason": "Reduce tokens"}
)

# Run with version 2
result2 = agent.run(
    f"Analyze this ticket:\n\n{support_ticket}",
    evaluate=True
)

print(f"\n✅ Run 2 with prompt v{agent.prompt_registry.get_latest().version}:")
print(f"   Response: {result2.response[:150]}...")
print(f"   Tokens: {result2.trace.total_tokens}")
print(f"   Cost: ${result2.trace.total_cost:.4f}")

# Compare versions
print("\n📊 Prompt Version Comparison:")
print(f"{'Metric':<20} {'v1':<20} {'v2':<20} {'Change':<20}")
print("-" * 80)
print(f"{'Tokens':<20} {result1.trace.total_tokens:<20} {result2.trace.total_tokens:<20} {result2.trace.total_tokens - result1.trace.total_tokens:<20}")
print(f"{'Cost':<20} ${result1.trace.total_cost:<19.4f} ${result2.trace.total_cost:<19.4f} ${result2.trace.total_cost - result1.trace.total_cost:<19.4f}")

# Show version history
print("\n📚 Prompt Version History:")
history = agent.prompt_registry.list_versions()
for item in history:
    version = item['version']
    change = item['metadata'].get('change', 'Initial version')
    print(f"   v{version}: {change}")

print(f"\n💡 All versions tracked in: {agent.prompt_registry.registry_path}")

# ============================================================================
# PART 3: DSPy-STYLE OPTIMIZATION
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: DSPy-STYLE PROMPT OPTIMIZATION 🧠")
print("=" * 80)

print("\n🎯 Using LLM to analyze traces and suggest improvements\n")

# Create test queries for optimization
test_queries = [
    "Summarize: User can't login",
    "Summarize: Dashboard shows error 500",
    "Summarize: Payment failed with timeout",
]

print("📝 Running test queries to collect traces...")
test_responses = []
for i, query in enumerate(test_queries, 1):
    print(f"   {i}. {query}")
    resp = mla.query(
        model='claude-3-5-sonnet',
        prompt=query,
        input=support_ticket
    )
    test_responses.append(resp)

print(f"\n✅ Collected {len(test_responses)} traces")

# Get LLM-powered suggestions (DSPy-style)
print("\n🧠 Running DSPy-style optimization...")
print("   (LLM analyzes all traces and suggests improvements)")

# Enable LLM-powered suggestions
mla.set_suggestion_provider("claude-3-5-sonnet")

print("\n📊 Individual Query Analysis:")
for i, resp in enumerate(test_responses[:2], 1):  # Analyze first 2
    print(f"\n--- Query {i} ---")
    suggestions = mla.suggest_improvement(resp)
    
    print(f"Metrics:")
    for metric, value in list(suggestions["current_metrics"].items())[:3]:
        if isinstance(value, float):
            print(f"  • {metric}: {value:.3f}")
        else:
            print(f"  • {metric}: {value}")
    
    print(f"\nTop 3 Suggestions:")
    for j, suggestion in enumerate(suggestions["improvements"][:3], 1):
        clean = suggestion.lstrip("0123456789.-) ").strip()
        print(f"  {j}. {clean}")

# Agent-level optimization
print("\n" + "-" * 80)
print("🔬 Agent-Level Optimization (Batch Analysis)")
print("-" * 80)

optimization_result = agent.optimize(
    num_runs=3,
    test_queries=test_queries
)

if optimization_result["status"] == "success":
    print("\n✅ Optimization Complete!")
    print(f"\n📊 Results:")
    print(f"   • Traces analyzed: {optimization_result['metadata']['num_traces_analyzed']}")
    avg_tokens = optimization_result['metadata'].get('avg_tokens', 'N/A')
    avg_latency = optimization_result['metadata'].get('avg_latency', None)
    print(f"   • Avg tokens: {avg_tokens}")
    if avg_latency is not None:
        print(f"   • Avg latency: {avg_latency:.2f}s")
    else:
        print(f"   • Avg latency: N/A")
    
    if optimization_result['patterns']['successful_patterns']:
        print(f"\n✨ Learned Patterns:")
        for pattern in optimization_result['patterns']['successful_patterns'][:3]:
            print(f"   • {pattern}")
    
    if optimization_result.get('suggested_prompt'):
        print(f"\n📝 Suggested Prompt:")
        print(f"   {optimization_result['suggested_prompt'][:200]}...")
        print(f"\n💡 You can apply this with:")
        print(f"   agent.prompt_registry.add_version(")
        print(f"       system_prompt='...', ")
        print(f"       user_template='{{query}}',  # Use {{query}} for Agent")
        print(f"       examples=[], ")
        print(f"       metadata={{'optimized': True}}")
        print(f"   )")
else:
    print(f"\n⚠️  Optimization status: {optimization_result['status']}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ COMPLETE DEMO FINISHED!")
print("=" * 80)

print(f"""
📊 What You Saw:

1️⃣  AUTOMATIC TRACING
   • Every LLM call is traced to MLflow automatically
   • No manual logging code needed
   • Tracks: latency, cost, tokens, model, scores
   • View traces: mlflow ui

2️⃣  PROMPT MANAGEMENT
   • Version control for prompts
   • Compare different prompt versions
   • Track what changes improved performance
   • Rollback to previous versions if needed

3️⃣  DSPy-STYLE OPTIMIZATION
   • LLM analyzes your query patterns
   • Suggests specific prompt improvements
   • Learns from successful patterns
   • Provides actionable recommendations

🚀 Next Steps:

   1. View traces:
      mlflow ui
      
   2. See prompt versions:
      ls {agent.prompt_registry.registry_path}
      
   3. Try optimization on your own queries!
      
   4. Apply suggested improvements and measure impact

💡 All of this happens automatically with minimal code!
""")

print("=" * 80)

