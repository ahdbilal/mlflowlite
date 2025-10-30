#!/usr/bin/env python3
"""
Test script to verify all key notebook cells work correctly.
Run this before running the actual notebook to catch issues early.
"""

import os
import sys

# Set API key
api_key = os.getenv('ANTHROPIC_API_KEY')
if not api_key:
    print("❌ ANTHROPIC_API_KEY not set!")
    print("Set it with: export ANTHROPIC_API_KEY='your-key'")
    sys.exit(1)

os.environ['ANTHROPIC_API_KEY'] = api_key

print("=" * 60)
print("Testing mlflowlite Demo Notebook Components")
print("=" * 60)

# Clean up old experiments
import shutil
if os.path.exists('mlruns'):
    shutil.rmtree('mlruns')
    print("✅ Cleaned up old experiments\n")

# Test 1: Basic import and query
print("\n📝 Test 1: Basic Query")
print("-" * 60)
try:
    import mlflowlite as mla
    
    support_ticket = """
    Customer: Sarah Johnson
    Issue: Cannot log into the mobile app after updating to version 2.1
    Priority: High
    Previous tickets: 0
    Account type: Premium
    """
    
    response1 = mla.query(
        model='claude-3-5-sonnet',
        prompt='Summarize this support ticket in 2 sentences',
        input=support_ticket
    )
    
    print(f"✅ Response received: {len(response1.content)} chars")
    print(f"✅ Cost: ${response1.cost:.4f}")
    print(f"✅ Latency: {response1.latency:.2f}s")
    print(f"✅ Trace ID: {response1.trace_id}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

# Test 2: Print response
print("\n📝 Test 2: Print Response")
print("-" * 60)
try:
    mla.print_response(response1)
    print("✅ print_response() works")
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

# Test 3: Agent with prompt versioning
print("\n📝 Test 3: Agent with Prompt Versioning")
print("-" * 60)
try:
    from mlflowlite import Agent
    
    agent = Agent(
        name="support_bot",
        model="claude-3-5-sonnet",
        system_prompt="You are a helpful support agent. Be concise."
    )
    
    result_v1 = agent.run("How do I reset my password?")
    print(f"✅ v1 Response: {len(result_v1.response)} chars, {result_v1.trace.total_tokens} tokens")
    
    # Add new version
    agent.prompt_registry.add_version(
        system_prompt="You are a support agent. Answer in 1 sentence maximum.",
        user_template="{query}",
        examples=[],
        metadata={"change": "More concise, 1 sentence max"}
    )
    
    result_v2 = agent.run("How do I reset my password?")
    print(f"✅ v2 Response: {len(result_v2.response)} chars, {result_v2.trace.total_tokens} tokens")
    print(f"✅ Token reduction: {result_v1.trace.total_tokens - result_v2.trace.total_tokens} tokens")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Suggestions
print("\n📝 Test 4: Improvement Suggestions")
print("-" * 60)
try:
    mla.set_suggestion_provider("claude-3-5-sonnet")
    suggestions = mla.suggest_improvement(response1)
    print("✅ Suggestions generated")
    mla.print_suggestions(response1)
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

# Test 5: Reliability features
print("\n📝 Test 5: Reliability Features")
print("-" * 60)
try:
    # Global config
    mla.set_timeout(30)
    mla.set_max_retries(3)
    mla.set_fallback_models(['claude-3-haiku'])
    
    response_reliable = mla.query(
        model='claude-3-5-sonnet',
        prompt='Test reliability',
        timeout=15,
        max_retries=2
    )
    print(f"✅ Reliable query: {len(response_reliable.content)} chars")
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

# Test 6: Smart routing
print("\n📝 Test 6: Smart Routing")
print("-" * 60)
try:
    decision, response = mla.smart_query(
        "What is 2+2?",
        model_tier='auto'
    )
    print(f"✅ Routed to: {decision.model}")
    print(f"✅ Reason: {decision.reason}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

# Test 7: A/B Testing
print("\n📝 Test 7: A/B Testing")
print("-" * 60)
try:
    test = mla.create_ab_test(
        name="model_comparison",
        variants={
            'A': {'model': 'claude-3-5-sonnet'},
            'B': {'model': 'claude-3-haiku'}
        }
    )
    
    for i in range(2):
        variant, response = test.run(
            messages=[{"role": "user", "content": f"Test {i}"}]
        )
        print(f"✅ Test {i}: variant {variant}")
    
    report = test.get_report()
    print(f"✅ A/B test report generated: {len(report['variants'])} variants")
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("🎉 ALL TESTS PASSED!")
print("=" * 60)
print("\nThe notebook should work correctly now.")
print("Run: ./start_notebook.sh")

