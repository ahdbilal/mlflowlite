"""
Advanced routing capabilities for mlflowlite
Includes smart routing and A/B testing
"""

import random
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib

from mlflowlite.litellm_style_api import completion, Response


@dataclass
class RoutingDecision:
    """Decision made by the router"""
    model: str
    reason: str
    complexity_score: float
    estimated_cost: float


class SmartRouter:
    """
    Smart routing: Automatically selects the best model based on query complexity.
    
    Rules:
    - Simple queries → Fast, cheap models (gpt-3.5-turbo, claude-haiku)
    - Medium queries → Balanced models (claude-3-5-sonnet, gpt-4o)
    - Complex queries → High-quality models (gpt-4o, claude-opus)
    """
    
    # Model tiers by capability and cost
    TIERS = {
        'fast': ['gpt-3.5-turbo', 'claude-3-haiku'],
        'balanced': ['claude-3-5-sonnet', 'gpt-4o'],
        'quality': ['gpt-4o', 'claude-3-opus']
    }
    
    def __init__(self, default_tier: str = 'balanced'):
        self.default_tier = default_tier
    
    def analyze_complexity(self, messages: List[Dict[str, str]]) -> float:
        """
        Analyze query complexity (0-1 scale).
        
        Factors:
        - Length of input
        - Number of questions
        - Technical keywords
        - Request for reasoning
        """
        # Combine all message content
        text = ' '.join([msg.get('content', '') for msg in messages]).lower()
        
        score = 0.3  # Base score
        
        # Length factor (longer = more complex)
        if len(text) > 500:
            score += 0.2
        elif len(text) > 200:
            score += 0.1
        
        # Multiple questions
        question_count = text.count('?')
        if question_count > 2:
            score += 0.15
        elif question_count > 0:
            score += 0.05
        
        # Complexity keywords
        complex_keywords = [
            'analyze', 'explain', 'compare', 'evaluate', 'reasoning',
            'step by step', 'detailed', 'comprehensive', 'complex'
        ]
        for keyword in complex_keywords:
            if keyword in text:
                score += 0.1
                break
        
        # Simple task keywords (reduce score)
        simple_keywords = ['summarize', 'list', 'yes or no', 'true or false']
        for keyword in simple_keywords:
            if keyword in text:
                score -= 0.1
                break
        
        return min(max(score, 0.0), 1.0)  # Clamp between 0-1
    
    def select_model(
        self,
        messages: List[Dict[str, str]],
        complexity: Optional[str] = None,
        prefer_speed: bool = False,
        prefer_quality: bool = False
    ) -> RoutingDecision:
        """
        Select the best model for the query.
        
        Args:
            messages: Query messages
            complexity: Force complexity tier ('fast', 'balanced', 'quality', 'auto')
            prefer_speed: Prefer faster/cheaper models
            prefer_quality: Prefer higher quality models
        """
        if complexity and complexity != 'auto':
            tier = complexity
            reason = f"User specified tier: {complexity}"
            complexity_score = {'fast': 0.2, 'balanced': 0.5, 'quality': 0.8}[tier]
        else:
            # Analyze complexity
            complexity_score = self.analyze_complexity(messages)
            
            # Apply preferences
            if prefer_speed:
                complexity_score *= 0.7
            elif prefer_quality:
                complexity_score = min(complexity_score * 1.3, 1.0)
            
            # Map to tier
            if complexity_score < 0.35:
                tier = 'fast'
                reason = "Simple query → fast model"
            elif complexity_score < 0.65:
                tier = 'balanced'
                reason = "Medium complexity → balanced model"
            else:
                tier = 'quality'
                reason = "Complex query → high-quality model"
        
        # Select model from tier
        model = self.TIERS[tier][0]  # Use first model in tier
        
        # Estimate cost (rough)
        cost_per_tier = {'fast': 0.001, 'balanced': 0.003, 'quality': 0.01}
        estimated_cost = cost_per_tier[tier]
        
        return RoutingDecision(
            model=model,
            reason=reason,
            complexity_score=complexity_score,
            estimated_cost=estimated_cost
        )


class ABTest:
    """
    A/B testing: Split traffic between variants and track performance.
    
    Automatically tracks:
    - Cost per variant
    - Latency per variant
    - Quality scores per variant
    - Success rate per variant
    """
    
    def __init__(
        self,
        name: str,
        variants: Dict[str, Dict[str, Any]],
        split: Optional[List[float]] = None,
        sticky: bool = True  # Same user gets same variant
    ):
        """
        Initialize A/B test.
        
        Args:
            name: Test name
            variants: Dict of variant_name -> config (model, prompt, etc.)
            split: Traffic split ratios (default: equal split)
            sticky: If True, same input always gets same variant
        """
        self.name = name
        self.variants = variants
        self.variant_names = list(variants.keys())
        
        # Default to equal split
        if split is None:
            split = [1.0 / len(variants)] * len(variants)
        
        self.split = split
        self.sticky = sticky
        
        # Track stats
        self.stats = {
            name: {
                'count': 0,
                'total_cost': 0.0,
                'total_latency': 0.0,
                'total_tokens': 0,
                'scores': []
            }
            for name in self.variant_names
        }
    
    def select_variant(self, input_text: str = None) -> str:
        """Select variant based on split ratios."""
        if self.sticky and input_text:
            # Hash input to get consistent variant
            hash_val = int(hashlib.md5(input_text.encode()).hexdigest(), 16)
            cumulative = 0
            threshold = (hash_val % 1000) / 1000.0
            
            for variant, ratio in zip(self.variant_names, self.split):
                cumulative += ratio
                if threshold <= cumulative:
                    return variant
            return self.variant_names[-1]
        else:
            # Random selection based on split
            return random.choices(self.variant_names, weights=self.split)[0]
    
    def run(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Tuple[str, Response]:
        """
        Run A/B test and return variant name + response.
        
        Args:
            messages: Query messages
            **kwargs: Additional completion args
        """
        # Select variant
        input_text = messages[0].get('content', '') if messages else ''
        variant_name = self.select_variant(input_text)
        variant_config = self.variants[variant_name]
        
        # Merge variant config with kwargs
        call_args = {**variant_config, **kwargs}
        
        # Make the call
        start_time = time.time()
        response = completion(messages=messages, **call_args)
        
        # Track stats
        self.stats[variant_name]['count'] += 1
        self.stats[variant_name]['total_cost'] += response.cost
        self.stats[variant_name]['total_latency'] += response.latency
        self.stats[variant_name]['total_tokens'] += response.usage.get('total_tokens', 0)
        if response.scores:
            self.stats[variant_name]['scores'].append(response.scores)
        
        return variant_name, response
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance stats for all variants."""
        results = {}
        
        for variant_name in self.variant_names:
            stats = self.stats[variant_name]
            count = stats['count']
            
            if count == 0:
                results[variant_name] = {'count': 0, 'message': 'No data yet'}
                continue
            
            # Calculate averages
            avg_cost = stats['total_cost'] / count
            avg_latency = stats['total_latency'] / count
            avg_tokens = stats['total_tokens'] / count
            
            # Average scores
            avg_scores = {}
            if stats['scores']:
                score_keys = stats['scores'][0].keys()
                for key in score_keys:
                    avg_scores[key] = sum(s[key] for s in stats['scores']) / len(stats['scores'])
            
            results[variant_name] = {
                'count': count,
                'avg_cost': avg_cost,
                'avg_latency': avg_latency,
                'avg_tokens': avg_tokens,
                'avg_scores': avg_scores,
                'total_cost': stats['total_cost']
            }
        
        return results
    
    def get_winner(self, metric: str = 'cost') -> Tuple[str, Dict[str, Any]]:
        """
        Determine the winning variant based on a metric.
        
        Args:
            metric: 'cost', 'latency', 'quality' (avg of scores)
        """
        stats = self.get_stats()
        
        # Filter variants with data
        valid_variants = {k: v for k, v in stats.items() if v.get('count', 0) > 0}
        
        if not valid_variants:
            return None, {'message': 'No data to determine winner'}
        
        if metric == 'cost':
            winner = min(valid_variants.items(), key=lambda x: x[1]['avg_cost'])
        elif metric == 'latency':
            winner = min(valid_variants.items(), key=lambda x: x[1]['avg_latency'])
        elif metric == 'quality':
            # Average all scores
            def avg_quality(stats):
                scores = stats['avg_scores']
                return sum(scores.values()) / len(scores) if scores else 0
            winner = max(valid_variants.items(), key=lambda x: avg_quality(x[1]))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return winner[0], winner[1]
    
    def print_report(self):
        """Print a formatted report of the A/B test results."""
        print("=" * 70)
        print(f"📊 A/B Test Report: {self.name}")
        print("=" * 70)
        
        stats = self.get_stats()
        
        for variant_name in self.variant_names:
            vstats = stats[variant_name]
            print(f"\n🔹 Variant: {variant_name}")
            print(f"   Config: {self.variants[variant_name]}")
            
            if vstats.get('count', 0) == 0:
                print("   Status: No data yet")
                continue
            
            print(f"   Requests: {vstats['count']}")
            print(f"   Avg Cost: ${vstats['avg_cost']:.4f}")
            print(f"   Avg Latency: {vstats['avg_latency']:.2f}s")
            print(f"   Avg Tokens: {vstats['avg_tokens']:.0f}")
            
            if vstats['avg_scores']:
                print(f"   Avg Scores: {vstats['avg_scores']}")
        
        # Show winners
        print(f"\n{'='*70}")
        print("🏆 Winners:")
        
        for metric in ['cost', 'latency', 'quality']:
            try:
                winner, winner_stats = self.get_winner(metric)
                if winner:
                    value = winner_stats.get(f'avg_{metric}', 'N/A')
                    print(f"   • Best {metric}: {winner} ({value})")
            except:
                pass
        
        print("=" * 70)


# Global router instance
_router = SmartRouter()


def smart_query(
    prompt: str,
    input: Optional[str] = None,
    complexity: str = 'auto',
    prefer_speed: bool = False,
    prefer_quality: bool = False,
    **kwargs
) -> Tuple[RoutingDecision, Response]:
    """
    Smart query with automatic model selection.
    
    Args:
        prompt: The prompt/instruction
        input: Optional input text
        complexity: 'auto', 'fast', 'balanced', or 'quality'
        prefer_speed: Prefer faster/cheaper models
        prefer_quality: Prefer higher quality models
        **kwargs: Additional completion args
    
    Returns:
        Tuple of (routing_decision, response)
    
    Example:
        >>> decision, response = smart_query("Explain quantum computing")
        >>> print(f"Selected: {decision.model} ({decision.reason})")
        >>> print(response.content)
    """
    # Build messages
    if input:
        content = f"{prompt}\n\nInput:\n{input}"
    else:
        content = prompt
    
    messages = [{"role": "user", "content": content}]
    
    # Get routing decision
    decision = _router.select_model(
        messages=messages,
        complexity=complexity,
        prefer_speed=prefer_speed,
        prefer_quality=prefer_quality
    )
    
    # Make the call with selected model
    response = completion(
        model=decision.model,
        messages=messages,
        **kwargs
    )
    
    return decision, response


def create_ab_test(
    name: str,
    variants: Dict[str, Dict[str, Any]],
    split: Optional[List[float]] = None,
    sticky: bool = True
) -> ABTest:
    """
    Create an A/B test.
    
    Args:
        name: Test name
        variants: Dict of variant_name -> config (must include 'model')
        split: Traffic split ratios (default: equal)
        sticky: Same input gets same variant
    
    Returns:
        ABTest instance
    
    Example:
        >>> test = create_ab_test(
        ...     name="prompt_test",
        ...     variants={
        ...         'A': {'model': 'gpt-4o', 'temperature': 0.7},
        ...         'B': {'model': 'claude-3-5-sonnet', 'temperature': 0.5}
        ...     }
        ... )
        >>> variant, response = test.run(messages=[...])
        >>> test.print_report()
    """
    return ABTest(name=name, variants=variants, split=split, sticky=sticky)


