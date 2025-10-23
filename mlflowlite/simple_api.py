"""
Simple API for notebook-style AI queries.

This provides the streamlined experience:
- ai_query() for simple calls
- Automatic tracing and evaluation
- IntelliGen-style summaries
- Model comparison
"""

import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import mlflow

from mlflowlite.llm.providers import get_provider
from mlflowlite.llm.base import Message, MessageRole
from mlflowlite.tracing.mlflow_tracer import MLflowTracer
from mlflowlite.evaluation.evaluator import AgentEvaluator


@dataclass
class QueryResult:
    """Result from ai_query()."""
    response: str
    trace_id: str
    model: str
    summary: str
    metrics: Dict[str, Any]
    run_id: Optional[str] = None
    
    def __str__(self) -> str:
        return self.response
    
    def print_summary(self):
        """Print IntelliGen-style summary."""
        print("=" * 60)
        print("📊 Query Summary")
        print("=" * 60)
        print(self.summary)
        print("\n📈 Metrics:")
        for key, value in self.metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        print("=" * 60)


class SimpleAPI:
    """Simple API for AI queries with automatic tracing."""
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """Initialize Simple API."""
        self.tracking_uri = tracking_uri
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set default experiment
        try:
            mlflow.set_experiment("ai_gateway_queries")
        except Exception as e:
            # Handle deleted experiment case
            if "already exists in deleted state" in str(e) or "deleted experiment" in str(e):
                try:
                    # Try to restore the experiment
                    exp = mlflow.get_experiment_by_name("ai_gateway_queries")
                    if exp and exp.lifecycle_stage == "deleted":
                        mlflow.tracking.MlflowClient().restore_experiment(exp.experiment_id)
                        mlflow.set_experiment("ai_gateway_queries")
                except Exception:
                    # If restore fails, use a timestamped name
                    exp_name = f"ai_gateway_queries_{int(time.time())}"
                    mlflow.create_experiment(exp_name)
                    mlflow.set_experiment(exp_name)
            else:
                # Try to create new experiment
                try:
                    mlflow.create_experiment("ai_gateway_queries")
                    mlflow.set_experiment("ai_gateway_queries")
                except Exception:
                    pass  # Continue without experiment
        
        # Cache for query results (for comparison)
        self.query_history: List[QueryResult] = []
    
    def query(
        self,
        model: str,
        prompt: str,
        input_text: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        evaluate: bool = True,
        tags: Optional[Dict[str, str]] = None,
    ) -> QueryResult:
        """
        Execute an AI query with automatic tracing and evaluation.
        
        Args:
            model: Model name (e.g., 'claude-3-5-sonnet', 'gpt-4o')
            prompt: The prompt/instruction
            input_text: Optional input text to process
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            evaluate: Whether to auto-evaluate
            tags: Optional tags for MLflow
        
        Returns:
            QueryResult with response, trace, and summary
        """
        start_time = time.time()
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{model}_{int(start_time)}") as run:
            run_id = run.info.run_id
            
            # Log parameters
            mlflow.log_param("model", model)
            mlflow.log_param("temperature", temperature)
            mlflow.log_param("prompt_preview", prompt[:100])
            if tags:
                mlflow.set_tags(tags)
            
            # Build message
            if input_text:
                full_prompt = f"{prompt}\n\nInput:\n{input_text}"
            else:
                full_prompt = prompt
            
            # Initialize provider
            provider = get_provider(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Create trace
            with mlflow.start_span(name="llm_query", span_type="LLM") as span:
                span.set_attribute("model", model)
                span.set_attribute("prompt_length", len(full_prompt))
                
                # Call LLM
                messages = [Message(role=MessageRole.USER, content=full_prompt)]
                response = provider.complete(messages)
                
                # Calculate metrics
                duration = time.time() - start_time
                tokens = response.usage.get("total_tokens", 0) if response.usage else 0
                prompt_tokens = response.usage.get("prompt_tokens", 0) if response.usage else 0
                completion_tokens = response.usage.get("completion_tokens", 0) if response.usage else 0
                
                # Estimate cost (rough, model-dependent)
                cost = self._estimate_cost(model, prompt_tokens, completion_tokens)
                
                # Log metrics
                mlflow.log_metric("latency_ms", duration * 1000)
                mlflow.log_metric("total_tokens", tokens)
                mlflow.log_metric("prompt_tokens", prompt_tokens)
                mlflow.log_metric("completion_tokens", completion_tokens)
                mlflow.log_metric("cost_usd", cost)
                
                span.set_attribute("latency_ms", duration * 1000)
                span.set_attribute("total_tokens", tokens)
                span.set_attribute("cost", cost)
            
            # Generate IntelliGen-style summary
            summary = self._generate_intelligen_summary(
                model=model,
                prompt=prompt,
                response=response.content,
                duration=duration,
                tokens=tokens,
                cost=cost,
                evaluate=evaluate,
            )
            
            # Log summary
            mlflow.log_text(summary, "summary.txt")
            
            # Evaluation scores
            eval_scores = {}
            if evaluate:
                eval_scores = self._quick_evaluate(
                    prompt=full_prompt,
                    response=response.content,
                    duration=duration,
                    tokens=tokens,
                )
                
                for metric, score in eval_scores.items():
                    mlflow.log_metric(f"eval_{metric}", score)
            
            # Create result
            metrics = {
                "latency_ms": duration * 1000,
                "tokens": tokens,
                "cost_usd": cost,
                **eval_scores,
            }
            
            result = QueryResult(
                response=response.content,
                trace_id=run_id,
                model=model,
                summary=summary,
                metrics=metrics,
                run_id=run_id,
            )
            
            # Add to history for comparison
            self.query_history.append(result)
            
            return result
    
    def _estimate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Estimate cost based on model pricing."""
        # Rough pricing (as of 2024-2025, adjust as needed)
        pricing = {
            # OpenAI
            "gpt-4o": (0.005, 0.015),  # per 1K tokens (input, output)
            "gpt-4-turbo": (0.01, 0.03),
            "gpt-3.5-turbo": (0.0005, 0.0015),
            # Anthropic
            "claude-3-5-sonnet": (0.003, 0.015),
            "claude-3-opus": (0.015, 0.075),
            "claude-3-sonnet": (0.003, 0.015),
            "claude-3-haiku": (0.00025, 0.00125),
            # Gemini
            "gemini-pro": (0.00025, 0.0005),
            "gemini-1.5-pro": (0.0035, 0.0105),
        }
        
        # Find matching pricing
        model_lower = model.lower()
        for model_key, (input_price, output_price) in pricing.items():
            if model_key in model_lower:
                return (prompt_tokens / 1000 * input_price) + (completion_tokens / 1000 * output_price)
        
        # Default rough estimate
        return (prompt_tokens + completion_tokens) / 1000 * 0.01
    
    def _generate_intelligen_summary(
        self,
        model: str,
        prompt: str,
        response: str,
        duration: float,
        tokens: int,
        cost: float,
        evaluate: bool,
    ) -> str:
        """Generate IntelliGen-style 3-sentence summary."""
        # Sentence 1: What happened
        sentence1 = f"Queried {model} with a {len(prompt)}-character prompt."
        
        # Sentence 2: Performance metrics
        sentence2 = f"Response generated in {duration:.2f}s using {tokens} tokens at estimated cost ${cost:.4f}."
        
        # Sentence 3: Quality assessment
        if evaluate:
            response_len = len(response)
            if response_len < 50:
                quality = "very concise"
            elif response_len < 200:
                quality = "concise and clear"
            elif response_len < 500:
                quality = "detailed"
            else:
                quality = "comprehensive"
            
            sentence3 = f"Generated a {quality} response of {response_len} characters."
        else:
            sentence3 = f"Response contains {len(response)} characters."
        
        return f"{sentence1} {sentence2} {sentence3}"
    
    def _quick_evaluate(
        self,
        prompt: str,
        response: str,
        duration: float,
        tokens: int,
    ) -> Dict[str, float]:
        """Quick heuristic evaluation."""
        scores = {}
        
        # Helpfulness (heuristic: longer responses tend to be more detailed)
        response_len = len(response)
        if response_len > 200:
            scores["helpfulness"] = 0.9
        elif response_len > 100:
            scores["helpfulness"] = 0.8
        elif response_len > 50:
            scores["helpfulness"] = 0.7
        else:
            scores["helpfulness"] = 0.6
        
        # Conciseness (inverse relationship with length)
        if response_len < 100:
            scores["conciseness"] = 1.0
        elif response_len < 300:
            scores["conciseness"] = 0.9
        elif response_len < 500:
            scores["conciseness"] = 0.7
        else:
            scores["conciseness"] = 0.6
        
        # Efficiency (based on latency)
        if duration < 1.0:
            scores["efficiency"] = 1.0
        elif duration < 3.0:
            scores["efficiency"] = 0.9
        elif duration < 5.0:
            scores["efficiency"] = 0.7
        else:
            scores["efficiency"] = 0.6
        
        # Cost efficiency
        cost_per_char = (tokens / 1000 * 0.01) / max(response_len, 1)
        if cost_per_char < 0.0001:
            scores["cost_efficiency"] = 1.0
        elif cost_per_char < 0.0005:
            scores["cost_efficiency"] = 0.9
        else:
            scores["cost_efficiency"] = 0.7
        
        return scores
    
    def compare(
        self,
        result1: QueryResult,
        result2: QueryResult,
    ) -> Dict[str, Any]:
        """
        Compare two query results side-by-side.
        
        Args:
            result1: First query result
            result2: Second query result
        
        Returns:
            Comparison dictionary
        """
        comparison = {
            "model_comparison": {
                "model_1": result1.model,
                "model_2": result2.model,
            },
            "performance": {
                "latency_ms": {
                    "model_1": result1.metrics.get("latency_ms", 0),
                    "model_2": result2.metrics.get("latency_ms", 0),
                    "winner": result1.model if result1.metrics.get("latency_ms", 0) < result2.metrics.get("latency_ms", 0) else result2.model,
                },
                "cost_usd": {
                    "model_1": result1.metrics.get("cost_usd", 0),
                    "model_2": result2.metrics.get("cost_usd", 0),
                    "winner": result1.model if result1.metrics.get("cost_usd", 0) < result2.metrics.get("cost_usd", 0) else result2.model,
                },
                "tokens": {
                    "model_1": result1.metrics.get("tokens", 0),
                    "model_2": result2.metrics.get("tokens", 0),
                },
            },
            "quality": {},
        }
        
        # Compare quality metrics
        for metric in ["helpfulness", "conciseness", "efficiency"]:
            score1 = result1.metrics.get(metric, 0)
            score2 = result2.metrics.get(metric, 0)
            comparison["quality"][metric] = {
                "model_1": score1,
                "model_2": score2,
                "winner": result1.model if score1 > score2 else result2.model,
            }
        
        return comparison
    
    def print_comparison(
        self,
        result1: QueryResult,
        result2: QueryResult,
    ):
        """Print a formatted comparison."""
        comparison = self.compare(result1, result2)
        
        print("=" * 70)
        print("📊 Model Comparison")
        print("=" * 70)
        print(f"\n{'Metric':<20} {'Model 1':<20} {'Model 2':<20} {'Winner':<10}")
        print("-" * 70)
        
        print(f"{'Model':<20} {result1.model:<20} {result2.model:<20}")
        print()
        
        # Performance metrics
        print("Performance:")
        perf = comparison["performance"]
        for metric, data in perf.items():
            if metric == "tokens":
                print(f"  {metric:<18} {data['model_1']:<20} {data['model_2']:<20}")
            else:
                winner_mark = "🏆" if "winner" in data else ""
                print(f"  {metric:<18} {data['model_1']:<20.3f} {data['model_2']:<20.3f} {winner_mark}")
        
        # Quality metrics
        print("\nQuality:")
        for metric, data in comparison["quality"].items():
            winner = data["winner"]
            winner_mark = "🏆 " + winner
            print(f"  {metric:<18} {data['model_1']:<20.3f} {data['model_2']:<20.3f} {winner_mark}")
        
        print("=" * 70)
    
    def suggest_improvement(
        self,
        result: QueryResult,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Suggest improvements to the prompt based on the result.
        
        Args:
            result: Query result to analyze
            context: Optional context about the goal
        
        Returns:
            Dictionary with suggestions
        """
        suggestions = {
            "current_metrics": result.metrics,
            "improvements": [],
            "suggested_prompt": None,
        }
        
        # Analyze metrics and suggest improvements
        if result.metrics.get("helpfulness", 1.0) < 0.8:
            suggestions["improvements"].append(
                "Response could be more helpful. Consider adding: 'Provide detailed explanations with examples.'"
            )
        
        if result.metrics.get("conciseness", 1.0) < 0.7:
            suggestions["improvements"].append(
                "Response is verbose. Consider adding: 'Be concise and focus on key points.'"
            )
        
        if result.metrics.get("latency_ms", 0) > 3000:
            suggestions["improvements"].append(
                "High latency detected. Consider using a faster model or simplifying the prompt."
            )
        
        if result.metrics.get("cost_usd", 0) > 0.05:
            suggestions["improvements"].append(
                "High cost per query. Consider using a more cost-efficient model or shorter prompts."
            )
        
        if not suggestions["improvements"]:
            suggestions["improvements"].append(
                "Performance is good! Current prompt is effective."
            )
        
        return suggestions
    
    def print_suggestions(
        self,
        result: QueryResult,
        context: Optional[str] = None,
    ):
        """Print improvement suggestions."""
        suggestions = self.suggest_improvement(result, context)
        
        print("=" * 60)
        print("💡 Improvement Suggestions")
        print("=" * 60)
        print("\n📊 Current Performance:")
        for metric, value in suggestions["current_metrics"].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
        
        print("\n🔧 Suggestions:")
        for i, suggestion in enumerate(suggestions["improvements"], 1):
            print(f"  {i}. {suggestion}")
        
        print("=" * 60)
    
    def get_recent_queries(self, n: int = 5) -> List[QueryResult]:
        """Get the N most recent queries."""
        return self.query_history[-n:]


# Global instance for convenience
_api_instance = SimpleAPI()


def ai_query(
    model: str,
    prompt: str,
    input_text: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    evaluate: bool = True,
    print_summary: bool = False,
) -> QueryResult:
    """
    Simple AI query function - notebook-friendly interface.
    
    This is the main entry point for the simplified experience described in the flow.
    
    Args:
        model: Model name (e.g., 'claude-3-5-sonnet', 'gpt-4o')
        prompt: The prompt/instruction
        input_text: Optional input text to process
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        evaluate: Whether to auto-evaluate
        print_summary: Whether to print summary immediately
    
    Returns:
        QueryResult with response, trace, and IntelliGen-style summary
    
    Example:
        >>> result = ai_query('claude-3-5-sonnet', 'Summarize this support ticket', input_text)
        >>> print(result.response)
        >>> result.print_summary()
    """
    result = _api_instance.query(
        model=model,
        prompt=prompt,
        input_text=input_text,
        temperature=temperature,
        max_tokens=max_tokens,
        evaluate=evaluate,
    )
    
    if print_summary:
        result.print_summary()
    
    return result


def compare_models(result1: QueryResult, result2: QueryResult):
    """Compare two query results."""
    _api_instance.print_comparison(result1, result2)


def suggest_improvement(result: QueryResult, context: Optional[str] = None):
    """Suggest improvements for a query."""
    _api_instance.print_suggestions(result, context)

