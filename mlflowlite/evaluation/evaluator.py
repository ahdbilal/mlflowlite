"""Agent evaluation system."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class EvaluationResult:
    """Result of evaluating an agent execution."""
    trace_id: str
    scores: Dict[str, float]
    feedback: Dict[str, str]
    suggestions: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "scores": self.scores,
            "feedback": self.feedback,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp,
        }


class AgentEvaluator:
    """Evaluates agent performance and suggests improvements."""
    
    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        metrics: Optional[List[str]] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            llm_provider: LLM provider for LLM-based evaluation
            metrics: List of metrics to evaluate (accuracy, relevance, efficiency, cost)
        """
        self.llm_provider = llm_provider
        self.metrics = metrics or ["accuracy", "relevance", "efficiency", "cost"]
    
    def evaluate(
        self,
        trace: Any,  # AgentTrace
        ground_truth: Optional[str] = None,
        custom_metrics: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """
        Evaluate an agent execution trace.
        
        Args:
            trace: Agent execution trace
            ground_truth: Expected output (if available)
            custom_metrics: Custom evaluation metrics
        
        Returns:
            Evaluation result with scores and suggestions
        """
        scores = {}
        feedback = {}
        suggestions = []
        
        # 1. Efficiency metrics (always available)
        if "efficiency" in self.metrics:
            efficiency_score = self._evaluate_efficiency(trace)
            scores["efficiency"] = efficiency_score
            
            if efficiency_score < 0.7:
                suggestions.append("Consider optimizing tool usage to reduce latency")
                feedback["efficiency"] = "Agent took longer than expected"
            else:
                feedback["efficiency"] = "Good execution speed"
        
        # 2. Cost metrics
        if "cost" in self.metrics:
            cost_score = self._evaluate_cost(trace)
            scores["cost"] = cost_score
            
            if cost_score < 0.7:
                suggestions.append("High token usage detected. Consider more concise prompts")
                feedback["cost"] = "Token usage is higher than optimal"
            else:
                feedback["cost"] = "Token usage is within acceptable range"
        
        # 3. Tool usage metrics
        tool_score, tool_feedback = self._evaluate_tool_usage(trace)
        scores["tool_usage"] = tool_score
        feedback["tool_usage"] = tool_feedback
        
        # 4. Reasoning quality (heuristic-based)
        reasoning_score = self._evaluate_reasoning_quality(trace)
        scores["reasoning_quality"] = reasoning_score
        
        if reasoning_score < 0.6:
            suggestions.append("Add more structured reasoning steps to improve clarity")
        
        # 5. LLM-based evaluation (if available and requested)
        if self.llm_provider and "accuracy" in self.metrics:
            if ground_truth:
                accuracy_score = self._llm_evaluate_accuracy(trace, ground_truth)
                scores["accuracy"] = accuracy_score
                
                if accuracy_score < 0.7:
                    suggestions.append("Output doesn't fully match expected result. Consider adding examples")
                    feedback["accuracy"] = "Output accuracy can be improved"
                else:
                    feedback["accuracy"] = "Output is accurate"
        
        # 6. Relevance evaluation
        if "relevance" in self.metrics:
            relevance_score = self._evaluate_relevance(trace)
            scores["relevance"] = relevance_score
            
            if relevance_score < 0.7:
                suggestions.append("Response included irrelevant information. Focus on the user's query")
        
        # 7. Apply custom metrics
        if custom_metrics:
            for metric_name, metric_func in custom_metrics.items():
                try:
                    scores[metric_name] = metric_func(trace)
                except Exception as e:
                    feedback[metric_name] = f"Error evaluating: {str(e)}"
        
        # Generate overall suggestions
        avg_score = sum(scores.values()) / len(scores) if scores else 0.5
        
        if avg_score < 0.6:
            suggestions.append("Overall performance is below target. Consider prompt optimization")
        elif avg_score > 0.85:
            suggestions.append("Excellent performance! Current prompt is effective")
        
        return EvaluationResult(
            trace_id=trace.trace_id,
            scores=scores,
            feedback=feedback,
            suggestions=suggestions,
        )
    
    def _evaluate_efficiency(self, trace: Any) -> float:
        """Evaluate execution efficiency based on duration and steps."""
        # Heuristic: good if under 5 seconds and under 10 steps
        duration_score = min(5.0 / max(trace.duration, 0.1), 1.0)
        steps_score = min(10 / max(len(trace.steps), 1), 1.0)
        
        return (duration_score * 0.6 + steps_score * 0.4)
    
    def _evaluate_cost(self, trace: Any) -> float:
        """Evaluate cost efficiency based on tokens used."""
        # Heuristic: penalize high token usage
        # Assume good range is under 2000 tokens
        if trace.total_tokens == 0:
            return 1.0
        
        optimal_tokens = 2000
        if trace.total_tokens <= optimal_tokens:
            return 1.0
        else:
            # Score decreases as tokens increase
            return max(optimal_tokens / trace.total_tokens, 0.3)
    
    def _evaluate_tool_usage(self, trace: Any) -> tuple:
        """Evaluate tool usage effectiveness."""
        tool_calls = [s for s in trace.steps if s.step_type == "tool_call"]
        tool_results = [s for s in trace.steps if s.step_type == "tool_result"]
        
        if not tool_calls:
            return 1.0, "No tools used"
        
        # Check if all tool calls got results
        if len(tool_calls) != len(tool_results):
            return 0.5, "Some tool calls failed"
        
        # Check for redundant tool calls (same tool called multiple times)
        tool_names = [t.metadata.get("tool_name") for t in tool_calls]
        unique_tools = len(set(tool_names))
        redundancy_score = unique_tools / len(tool_names) if tool_names else 1.0
        
        if redundancy_score < 0.7:
            return redundancy_score, "Redundant tool calls detected"
        
        return 1.0, "Tool usage is effective"
    
    def _evaluate_reasoning_quality(self, trace: Any) -> float:
        """Evaluate quality of reasoning steps."""
        reasoning_steps = [s for s in trace.steps if s.step_type == "reasoning"]
        
        if not reasoning_steps:
            return 0.5  # No explicit reasoning
        
        # Heuristic: check for structured thinking
        avg_length = sum(len(s.content) for s in reasoning_steps) / len(reasoning_steps)
        
        # Good reasoning should be detailed but not verbose
        if 50 < avg_length < 500:
            return 1.0
        elif avg_length < 50:
            return 0.6  # Too brief
        else:
            return 0.7  # Too verbose
    
    def _evaluate_relevance(self, trace: Any) -> float:
        """Evaluate relevance of output to input."""
        # Simple heuristic: check if key terms from query appear in output
        query_words = set(trace.input_query.lower().split())
        output_words = set(trace.output.lower().split())
        
        # Remove common words
        common_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for"}
        query_words = query_words - common_words
        output_words = output_words - common_words
        
        if not query_words:
            return 1.0
        
        overlap = len(query_words & output_words) / len(query_words)
        return min(overlap * 1.2, 1.0)  # Allow some bonus
    
    def _llm_evaluate_accuracy(self, trace: Any, ground_truth: str) -> float:
        """Use LLM to evaluate accuracy against ground truth."""
        if not self.llm_provider:
            return 0.5
        
        try:
            from mlflowlite.llm.base import Message, MessageRole
            
            eval_prompt = f"""Evaluate the accuracy of this agent's output compared to the expected result.

Input Query: {trace.input_query}

Expected Output: {ground_truth}

Actual Output: {trace.output}

Rate the accuracy on a scale of 0.0 to 1.0, where:
- 1.0 = Perfect match or equivalent answer
- 0.7-0.9 = Mostly correct with minor issues
- 0.4-0.6 = Partially correct
- 0.0-0.3 = Incorrect or irrelevant

Respond with ONLY a number between 0.0 and 1.0."""
            
            messages = [Message(role=MessageRole.USER, content=eval_prompt)]
            response = self.llm_provider.complete(messages)
            
            # Parse score from response
            try:
                score = float(response.content.strip())
                return max(0.0, min(1.0, score))
            except ValueError:
                return 0.5
        
        except Exception:
            return 0.5
    
    def generate_improvement_suggestions(
        self,
        evaluation_results: List[EvaluationResult],
        min_samples: int = 3,
    ) -> Dict[str, Any]:
        """
        Analyze multiple evaluation results to generate improvement suggestions.
        
        Args:
            evaluation_results: List of evaluation results
            min_samples: Minimum number of samples to analyze
        
        Returns:
            Dictionary with improvement suggestions
        """
        if len(evaluation_results) < min_samples:
            return {
                "status": "insufficient_data",
                "message": f"Need at least {min_samples} evaluations to generate suggestions",
            }
        
        # Aggregate scores
        all_scores: Dict[str, List[float]] = {}
        all_suggestions: List[str] = []
        
        for result in evaluation_results:
            for metric, score in result.scores.items():
                if metric not in all_scores:
                    all_scores[metric] = []
                all_scores[metric].append(score)
            
            all_suggestions.extend(result.suggestions)
        
        # Calculate averages
        avg_scores = {
            metric: sum(scores) / len(scores)
            for metric, scores in all_scores.items()
        }
        
        # Find most common suggestions
        suggestion_counts = {}
        for suggestion in all_suggestions:
            suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1
        
        top_suggestions = sorted(
            suggestion_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Identify areas for improvement
        weak_areas = [
            metric for metric, score in avg_scores.items()
            if score < 0.7
        ]
        
        return {
            "status": "success",
            "sample_count": len(evaluation_results),
            "average_scores": avg_scores,
            "weak_areas": weak_areas,
            "top_suggestions": [s[0] for s in top_suggestions],
            "recommendation": self._generate_recommendation(avg_scores, weak_areas),
        }
    
    def _generate_recommendation(
        self,
        avg_scores: Dict[str, float],
        weak_areas: List[str],
    ) -> str:
        """Generate overall recommendation based on scores."""
        overall_avg = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0
        
        if overall_avg > 0.85:
            return "Agent is performing excellently. Consider this prompt as a baseline."
        elif overall_avg > 0.7:
            return "Agent is performing well. Minor optimizations recommended."
        elif weak_areas:
            return f"Focus on improving: {', '.join(weak_areas)}. Consider prompt optimization."
        else:
            return "Agent needs significant improvement. Review prompt and examples."

