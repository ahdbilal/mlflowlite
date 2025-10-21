"""DSPy-based prompt optimization."""

from typing import List, Dict, Any, Optional
import json


class DSPyOptimizer:
    """Optimizes agent prompts using DSPy-inspired techniques."""
    
    def __init__(
        self,
        agent_name: str,
        llm_provider: Optional[Any] = None,
    ):
        """
        Initialize DSPy optimizer.
        
        Args:
            agent_name: Name of the agent
            llm_provider: LLM provider for optimization
        """
        self.agent_name = agent_name
        self.llm_provider = llm_provider
    
    def optimize_from_traces(
        self,
        traces: List[Any],  # List of AgentTrace
        evaluation_results: List[Any],  # List of EvaluationResult
        current_prompt: str,
    ) -> Dict[str, Any]:
        """
        Optimize prompt based on execution traces and evaluations.
        
        Args:
            traces: List of agent execution traces
            evaluation_results: Corresponding evaluation results
            current_prompt: Current system prompt
        
        Returns:
            Dictionary with optimized prompt and metadata
        """
        if len(traces) < 3:
            return {
                "status": "insufficient_data",
                "message": "Need at least 3 traces to optimize",
                "optimized_prompt": current_prompt,
            }
        
        # Analyze successful vs unsuccessful patterns
        successful_traces = []
        unsuccessful_traces = []
        
        for trace, eval_result in zip(traces, evaluation_results):
            avg_score = sum(eval_result.scores.values()) / len(eval_result.scores)
            if avg_score > 0.75:
                successful_traces.append(trace)
            else:
                unsuccessful_traces.append(trace)
        
        # Extract patterns
        patterns = self._extract_patterns(successful_traces, unsuccessful_traces)
        
        # Generate optimized prompt
        if self.llm_provider:
            optimized_prompt = self._llm_optimize_prompt(
                current_prompt,
                patterns,
                traces,
                evaluation_results,
            )
        else:
            optimized_prompt = self._heuristic_optimize_prompt(
                current_prompt,
                patterns,
            )
        
        # Extract few-shot examples from successful traces
        examples = self._extract_examples(successful_traces)
        
        return {
            "status": "success",
            "optimized_prompt": optimized_prompt,
            "examples": examples,
            "patterns": patterns,
            "metadata": {
                "num_traces_analyzed": len(traces),
                "num_successful": len(successful_traces),
                "num_unsuccessful": len(unsuccessful_traces),
            }
        }
    
    def _extract_patterns(
        self,
        successful_traces: List[Any],
        unsuccessful_traces: List[Any],
    ) -> Dict[str, Any]:
        """Extract patterns from successful vs unsuccessful traces."""
        patterns = {
            "successful_patterns": [],
            "unsuccessful_patterns": [],
        }
        
        # Analyze tool usage patterns
        if successful_traces:
            successful_tools = []
            for trace in successful_traces:
                tools_used = [
                    s.metadata.get("tool_name")
                    for s in trace.steps
                    if s.step_type == "tool_call"
                ]
                successful_tools.extend(tools_used)
            
            if successful_tools:
                patterns["successful_patterns"].append(
                    f"Effective tool usage: {', '.join(set(successful_tools))}"
                )
        
        if unsuccessful_traces:
            unsuccessful_tools = []
            for trace in unsuccessful_traces:
                tools_used = [
                    s.metadata.get("tool_name")
                    for s in trace.steps
                    if s.step_type == "tool_call"
                ]
                unsuccessful_tools.extend(tools_used)
            
            # Check for excessive tool calls
            if len(unsuccessful_tools) > len(successful_tools) / len(successful_traces) * 2:
                patterns["unsuccessful_patterns"].append(
                    "Excessive tool usage detected"
                )
        
        # Analyze reasoning patterns
        if successful_traces:
            avg_reasoning_steps = sum(
                len([s for s in t.steps if s.step_type == "reasoning"])
                for t in successful_traces
            ) / len(successful_traces)
            
            patterns["successful_patterns"].append(
                f"Average reasoning steps: {avg_reasoning_steps:.1f}"
            )
        
        return patterns
    
    def _heuristic_optimize_prompt(
        self,
        current_prompt: str,
        patterns: Dict[str, Any],
    ) -> str:
        """Heuristically optimize prompt based on patterns."""
        optimized = current_prompt
        
        # Add guidance based on patterns
        improvements = []
        
        if patterns["unsuccessful_patterns"]:
            if "Excessive tool usage" in patterns["unsuccessful_patterns"]:
                improvements.append(
                    "\nIMPORTANT: Use tools judiciously. Only call a tool when necessary."
                )
        
        if patterns["successful_patterns"]:
            for pattern in patterns["successful_patterns"]:
                if "Effective tool usage" in pattern:
                    # Extract tools and add emphasis
                    improvements.append(
                        f"\nRECOMMENDED: {pattern}"
                    )
        
        if improvements:
            optimized = current_prompt + "\n\n" + "\n".join(improvements)
        
        return optimized
    
    def _llm_optimize_prompt(
        self,
        current_prompt: str,
        patterns: Dict[str, Any],
        traces: List[Any],
        evaluation_results: List[Any],
    ) -> str:
        """Use LLM to optimize prompt based on traces and patterns."""
        try:
            from mlflowlite.llm.base import Message, MessageRole
            
            # Prepare optimization prompt
            trace_summaries = []
            for trace, eval_result in zip(traces[:5], evaluation_results[:5]):  # Limit to avoid token overflow
                avg_score = sum(eval_result.scores.values()) / len(eval_result.scores)
                trace_summaries.append({
                    "input": trace.input_query[:100],
                    "output": trace.output[:100],
                    "score": round(avg_score, 2),
                    "duration": round(trace.duration, 2),
                    "steps": len(trace.steps),
                })
            
            optimization_prompt = f"""You are a prompt optimization expert. Improve the following agent system prompt based on execution traces and patterns.

CURRENT PROMPT:
{current_prompt}

EXECUTION PATTERNS:
Successful patterns: {patterns.get('successful_patterns', [])}
Unsuccessful patterns: {patterns.get('unsuccessful_patterns', [])}

SAMPLE TRACES:
{json.dumps(trace_summaries, indent=2)}

Generate an improved system prompt that:
1. Maintains the core functionality
2. Addresses identified weaknesses
3. Reinforces successful patterns
4. Is clear and concise

Return ONLY the improved prompt text, nothing else."""
            
            messages = [Message(role=MessageRole.USER, content=optimization_prompt)]
            response = self.llm_provider.complete(messages, temperature=0.7)
            
            optimized = response.content.strip()
            
            # Validate it's not empty or too short
            if len(optimized) < 50:
                return current_prompt
            
            return optimized
        
        except Exception as e:
            print(f"Warning: LLM optimization failed: {e}")
            return self._heuristic_optimize_prompt(current_prompt, patterns)
    
    def _extract_examples(
        self,
        successful_traces: List[Any],
        max_examples: int = 3,
    ) -> List[Dict[str, str]]:
        """Extract few-shot examples from successful traces."""
        examples = []
        
        for trace in successful_traces[:max_examples]:
            # Create a concise example showing input, reasoning, and output
            reasoning_steps = [
                s.content for s in trace.steps
                if s.step_type == "reasoning"
            ]
            
            reasoning_summary = reasoning_steps[0][:200] if reasoning_steps else "Direct response"
            
            examples.append({
                "input": trace.input_query,
                "reasoning": reasoning_summary,
                "output": trace.output[:200],
            })
        
        return examples
    
    def auto_optimize(
        self,
        agent: Any,  # Agent instance
        num_runs: int = 10,
        test_queries: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Automatically run test queries and optimize based on results.
        
        Args:
            agent: Agent instance to optimize
            num_runs: Number of test runs per iteration
            test_queries: List of test queries (generated if not provided)
        
        Returns:
            Optimization results
        """
        if not test_queries:
            test_queries = self._generate_test_queries(agent)
        
        # Run agent with test queries
        traces = []
        evaluations = []
        
        for query in test_queries[:num_runs]:
            try:
                result = agent.run(query, evaluate=True)
                traces.append(result.trace)
                if hasattr(result, 'evaluation'):
                    evaluations.append(result.evaluation)
            except Exception as e:
                print(f"Warning: Test run failed: {e}")
        
        if not traces:
            return {
                "status": "failed",
                "message": "No successful test runs",
            }
        
        # Optimize based on collected data
        current_prompt = agent.prompt_registry.get_latest().system_prompt
        
        optimization_result = self.optimize_from_traces(
            traces,
            evaluations,
            current_prompt,
        )
        
        # Add new version to registry if optimization succeeded
        if optimization_result["status"] == "success":
            agent.prompt_registry.add_version(
                system_prompt=optimization_result["optimized_prompt"],
                examples=optimization_result["examples"],
                metadata={
                    "optimization": "dspy_auto",
                    "num_traces": len(traces),
                }
            )
        
        return optimization_result
    
    def _generate_test_queries(self, agent: Any) -> List[str]:
        """Generate test queries for the agent."""
        # Simple default test queries
        # In production, could use LLM to generate domain-specific queries
        return [
            "What is the current date?",
            "Calculate 25 * 17",
            "Summarize the benefits of machine learning",
            "Search for recent news about AI",
            "What are the key principles of software engineering?",
        ]

