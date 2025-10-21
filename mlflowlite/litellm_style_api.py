"""
LiteLLM-style API for MLflow Agents

Designed to feel like LiteLLM but with automatic MLflow tracing and evaluation.

Usage:
    import mlflowlite as mla
    
    # Simple completion (like litellm.completion)
    response = mla.completion(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "Hello"}]
    )
    
    # Or use our simplified query interface
    response = mla.query(
        model="claude-3-5-sonnet",
        prompt="Summarize this",
        input="Your text here"
    )
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import time
import mlflow
from mlflowlite.llm.providers import get_provider
from mlflowlite.llm.base import Message, MessageRole, LLMResponse


@dataclass
class Response:
    """
    Response object (similar to LiteLLM's response).
    
    Attributes:
        content: The response text
        model: Model used
        usage: Token usage statistics
        latency: Response time in seconds
        cost: Estimated cost in USD
        trace_id: MLflow trace ID
        metadata: Additional metadata
    """
    content: str
    model: str
    usage: Dict[str, int]
    latency: float
    cost: float
    trace_id: str
    metadata: Dict[str, Any]
    
    # Evaluation scores (auto-calculated)
    scores: Optional[Dict[str, float]] = None
    
    def __str__(self) -> str:
        return self.content
    
    def __repr__(self) -> str:
        return f"Response(model={self.model!r}, tokens={self.usage.get('total_tokens', 0)}, cost=${self.cost:.4f})"


# Global tracking state
_mlflow_enabled = True
_auto_evaluate = True
_provider_for_suggestions = None


def set_mlflow_tracking(enabled: bool = True):
    """Enable or disable MLflow tracking."""
    global _mlflow_enabled
    _mlflow_enabled = enabled


def set_auto_evaluate(enabled: bool = True):
    """Enable or disable automatic evaluation."""
    global _auto_evaluate
    _auto_evaluate = enabled


def set_suggestion_provider(model: str = "claude-3-5-sonnet"):
    """
    Enable LLM-powered improvement suggestions using DSPy-style analysis.
    
    By default, suggestions use fast heuristic rules.
    Call this to enable smarter LLM-powered suggestions.
    
    Args:
        model: Model to use for generating suggestions
    
    Example:
        >>> import mlflowlite as mla
        >>> mla.set_suggestion_provider("claude-3-5-sonnet")
        >>> # Now suggest_improvement() will use LLM analysis
    """
    global _provider_for_suggestions
    _provider_for_suggestions = get_provider(model=model)


def completion(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    stream: bool = False,
    api_key: Optional[str] = None,
    **kwargs
) -> Response:
    """
    Create a completion (LiteLLM-style interface).
    
    This is the main function - works just like litellm.completion() but with
    automatic MLflow tracing and evaluation.
    
    Args:
        model: Model name (e.g., "claude-3-5-sonnet", "gpt-4o")
        messages: List of message dicts with "role" and "content"
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens in response
        tools: Tool definitions (OpenAI format)
        stream: Whether to stream the response (not yet implemented)
        api_key: API key for the provider
        **kwargs: Additional provider-specific parameters
    
    Returns:
        Response object with content, usage, cost, and trace_id
    
    Example:
        >>> import mlflowlite as mla
        >>> response = mla.completion(
        ...     model="claude-3-5-sonnet",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>> print(response.content)
        >>> print(f"Cost: ${response.cost:.4f}")
    """
    start_time = time.time()
    
    # Start MLflow tracking if enabled
    trace_id = None
    run_context = None
    
    if _mlflow_enabled:
        try:
            # Set experiment
            mlflow.set_experiment("mlflowlite")
        except Exception:
            mlflow.create_experiment("mlflowlite")
            mlflow.set_experiment("mlflowlite")
        
        # Start run
        run_context = mlflow.start_run(run_name=f"{model}_{int(start_time)}")
        trace_id = run_context.__enter__().info.run_id
        
        # Log parameters
        mlflow.log_param("model", model)
        mlflow.log_param("temperature", temperature)
        mlflow.log_param("message_count", len(messages))
    
    try:
        # Get provider and call LLM
        provider = get_provider(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            **kwargs
        )
        
        # Convert messages to our format
        message_objects = [
            Message(
                role=MessageRole(msg["role"]),
                content=msg["content"]
            )
            for msg in messages
        ]
        
        # Make the call
        if _mlflow_enabled:
            with mlflow.start_span(name="llm_completion", span_type="LLM") as span:
                span.set_attribute("model", model)
                llm_response = provider.complete(message_objects, tools=tools)
        else:
            llm_response = provider.complete(message_objects, tools=tools)
        
        # Calculate metrics
        latency = time.time() - start_time
        usage = llm_response.usage or {}
        total_tokens = usage.get("total_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        # Estimate cost
        cost = _estimate_cost(model, prompt_tokens, completion_tokens)
        
        # Auto-evaluate if enabled
        scores = None
        if _auto_evaluate:
            scores = _quick_evaluate(
                response_text=llm_response.content,
                latency=latency,
                tokens=total_tokens,
            )
        
        # Log to MLflow
        if _mlflow_enabled:
            mlflow.log_metric("latency_seconds", latency)
            mlflow.log_metric("total_tokens", total_tokens)
            mlflow.log_metric("cost_usd", cost)
            
            if scores:
                for metric, score in scores.items():
                    mlflow.log_metric(f"score_{metric}", score)
            
            # Log response preview
            mlflow.log_param("response_preview", llm_response.content[:200])
        
        # Create response object
        response = Response(
            content=llm_response.content,
            model=model,
            usage=usage,
            latency=latency,
            cost=cost,
            trace_id=trace_id or "no_trace",
            metadata={
                "provider": provider.provider_name,
                "finish_reason": llm_response.finish_reason,
            },
            scores=scores,
        )
        
        return response
        
    finally:
        if _mlflow_enabled and run_context:
            run_context.__exit__(None, None, None)


def query(
    model: str,
    prompt: str,
    input: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> Response:
    """
    Simplified query interface (even easier than completion).
    
    Args:
        model: Model name
        prompt: The prompt/instruction
        input: Optional input text to process
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        api_key: API key
        **kwargs: Additional parameters
    
    Returns:
        Response object
    
    Example:
        >>> import mlflowlite as mla
        >>> response = mla.query(
        ...     model="claude-3-5-sonnet",
        ...     prompt="Summarize this",
        ...     input="Long text here..."
        ... )
        >>> print(response)
    """
    # Build messages
    if input:
        content = f"{prompt}\n\nInput:\n{input}"
    else:
        content = prompt
    
    messages = [{"role": "user", "content": content}]
    
    return completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        **kwargs
    )


def suggest_improvement(
    response: Response,
    context: Optional[str] = None,
    use_llm: bool = None,
) -> Dict[str, Any]:
    """
    Get improvement suggestions for a response.
    
    By default uses fast heuristic rules.
    For smarter suggestions, either:
      1. Call set_suggestion_provider() first, OR
      2. Pass use_llm=True
    
    Args:
        response: Response object to analyze
        context: Optional context about the goal
        use_llm: Force LLM-powered suggestions (slower, smarter)
    
    Returns:
        Dictionary with suggestions
    
    Example:
        >>> # Fast heuristic suggestions (default)
        >>> suggestions = mla.suggest_improvement(response)
        
        >>> # Smart LLM-powered suggestions
        >>> mla.set_suggestion_provider("claude-3-5-sonnet")
        >>> suggestions = mla.suggest_improvement(response)
        
        >>> # Or one-time LLM suggestion
        >>> suggestions = mla.suggest_improvement(response, use_llm=True)
    """
    global _provider_for_suggestions
    
    # Determine if we should use LLM
    should_use_llm = use_llm or (_provider_for_suggestions is not None)
    
    if should_use_llm:
        return _llm_suggest_improvement(response, context)
    else:
        return _heuristic_suggest_improvement(response)


def _heuristic_suggest_improvement(response: Response) -> Dict[str, Any]:
    """Fast heuristic-based suggestions."""
    suggestions = {
        "method": "heuristic",
        "current_metrics": {
            "latency_ms": response.latency * 1000,
            "tokens": response.usage.get("total_tokens", 0),
            "cost_usd": response.cost,
            **(response.scores or {})
        },
        "improvements": [],
    }
    
    scores = response.scores or {}
    
    # Check metrics against thresholds
    if scores.get("helpfulness", 1.0) < 0.8:
        suggestions["improvements"].append(
            "Response could be more helpful. Consider adding: 'Provide detailed explanations with examples.'"
        )
    
    if scores.get("conciseness", 1.0) < 0.7:
        suggestions["improvements"].append(
            "Response is verbose. Consider adding: 'Be concise and focus on key points.'"
        )
    
    if response.latency > 3.0:
        suggestions["improvements"].append(
            "High latency detected. Consider using a faster model or simplifying the prompt."
        )
    
    if response.cost > 0.05:
        suggestions["improvements"].append(
            "High cost per query. Consider using a more cost-efficient model or shorter prompts."
        )
    
    if not suggestions["improvements"]:
        suggestions["improvements"].append(
            "Performance is good! Current prompt is effective."
        )
    
    return suggestions


def _llm_suggest_improvement(response: Response, context: Optional[str]) -> Dict[str, Any]:
    """LLM-powered suggestions using DSPy-style analysis."""
    global _provider_for_suggestions
    
    if _provider_for_suggestions is None:
        _provider_for_suggestions = get_provider(model="claude-3-5-sonnet")
    
    # Build analysis prompt
    analysis_prompt = f"""You are an expert prompt engineer. Analyze this LLM response and provide actionable improvement suggestions.

Response Analysis:
- Content: {response.content[:300]}{'...' if len(response.content) > 300 else ''}
- Model: {response.model}
- Latency: {response.latency:.2f}s
- Tokens: {response.usage.get('total_tokens', 0)}
- Cost: ${response.cost:.4f}
- Scores: {response.scores}

{f'Context: {context}' if context else ''}

Provide 2-3 specific, actionable suggestions to improve:
1. Response quality (helpfulness, accuracy)
2. Efficiency (speed, cost)
3. Prompt design

Be specific and practical. Format as a numbered list."""
    
    try:
        messages = [Message(role=MessageRole.USER, content=analysis_prompt)]
        llm_response = _provider_for_suggestions.complete(messages, temperature=0.7)
        
        return {
            "method": "llm",
            "current_metrics": {
                "latency_ms": response.latency * 1000,
                "tokens": response.usage.get("total_tokens", 0),
                "cost_usd": response.cost,
                **(response.scores or {})
            },
            "improvements": [
                line.strip() 
                for line in llm_response.content.split("\n") 
                if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith("-"))
            ],
            "analysis_full": llm_response.content,
        }
    except Exception as e:
        print(f"Warning: LLM suggestions failed, falling back to heuristics: {e}")
        return _heuristic_suggest_improvement(response)


def print_suggestions(
    response: Response,
    context: Optional[str] = None,
    use_llm: bool = None,
):
    """
    Pretty print improvement suggestions.
    
    Args:
        response: Response object
        context: Optional context
        use_llm: Whether to use LLM-powered suggestions
    """
    suggestions = suggest_improvement(response, context, use_llm)
    
    print("=" * 60)
    print(f"💡 Improvement Suggestions ({suggestions['method'].upper()})")
    print("=" * 60)
    print("\n📊 Current Performance:")
    for metric, value in suggestions["current_metrics"].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")
    
    print("\n🔧 Suggestions:")
    for i, suggestion in enumerate(suggestions["improvements"], 1):
        # Clean up the suggestion text
        clean = suggestion.lstrip("0123456789.-) ")
        print(f"  {i}. {clean}")
    
    if suggestions["method"] == "llm" and "analysis_full" in suggestions:
        print(f"\n📝 Powered by LLM analysis")
    
    print("=" * 60)


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost based on model pricing."""
    pricing = {
        # OpenAI
        "gpt-4o": (0.005, 0.015),
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
    
    model_lower = model.lower()
    for model_key, (input_price, output_price) in pricing.items():
        if model_key in model_lower:
            return (prompt_tokens / 1000 * input_price) + (completion_tokens / 1000 * output_price)
    
    return (prompt_tokens + completion_tokens) / 1000 * 0.01


def _quick_evaluate(response_text: str, latency: float, tokens: int) -> Dict[str, float]:
    """Quick heuristic evaluation."""
    scores = {}
    
    response_len = len(response_text)
    
    # Helpfulness (based on detail)
    if response_len > 200:
        scores["helpfulness"] = 0.9
    elif response_len > 100:
        scores["helpfulness"] = 0.8
    elif response_len > 50:
        scores["helpfulness"] = 0.7
    else:
        scores["helpfulness"] = 0.6
    
    # Conciseness (inverse of length)
    if response_len < 100:
        scores["conciseness"] = 1.0
    elif response_len < 300:
        scores["conciseness"] = 0.9
    elif response_len < 500:
        scores["conciseness"] = 0.7
    else:
        scores["conciseness"] = 0.6
    
    # Speed (based on latency)
    if latency < 1.0:
        scores["speed"] = 1.0
    elif latency < 3.0:
        scores["speed"] = 0.9
    elif latency < 5.0:
        scores["speed"] = 0.7
    else:
        scores["speed"] = 0.6
    
    return scores


# Batch processing (like litellm.batch_completion)
def batch_completion(
    model: str,
    messages_list: List[List[Dict[str, str]]],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs
) -> List[Response]:
    """
    Process multiple completions in batch.
    
    Args:
        model: Model name
        messages_list: List of message lists
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        **kwargs: Additional parameters
    
    Returns:
        List of Response objects
    """
    responses = []
    for messages in messages_list:
        response = completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        responses.append(response)
    return responses


# Model listing (like litellm.model_list)
def get_available_models() -> Dict[str, List[str]]:
    """
    Get list of available models by provider.
    
    Returns:
        Dictionary mapping provider names to model lists
    """
    return {
        "openai": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-5-sonnet", "claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
        "google": ["gemini-pro", "gemini-1.5-pro"],
        "mistral": ["mistral-large", "mistral-medium", "mistral-small"],
    }


# Helper to print response nicely
def print_response(response: Response, show_metadata: bool = False):
    """
    Pretty print a response.
    
    Args:
        response: Response object
        show_metadata: Whether to show metadata
    """
    print("=" * 60)
    print(f"Model: {response.model}")
    print(f"Latency: {response.latency:.2f}s | Tokens: {response.usage.get('total_tokens', 0)} | Cost: ${response.cost:.4f}")
    if response.scores:
        scores_str = " | ".join([f"{k}: {v:.2f}" for k, v in response.scores.items()])
        print(f"Scores: {scores_str}")
    print("=" * 60)
    print(response.content)
    print("=" * 60)
    
    if show_metadata:
        print(f"\nMetadata: {response.metadata}")
        print(f"Trace ID: {response.trace_id}")
