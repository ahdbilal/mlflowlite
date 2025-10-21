"""MLflow Agents - Local-first, multi-LLM, self-improving agents."""

from mlflowlite.agent import Agent
from mlflowlite.tools.base import Tool
from mlflowlite.llm.base import LLMProvider

# LiteLLM-style API (recommended - most similar to litellm)
from mlflowlite.litellm_style_api import (
    completion,
    query,
    batch_completion,
    Response,
    set_mlflow_tracking,
    set_auto_evaluate,
    set_suggestion_provider,
    suggest_improvement,
    print_suggestions,
    get_available_models,
    print_response,
)

# Simple notebook-style API (alternative - legacy, for backwards compatibility)
from mlflowlite.simple_api import (
    ai_query,
    compare_models,
    QueryResult,
)

__version__ = "0.1.0"
__all__ = [
    # Core classes
    "Agent",
    "Tool",
    "LLMProvider",
    # LiteLLM-style (recommended)
    "completion",
    "query",
    "batch_completion",
    "Response",
    "set_mlflow_tracking",
    "set_auto_evaluate",
    "set_suggestion_provider",
    "suggest_improvement",
    "print_suggestions",
    "get_available_models",
    "print_response",
    # Simple API (alternative - legacy, for backwards compatibility)
    "ai_query",
    "compare_models",
    "QueryResult",
]

