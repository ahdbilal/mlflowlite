"""Evaluation and improvement system."""

# Legacy evaluator (trace-based)
from mlflowlite.evaluation.evaluator import AgentEvaluator, EvaluationResult

# MLflow GenAI-style evaluation API
from mlflowlite.evaluation.evaluate import (
    evaluate,
    evaluate_with_traces,
    EvaluationResults,
    EvaluationRow,
)

# Scorers
from mlflowlite.evaluation.scorers import (
    Scorer,
    ScorerResult,
    Correctness,
    Guidelines,
    Relevance,
    Conciseness,
    Faithfulness,
    scorer,
)

__all__ = [
    # Legacy evaluator
    "AgentEvaluator",
    "EvaluationResult",
    # MLflow GenAI-style API
    "evaluate",
    "evaluate_with_traces",
    "EvaluationResults",
    "EvaluationRow",
    # Scorers
    "Scorer",
    "ScorerResult",
    "Correctness",
    "Guidelines",
    "Relevance",
    "Conciseness",
    "Faithfulness",
    "scorer",
]

