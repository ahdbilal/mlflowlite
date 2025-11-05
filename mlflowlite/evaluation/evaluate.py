"""MLflow-style GenAI evaluation API."""

from typing import Any, Callable, Dict, List, Optional, Union
import pandas as pd
from dataclasses import dataclass, asdict
import time
import mlflow
from mlflow.entities import Metric, Param

from mlflowlite.evaluation.scorers import Scorer, ScorerResult


@dataclass
class EvaluationRow:
    """Single row result from evaluation."""
    inputs: Dict[str, Any]
    outputs: str
    expectations: Optional[Dict[str, Any]]
    scores: Dict[str, Union[float, int, bool]]
    rationales: Dict[str, str]
    latency_ms: float


class EvaluationResults:
    """Results from an evaluation run."""
    
    def __init__(
        self,
        rows: List[EvaluationRow],
        aggregate_scores: Dict[str, float],
        run_id: Optional[str] = None
    ):
        """
        Initialize evaluation results.
        
        Args:
            rows: Individual evaluation results
            aggregate_scores: Aggregated scores across all rows
            run_id: MLflow run ID if logged
        """
        self.rows = rows
        self.aggregate_scores = aggregate_scores
        self.run_id = run_id
        self._df = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        if self._df is None:
            data = []
            for row in self.rows:
                row_data = {
                    **{f"input_{k}": v for k, v in row.inputs.items()},
                    "output": row.outputs,
                    **{f"expected_{k}": v for k, v in (row.expectations or {}).items()},
                    **{f"score_{k}": v for k, v in row.scores.items()},
                    **{f"rationale_{k}": v for k, v in row.rationales.items()},
                    "latency_ms": row.latency_ms,
                }
                data.append(row_data)
            
            self._df = pd.DataFrame(data)
        
        return self._df
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of evaluation results."""
        return {
            "num_samples": len(self.rows),
            "aggregate_scores": self.aggregate_scores,
            "avg_latency_ms": sum(r.latency_ms for r in self.rows) / len(self.rows),
        }
    
    def __repr__(self) -> str:
        summary = self.summary()
        return f"""EvaluationResults(
    samples={summary['num_samples']},
    avg_latency={summary['avg_latency_ms']:.2f}ms,
    scores={summary['aggregate_scores']}
)"""


def evaluate(
    data: Union[List[Dict[str, Any]], pd.DataFrame],
    predict_fn: Callable,
    scorers: List[Union[Scorer, Callable]],
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    log_to_mlflow: bool = True,
    verbose: bool = True,
) -> EvaluationResults:
    """
    Evaluate a GenAI application using scorers.
    
    This function is similar to mlflow.genai.evaluate() and provides a comprehensive
    evaluation framework for LLM applications.
    
    Args:
        data: Evaluation dataset as list of dicts or DataFrame. Each item should have:
              - "inputs": Dict of inputs to predict_fn
              - "expectations": Optional dict of expected outputs/ground truth
        predict_fn: Function that takes inputs and returns a prediction
        scorers: List of Scorer instances or callable functions
        experiment_name: MLflow experiment name
        run_name: MLflow run name
        log_to_mlflow: Whether to log results to MLflow
        verbose: Whether to print progress
        
    Returns:
        EvaluationResults object with detailed results
        
    Example:
        ```python
        import mlflowlite as mf
        from mlflowlite.evaluation import evaluate, Correctness, Guidelines, scorer
        
        # Define prediction function
        def qa_predict_fn(question: str) -> str:
            response = mf.completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": question}]
            )
            return response.content
        
        # Define dataset
        eval_dataset = [
            {
                "inputs": {"question": "What is the capital of France?"},
                "expectations": {"expected_response": "Paris"},
            },
        ]
        
        # Define scorers
        @scorer
        def is_concise(outputs: str) -> bool:
            return len(outputs.split()) <= 5
        
        scorers = [
            Correctness(),
            Guidelines(name="is_english", guidelines="The answer must be in English"),
            is_concise,
        ]
        
        # Run evaluation
        results = evaluate(
            data=eval_dataset,
            predict_fn=qa_predict_fn,
            scorers=scorers,
        )
        ```
    """
    # Convert data to list of dicts if DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.to_dict('records')
    
    # Start MLflow run if needed
    run_id = None
    if log_to_mlflow:
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        
        mlflow.start_run(run_name=run_name or "evaluation")
        run_id = mlflow.active_run().info.run_id
    
    try:
        # Evaluate each sample
        rows = []
        all_scores: Dict[str, List[Union[float, int, bool]]] = {}
        
        for idx, sample in enumerate(data):
            if verbose:
                print(f"Evaluating sample {idx + 1}/{len(data)}...")
            
            inputs = sample.get("inputs", {})
            expectations = sample.get("expectations", None)
            
            # Get prediction
            start_time = time.time()
            try:
                # Call predict_fn with inputs
                if isinstance(inputs, dict):
                    # Try to unpack dict as kwargs
                    import inspect
                    sig = inspect.signature(predict_fn)
                    params = list(sig.parameters.keys())
                    
                    if len(params) == 1 and params[0] in inputs:
                        # Single parameter matching key in inputs
                        outputs = predict_fn(inputs[params[0]])
                    elif len(params) == len(inputs):
                        # Multiple parameters matching input keys
                        outputs = predict_fn(**inputs)
                    else:
                        # Pass entire inputs dict
                        outputs = predict_fn(inputs)
                else:
                    outputs = predict_fn(inputs)
            except Exception as e:
                if verbose:
                    print(f"  Error in prediction: {e}")
                outputs = f"ERROR: {str(e)}"
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Run scorers
            scores = {}
            rationales = {}
            
            for scorer in scorers:
                scorer_name = getattr(scorer, 'name', scorer.__name__ if hasattr(scorer, '__name__') else 'custom')
                
                try:
                    result = scorer.score(
                        inputs=inputs,
                        outputs=outputs,
                        expectations=expectations,
                    )
                    
                    scores[scorer_name] = result.score
                    rationales[scorer_name] = result.rationale or ""
                    
                    # Track for aggregation
                    if scorer_name not in all_scores:
                        all_scores[scorer_name] = []
                    all_scores[scorer_name].append(result.score)
                    
                    if verbose:
                        print(f"  {scorer_name}: {result.score}")
                        if result.rationale:
                            print(f"    → {result.rationale}")
                    
                except Exception as e:
                    if verbose:
                        print(f"  Error in scorer '{scorer_name}': {e}")
                    scores[scorer_name] = 0.0
                    rationales[scorer_name] = f"Error: {str(e)}"
            
            # Create row
            row = EvaluationRow(
                inputs=inputs,
                outputs=outputs,
                expectations=expectations,
                scores=scores,
                rationales=rationales,
                latency_ms=latency_ms,
            )
            rows.append(row)
        
        # Calculate aggregate scores
        aggregate_scores = {}
        for scorer_name, score_list in all_scores.items():
            # Convert boolean scores to 0/1 for aggregation
            numeric_scores = [
                float(s) if not isinstance(s, bool) else (1.0 if s else 0.0)
                for s in score_list
            ]
            aggregate_scores[f"{scorer_name}/mean"] = sum(numeric_scores) / len(numeric_scores)
            aggregate_scores[f"{scorer_name}/min"] = min(numeric_scores)
            aggregate_scores[f"{scorer_name}/max"] = max(numeric_scores)
        
        # Log to MLflow
        if log_to_mlflow and mlflow.active_run():
            # Log aggregate metrics
            for metric_name, value in aggregate_scores.items():
                mlflow.log_metric(metric_name, value)
            
            # Log parameters
            mlflow.log_param("num_samples", len(data))
            mlflow.log_param("num_scorers", len(scorers))
            mlflow.log_param("scorer_names", ",".join([
                getattr(s, 'name', s.__name__ if hasattr(s, '__name__') else 'custom')
                for s in scorers
            ]))
            
            # Log results as artifact
            df = EvaluationResults(rows, aggregate_scores, run_id).to_dataframe()
            mlflow.log_table(df, "evaluation_results.json")
            
            if verbose:
                print(f"\nLogged results to MLflow run: {run_id}")
        
        # Create results object
        results = EvaluationResults(
            rows=rows,
            aggregate_scores=aggregate_scores,
            run_id=run_id,
        )
        
        if verbose:
            print("\n" + "="*60)
            print("EVALUATION SUMMARY")
            print("="*60)
            print(results)
            print("="*60)
        
        return results
    
    finally:
        if log_to_mlflow and mlflow.active_run():
            mlflow.end_run()


def evaluate_with_traces(
    traces: List[Any],
    scorers: List[Union[Scorer, Callable]],
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    log_to_mlflow: bool = True,
    verbose: bool = True,
) -> EvaluationResults:
    """
    Evaluate existing traces with scorers.
    
    This is useful when you've already captured traces from your application
    and want to evaluate them retroactively.
    
    Args:
        traces: List of AgentTrace objects
        scorers: List of Scorer instances
        experiment_name: MLflow experiment name
        run_name: MLflow run name
        log_to_mlflow: Whether to log to MLflow
        verbose: Whether to print progress
        
    Returns:
        EvaluationResults object
    """
    # Convert traces to evaluation dataset format
    data = []
    for trace in traces:
        data.append({
            "inputs": {"query": trace.input_query},
            "expectations": None,  # Usually not available for traces
        })
    
    # Create predict function that just returns trace output
    trace_outputs = [trace.output for trace in traces]
    current_idx = [0]
    
    def predict_fn_from_trace(query: str) -> str:
        output = trace_outputs[current_idx[0]]
        current_idx[0] += 1
        return output
    
    # Run evaluation
    return evaluate(
        data=data,
        predict_fn=predict_fn_from_trace,
        scorers=scorers,
        experiment_name=experiment_name,
        run_name=run_name,
        log_to_mlflow=log_to_mlflow,
        verbose=verbose,
    )

