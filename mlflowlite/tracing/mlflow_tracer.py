"""MLflow-based tracing for agent execution."""

import time
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import mlflow
from mlflow.entities import SpanType


@dataclass
class TraceStep:
    """A single step in the agent's execution trace."""
    step_id: str
    step_type: str  # "reasoning", "tool_call", "tool_result", "response"
    content: str
    timestamp: float
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AgentTrace:
    """Complete trace of an agent execution."""
    trace_id: str
    agent_name: str
    input_query: str
    output: str
    steps: List[TraceStep]
    start_time: float
    end_time: float
    total_tokens: int = 0
    total_cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["duration"] = self.duration
        return data
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        tool_calls = [s for s in self.steps if s.step_type == "tool_call"]
        reasoning_steps = [s for s in self.steps if s.step_type == "reasoning"]
        
        summary = f"""Agent Trace Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Agent: {self.agent_name}
Query: {self.input_query[:100]}{'...' if len(self.input_query) > 100 else ''}
Duration: {self.duration:.2f}s
Total Steps: {len(self.steps)}
Reasoning Steps: {len(reasoning_steps)}
Tool Calls: {len(tool_calls)}
Tokens Used: {self.total_tokens}
Estimated Cost: ${self.total_cost:.4f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        if tool_calls:
            summary += "\nTools Used:\n"
            for i, step in enumerate(tool_calls, 1):
                tool_name = step.metadata.get("tool_name", "unknown")
                summary += f"  {i}. {tool_name}\n"
        
        return summary


class MLflowTracer:
    """Traces agent execution to MLflow."""
    
    def __init__(
        self,
        agent_name: str,
        experiment_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
    ):
        """
        Initialize MLflow tracer.
        
        Args:
            agent_name: Name of the agent
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking URI
        """
        self.agent_name = agent_name
        
        # Configure MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Use provided experiment name (should always be provided now from Agent)
        exp_name = experiment_name
        
        # Set experiment (will be created if it doesn't exist by the main API)
        if exp_name:
            try:
                mlflow.set_experiment(exp_name)
            except Exception as e:
                # Handle deleted experiment case
                if "already exists in deleted state" in str(e) or "deleted experiment" in str(e):
                    try:
                        # Try to restore the experiment
                        exp = mlflow.get_experiment_by_name(exp_name)
                        if exp and exp.lifecycle_stage == "deleted":
                            mlflow.tracking.MlflowClient().restore_experiment(exp.experiment_id)
                            mlflow.set_experiment(exp_name)
                    except Exception:
                        pass  # Let the main API handle it
                else:
                    # Try to create new experiment
                    try:
                        mlflow.create_experiment(exp_name)
                        mlflow.set_experiment(exp_name)
                    except Exception:
                        pass  # Continue without experiment
    
    def start_trace(self, input_query: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start a new trace."""
        trace_id = str(uuid.uuid4())
        
        self.current_trace = AgentTrace(
            trace_id=trace_id,
            agent_name=self.agent_name,
            input_query=input_query,
            output="",
            steps=[],
            start_time=time.time(),
            end_time=0,
            metadata=metadata or {},
        )
        
        # Start MLflow run
        run = mlflow.start_run(run_name=f"{self.agent_name}_{trace_id[:8]}")
        self.active_run_id = run.info.run_id
        
        # Log input
        mlflow.log_param("agent_name", self.agent_name)
        mlflow.log_param("input_query", input_query[:250])
        
        if metadata:
            for key, value in metadata.items():
                mlflow.log_param(f"metadata_{key}", str(value)[:250])
        
        return trace_id
    
    def add_step(
        self,
        step_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ) -> TraceStep:
        """Add a step to the current trace."""
        if not self.current_trace:
            raise RuntimeError("No active trace. Call start_trace() first.")
        
        step = TraceStep(
            step_id=str(uuid.uuid4()),
            step_type=step_type,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {},
            parent_id=parent_id,
        )
        
        self.current_trace.steps.append(step)
        
        # Log to MLflow
        try:
            # Create span for this step
            span_type = self._map_step_type_to_span_type(step_type)
            with mlflow.start_span(
                name=f"{step_type}_{step.step_id[:8]}",
                span_type=span_type
            ) as span:
                span.set_attribute("step_type", step_type)
                span.set_attribute("content_preview", content[:200])
                
                if metadata:
                    for key, value in metadata.items():
                        span.set_attribute(key, str(value)[:200])
        except Exception as e:
            # If tracing fails, continue anyway
            print(f"Warning: Failed to create MLflow span: {e}")
        
        return step
    
    def end_trace(
        self,
        output: str,
        total_tokens: int = 0,
        total_cost: float = 0.0,
    ) -> AgentTrace:
        """End the current trace."""
        if not self.current_trace:
            raise RuntimeError("No active trace.")
        
        self.current_trace.end_time = time.time()
        self.current_trace.output = output
        self.current_trace.total_tokens = total_tokens
        self.current_trace.total_cost = total_cost
        
        # Log to MLflow
        try:
            mlflow.log_metric("duration_seconds", self.current_trace.duration)
            mlflow.log_metric("total_steps", len(self.current_trace.steps))
            mlflow.log_metric("total_tokens", total_tokens)
            mlflow.log_metric("total_cost", total_cost)
            
            mlflow.log_param("output_preview", output[:250])
            
            # Log trace as artifact
            trace_dict = self.current_trace.to_dict()
            mlflow.log_dict(trace_dict, "trace.json")
            
            # Log summary
            summary = self.current_trace.summary()
            mlflow.log_text(summary, "trace_summary.txt")
            
            # End run
            mlflow.end_run()
        except Exception as e:
            print(f"Warning: Failed to log to MLflow: {e}")
        
        completed_trace = self.current_trace
        self.current_trace = None
        self.active_run_id = None
        
        return completed_trace
    
    def _map_step_type_to_span_type(self, step_type: str) -> str:
        """Map our step types to MLflow span types."""
        mapping = {
            "reasoning": SpanType.LLM,
            "tool_call": SpanType.TOOL,
            "tool_result": SpanType.TOOL,
            "response": SpanType.LLM,
        }
        return mapping.get(step_type, SpanType.UNKNOWN)

