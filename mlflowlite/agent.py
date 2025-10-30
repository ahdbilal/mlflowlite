"""Core Agent implementation."""

import time
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from mlflowlite.llm.base import LLMProvider, Message, MessageRole
from mlflowlite.llm.providers import get_provider
from mlflowlite.litellm_style_api import _get_experiment_name
from mlflowlite.tools.base import Tool, ToolResult
from mlflowlite.tools.builtin import get_builtin_tool
from mlflowlite.tracing.mlflow_tracer import MLflowTracer, AgentTrace
from mlflowlite.prompts.registry import PromptRegistry
from mlflowlite.evaluation.evaluator import AgentEvaluator, EvaluationResult
from mlflowlite.optimization.dspy_optimizer import DSPyOptimizer


@dataclass
class AgentResult:
    """Result from an agent execution."""
    response: str
    trace: AgentTrace
    evaluation: Optional[EvaluationResult] = None
    metadata: Dict[str, Any] = None
    
    def __str__(self) -> str:
        return self.response


class Agent:
    """
    Unified interface for LLM calls - from simple queries to complex workflows.

    Simple usage (no prompt versioning):
        >>> agent = Agent(model="claude-3-5-sonnet")
        >>> response = agent(prompt="What is 2+2?")
        >>> print(response.content)

    Template usage (with MLflow Prompt Registry):
        >>> agent = Agent(
        ...     model="claude-3-5-sonnet",
        ...     prompt="Analyze this support ticket:\n\nTicket: {{ticket}}\n\nProvide: ISSUE, CAUSE, FIX",
        ...     prompt_name="support_bot"  # Registers to MLflow!
        ... )
        >>> response = agent(ticket="User can't login")
        >>> print(response.content)
        
        # View in MLflow UI → Prompts tab → "agent_support_bot_prompt"
        # Works on Databricks (Unity Catalog) and locally

    Advanced usage (tools + reasoning):
        >>> agent = Agent(
        ...     model="claude-3-5-sonnet",
        ...     prompt="{{query}}",
        ...     prompt_name="research_bot",
        ...     tools=["search", "calculator"]
        ... )
        >>> result = agent.run(query="What's the GDP of France?")
        >>> print(result.response)
    
    Note: 'prompt_name' is optional. Only needed for MLflow Prompt Registry versioning.
    """
    
    def __init__(
        self,
        model: str,
        prompt: Optional[str] = None,          # Prompt template with {{variables}}
        prompt_name: Optional[str] = None,     # Name for MLflow Prompt Registry
        system_prompt: Optional[str] = None,   # Legacy: kept for backwards compatibility
        name: Optional[str] = None,            # Legacy: alias for prompt_name
        tools: Optional[List[Union[str, Tool]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        max_iterations: int = 10,
        gateway_mode: bool = False,
        gateway_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize an Agent.
        
        Args:
            model: LLM model name (e.g., "claude-3-5-sonnet", "gpt-4o")
            prompt: Prompt template with {{variables}} in MLflow format
            prompt_name: Name for MLflow Prompt Registry (triggers registration)
            system_prompt: Legacy system prompt (backwards compatibility)
            name: Legacy agent name (alias for prompt_name)
            tools: List of tool names or Tool instances
            temperature: LLM temperature
            max_tokens: Maximum tokens for LLM responses
            api_key: API key for LLM provider
            tracking_uri: MLflow tracking URI
            experiment_name: MLflow experiment name
            max_iterations: Maximum reasoning iterations
            gateway_mode: Use Databricks AI Gateway
            gateway_url: Gateway URL (if gateway_mode=True)
            **kwargs: Additional LLM provider kwargs
        """
        # Handle prompt_name vs legacy name
        self.prompt_name = prompt_name or name
        self.name = self.prompt_name or f"agent_{id(self)}"  # For backwards compatibility
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.gateway_mode = gateway_mode
        self.gateway_url = gateway_url
        
        # Handle prompt vs legacy system_prompt
        self.prompt = prompt
        self.system_prompt = system_prompt
        
        # Convert legacy system_prompt to prompt format if needed
        if system_prompt and not prompt:
            self.prompt = f"{system_prompt}\n\n{{{{query}}}}"
        
        # Initialize LLM provider
        self.llm_provider = self._initialize_llm_provider(
            model, temperature, max_tokens, api_key, **kwargs
        )
        
        # Initialize tools
        self.tools: Dict[str, Tool] = {}
        if tools:
            self._load_tools(tools)
        
        # Initialize prompt registry (only if prompt_name provided)
        self.prompt_registry = None
        
        if self.prompt_name and self.prompt:
            # Register prompt to MLflow
            self.prompt_registry = PromptRegistry(agent_name=self.prompt_name)
            self.prompt_registry.add_version(
                system_prompt=self.prompt,  # Store full prompt as system_prompt for now
                user_template="{{query}}",  # Default user template
                metadata={"source": "user_provided", "format": "mlflow"}
            )
        
        # Initialize tracing - use main mlflowlite experiment if not specified
        if experiment_name is None:
            # Import here to avoid circular dependency
            from mlflowlite.litellm_style_api import _get_experiment_name
            experiment_name = _get_experiment_name()  # Use same experiment as queries
        
        self.tracer = MLflowTracer(
            agent_name=name,
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
        )
        
        # Initialize evaluator
        self.evaluator = AgentEvaluator(llm_provider=self.llm_provider)
        
        # Initialize optimizer
        self.optimizer = DSPyOptimizer(
            agent_name=name,
            llm_provider=self.llm_provider,
        )
        
        # Conversation history
        self.conversation_history: List[Message] = []
    
    def _initialize_llm_provider(
        self,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        api_key: Optional[str],
        **kwargs
    ) -> LLMProvider:
        """Initialize the LLM provider."""
        if self.gateway_mode and self.gateway_url:
            # TODO: Implement gateway provider
            print(f"Warning: Gateway mode not yet fully implemented. Using direct provider.")
        
        return get_provider(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            **kwargs
        )
    
    def _load_tools(self, tools: List[Union[str, Tool]]):
        """Load tools from names or instances."""
        for tool in tools:
            if isinstance(tool, str):
                # Load built-in tool
                builtin_tool = get_builtin_tool(tool)
                if builtin_tool:
                    self.tools[builtin_tool.name] = builtin_tool
                else:
                    print(f"Warning: Unknown built-in tool '{tool}'")
            elif isinstance(tool, Tool):
                self.tools[tool.name] = tool
            else:
                print(f"Warning: Invalid tool type: {type(tool)}")
    
    def __call__(self, prompt: Optional[str] = None, **variables):
        """
        Simple callable interface - use Agent like query().
        
        Args:
            prompt: Direct prompt (if no registered prompt with {{variables}})
            **variables: Variables to fill into {{placeholders}} (e.g., query="...", ticket="...")
        
        Returns:
            Response object (same as query())
        
        Examples:
            >>> # Simple mode: Direct prompt
            >>> agent = Agent(model="claude-3-5-sonnet")
            >>> response = agent(prompt="Hello!")
            
            >>> # Template mode: Fill variables
            >>> agent = Agent(model="claude-3-5-sonnet", 
            ...               prompt="Analyze: {{ticket}}", 
            ...               prompt_name="support_bot")
            >>> response = agent(ticket="User can't login")
        """
        # Use litellm_style_api's completion for consistent Response object
        from mlflowlite.litellm_style_api import completion
        
        messages = []
        
        # Determine what prompt to use
        if self.prompt and variables:
            # Template mode: Fill {{variables}} in registered prompt
            try:
                # Simple format: replace {{var}} with values
                filled_prompt = self.prompt
                for key, value in variables.items():
                    filled_prompt = filled_prompt.replace(f"{{{{{key}}}}}", str(value))
                messages.append({"role": "user", "content": filled_prompt})
            except Exception as e:
                raise ValueError(f"Failed to fill template variables: {e}")
        
        elif prompt:
            # Simple mode: Direct prompt
            # Add system prompt if available
            if self.prompt_registry:
                try:
                    current_prompt = self.prompt_registry.get_latest()
                    if current_prompt.system_prompt:
                        messages.append({"role": "system", "content": current_prompt.system_prompt})
                except:
                    pass
            elif self.system_prompt:
                # Use provided system prompt
                messages.append({"role": "system", "content": self.system_prompt})
            
            # Add user message
            messages.append({"role": "user", "content": prompt})
        
        else:
            raise ValueError("Provide either prompt=... for simple queries, or variables (e.g., ticket=...) for template prompts")
        
        # Get prompt metadata for linking trace to prompt
        prompt_metadata = {}
        if self.prompt_registry:
            try:
                current_prompt = self.prompt_registry.get_latest()
                prompt_metadata = {
                    "prompt_name": self.prompt_name,
                    "prompt_version": current_prompt.version,
                    "prompt_registry_name": self.prompt_registry.prompt_name,
                }
            except:
                pass
        
        return completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **prompt_metadata  # Pass prompt info to trace
        )
    
    def run(
        self,
        query: str,
        evaluate: bool = False,
        ground_truth: Optional[str] = None,
        **kwargs
    ) -> AgentResult:
        """
        Run the agent on a query.
        
        Args:
            query: User query
            evaluate: Whether to evaluate the execution
            ground_truth: Expected output (for evaluation)
            **kwargs: Additional parameters
        
        Returns:
            AgentResult with response, trace, and optional evaluation
        """
        # Start trace
        trace_id = self.tracer.start_trace(
            input_query=query,
            metadata={
                "model": self.model,
                "temperature": self.temperature,
                "num_tools": len(self.tools),
            }
        )
        
        try:
            # Initialize conversation with system prompt
            messages = []
            current_prompt = None
            
            if self.prompt_registry:
                # Get current prompt from registry
                current_prompt = self.prompt_registry.get_latest()
                if current_prompt.system_prompt:
                    messages.append(Message(
                        role=MessageRole.SYSTEM,
                        content=current_prompt.system_prompt
                    ))
            elif self.system_prompt:
                # Use provided system prompt
                messages.append(Message(
                    role=MessageRole.SYSTEM,
                    content=self.system_prompt
                ))
            
            # Add examples if available
            if current_prompt and current_prompt.examples:
                for example in current_prompt.examples[:3]:  # Limit to 3 examples
                    messages.append(Message(
                        role=MessageRole.USER,
                        content=example["input"]
                    ))
                    messages.append(Message(
                        role=MessageRole.ASSISTANT,
                        content=example["output"]
                    ))
            
            # Add user query
            if current_prompt and hasattr(current_prompt, 'user_template'):
                user_message = current_prompt.user_template.format(query=query)
            else:
                user_message = query
            
            messages.append(Message(
                role=MessageRole.USER,
                content=user_message
            ))
            
            # Execute agent loop
            response = self._execute_loop(messages)
            
            # Calculate token usage
            total_tokens = sum(
                self.llm_provider.count_tokens(m.content)
                for m in messages
            )
            
            # Estimate cost (rough estimate, $0.01 per 1K tokens)
            estimated_cost = (total_tokens / 1000) * 0.01
            
            # End trace
            trace = self.tracer.end_trace(
                output=response,
                total_tokens=total_tokens,
                total_cost=estimated_cost,
            )
            
            # Evaluate if requested
            evaluation = None
            if evaluate:
                evaluation = self.evaluator.evaluate(
                    trace=trace,
                    ground_truth=ground_truth,
                )
            
            return AgentResult(
                response=response,
                trace=trace,
                evaluation=evaluation,
                metadata={"trace_id": trace_id}
            )
        
        except Exception as e:
            # End trace with error
            error_msg = f"Error: {str(e)}"
            self.tracer.end_trace(output=error_msg, total_tokens=0)
            raise RuntimeError(f"Agent execution failed: {str(e)}") from e
    
    def _execute_loop(self, messages: List[Message]) -> str:
        """Execute the agent reasoning loop."""
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Get tools in OpenAI format
            tools_schema = [tool.to_openai_format() for tool in self.tools.values()] if self.tools else None
            
            # Call LLM
            self.tracer.add_step(
                step_type="reasoning",
                content=f"Iteration {iteration}: Calling LLM",
                metadata={"iteration": iteration}
            )
            
            response = self.llm_provider.complete(
                messages=messages,
                tools=tools_schema,
            )
            
            # Add assistant message
            messages.append(Message(
                role=MessageRole.ASSISTANT,
                content=response.content or "",
                tool_calls=response.tool_calls,
            ))
            
            # Check if we have tool calls
            if response.has_tool_calls and response.tool_calls:
                # Execute tools
                for tool_call in response.tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args_str = tool_call["function"]["arguments"]
                    
                    try:
                        tool_args = json.loads(tool_args_str)
                    except json.JSONDecodeError:
                        tool_args = {}
                    
                    # Log tool call
                    self.tracer.add_step(
                        step_type="tool_call",
                        content=f"Calling {tool_name} with args: {tool_args}",
                        metadata={
                            "tool_name": tool_name,
                            "args": tool_args,
                            "tool_call_id": tool_call.get("id"),
                        }
                    )
                    
                    # Execute tool
                    if tool_name in self.tools:
                        result = self.tools[tool_name].execute(**tool_args)
                    else:
                        result = ToolResult(
                            success=False,
                            output=None,
                            error=f"Tool {tool_name} not found"
                        )
                    
                    # Log tool result
                    self.tracer.add_step(
                        step_type="tool_result",
                        content=str(result.output) if result.success else result.error or "Unknown error",
                        metadata={
                            "tool_name": tool_name,
                            "success": result.success,
                        }
                    )
                    
                    # Add tool result to messages
                    messages.append(Message(
                        role=MessageRole.TOOL,
                        content=str(result.output) if result.success else result.error or "Tool execution failed",
                        name=tool_name,
                        tool_call_id=tool_call.get("id"),
                    ))
                
                # Continue loop to get next response
                continue
            
            # No tool calls, we have a final response
            self.tracer.add_step(
                step_type="response",
                content=response.content or "",
                metadata={"final": True}
            )
            
            return response.content or ""
        
        # Max iterations reached
        last_message = messages[-1].content if messages else "No response"
        return f"Max iterations reached. Last response: {last_message}"
    
    def chat(
        self,
        message: str,
        reset: bool = False,
        **kwargs
    ) -> str:
        """
        Chat with the agent (maintains conversation history).
        
        Args:
            message: User message
            reset: Reset conversation history
            **kwargs: Additional parameters
        
        Returns:
            Agent response
        """
        if reset:
            self.conversation_history = []
        
        # Start trace for this chat turn
        trace_id = self.tracer.start_trace(
            input_query=message,
            metadata={"mode": "chat", "history_length": len(self.conversation_history)}
        )
        
        try:
            # Add user message to history
            self.conversation_history.append(Message(
                role=MessageRole.USER,
                content=message
            ))
            
            # Build full conversation
            messages = []
            
            if self.prompt_registry:
                current_prompt = self.prompt_registry.get_latest()
                if current_prompt.system_prompt:
                    messages.append(Message(role=MessageRole.SYSTEM, content=current_prompt.system_prompt))
            elif self.system_prompt:
                messages.append(Message(role=MessageRole.SYSTEM, content=self.system_prompt))
            
            messages.extend(self.conversation_history)
            
            # Execute
            response = self._execute_loop(messages)
            
            # Add assistant response to history
            self.conversation_history.append(Message(
                role=MessageRole.ASSISTANT,
                content=response
            ))
            
            # End trace
            self.tracer.end_trace(output=response, total_tokens=0, total_cost=0.0)
            
            return response
        except Exception as e:
            # End trace with error
            self.tracer.end_trace(output=f"Error: {str(e)}", total_tokens=0)
            raise
    
    def optimize(
        self,
        num_runs: int = 10,
        test_queries: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Automatically optimize the agent's prompt based on test runs.
        
        Args:
            num_runs: Number of test runs to perform
            test_queries: List of test queries (auto-generated if not provided)
        
        Returns:
            Optimization results
        """
        return self.optimizer.auto_optimize(
            agent=self,
            num_runs=num_runs,
            test_queries=test_queries,
        )
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "name": self.name,
            "model": self.model,
            "temperature": self.temperature,
            "num_tools": len(self.tools),
            "tools": list(self.tools.keys()),
            "current_prompt_version": self.prompt_registry.get_latest().version,
            "total_prompt_versions": len(self.prompt_registry.versions),
        }
    
    def __repr__(self) -> str:
        tools_str = ", ".join(self.tools.keys()) if self.tools else "none"
        return f"Agent(name={self.name!r}, model={self.model!r}, tools=[{tools_str}])"

