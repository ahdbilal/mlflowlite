"""Base tool interface."""

from typing import Callable, Any, Dict, Optional, List
from dataclasses import dataclass
from inspect import signature, Parameter
import json


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Tool:
    """A tool that an agent can use."""
    
    def __init__(
        self,
        name: str,
        func: Callable,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a tool.
        
        Args:
            name: Tool name
            func: Function to execute
            description: Tool description (uses docstring if not provided)
            parameters: JSON schema for parameters (auto-generated if not provided)
        """
        self.name = name
        self.func = func
        self.description = description or func.__doc__ or f"Execute {name}"
        self.parameters = parameters or self._generate_parameters()
    
    def _generate_parameters(self) -> Dict[str, Any]:
        """Auto-generate parameter schema from function signature."""
        sig = signature(self.func)
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            
            param_type = "string"  # Default
            if param.annotation != Parameter.empty:
                annotation = param.annotation
                if annotation == int:
                    param_type = "integer"
                elif annotation == float:
                    param_type = "number"
                elif annotation == bool:
                    param_type = "boolean"
                elif annotation == list or annotation == List:
                    param_type = "array"
                elif annotation == dict or annotation == Dict:
                    param_type = "object"
            
            properties[param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name}",
            }
            
            if param.default == Parameter.empty:
                required.append(param_name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""
        try:
            result = self.func(**kwargs)
            return ToolResult(
                success=True,
                output=result,
                metadata={"tool": self.name, "args": kwargs}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                metadata={"tool": self.name, "args": kwargs}
            )
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert tool to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
    
    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"

