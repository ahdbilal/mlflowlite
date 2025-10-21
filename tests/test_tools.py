"""Tests for tools."""

import pytest
from mlflowlite.tools.base import Tool, ToolResult
from mlflowlite.tools.builtin import (
    calculator,
    summarize_text,
    get_builtin_tool,
    list_builtin_tools,
)


def test_tool_creation():
    """Test creating a tool."""
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    tool = Tool(name="add", func=add)
    
    assert tool.name == "add"
    assert "Add two numbers" in tool.description


def test_tool_execution():
    """Test tool execution."""
    def multiply(a: int, b: int) -> int:
        return a * b
    
    tool = Tool(name="multiply", func=multiply)
    result = tool.execute(a=3, b=4)
    
    assert result.success
    assert result.output == 12
    assert result.metadata["tool"] == "multiply"


def test_tool_execution_error():
    """Test tool execution with error."""
    def failing_func():
        raise ValueError("Test error")
    
    tool = Tool(name="fail", func=failing_func)
    result = tool.execute()
    
    assert not result.success
    assert result.output is None
    assert "Test error" in result.error


def test_tool_to_openai_format():
    """Test converting tool to OpenAI format."""
    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello {name}"
    
    tool = Tool(name="greet", func=greet)
    openai_format = tool.to_openai_format()
    
    assert openai_format["type"] == "function"
    assert openai_format["function"]["name"] == "greet"
    assert "name" in openai_format["function"]["parameters"]["properties"]


def test_calculator_tool():
    """Test calculator built-in tool."""
    result = calculator("2 + 2")
    assert "4" in result
    
    result = calculator("10 * 5")
    assert "50" in result


def test_summarize_tool():
    """Test summarize built-in tool."""
    long_text = "This is a very long text. " * 50
    result = summarize_text(long_text, max_length=100)
    
    assert len(result) <= 110  # Allow some buffer
    assert result.endswith("...")


def test_get_builtin_tool():
    """Test getting built-in tool."""
    tool = get_builtin_tool("calculator")
    assert tool is not None
    assert tool.name == "calculator"
    
    tool = get_builtin_tool("nonexistent")
    assert tool is None


def test_list_builtin_tools():
    """Test listing built-in tools."""
    tools = list_builtin_tools()
    
    assert "calculator" in tools
    assert "search" in tools
    assert "summarize" in tools
    assert len(tools) > 0

