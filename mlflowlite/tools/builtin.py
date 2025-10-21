"""Built-in tools for agents."""

from typing import Dict, Optional, List
import json
import requests
from mlflowlite.tools.base import Tool


# ============= Search Tool =============

def web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web for information.
    
    Args:
        query: Search query
        num_results: Number of results to return
    
    Returns:
        Search results as formatted text
    """
    # This is a placeholder - in production, integrate with real search API
    # (e.g., Google Custom Search, Bing Search, SerpAPI, etc.)
    
    return f"""Search results for: {query}

[Note: This is a placeholder. Configure a search API for real results.]

Mock Results:
1. Result 1: Information about {query}...
2. Result 2: More details on {query}...
3. Result 3: Additional context for {query}...
"""


# ============= Summarize Tool =============

def summarize_text(text: str, max_length: int = 200) -> str:
    """
    Summarize a long text.
    
    Args:
        text: Text to summarize
        max_length: Maximum length of summary
    
    Returns:
        Summarized text
    """
    # Simple extractive summary - take first sentences
    sentences = text.split(". ")
    summary = []
    length = 0
    
    for sentence in sentences:
        if length + len(sentence) > max_length:
            break
        summary.append(sentence)
        length += len(sentence)
    
    result = ". ".join(summary)
    if result and not result.endswith("."):
        result += "..."
    
    return result or text[:max_length] + "..."


# ============= Calculator Tool =============

def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate
    
    Returns:
        Result of the calculation
    """
    try:
        # Safe eval with limited scope
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow,
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


# ============= HTTP Request Tool =============

def http_request(
    url: str,
    method: str = "GET",
    headers: Optional[str] = None,
    body: Optional[str] = None
) -> str:
    """
    Make an HTTP request.
    
    Args:
        url: URL to request
        method: HTTP method (GET, POST, etc.)
        headers: JSON string of headers
        body: Request body (for POST, PUT, etc.)
    
    Returns:
        Response text
    """
    try:
        headers_dict = json.loads(headers) if headers else {}
        
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=headers_dict,
            data=body,
            timeout=30
        )
        
        return f"Status: {response.status_code}\n\n{response.text[:1000]}"
    except Exception as e:
        return f"Error making request: {str(e)}"


# ============= File Operations =============

def read_file(filepath: str) -> str:
    """
    Read contents of a file.
    
    Args:
        filepath: Path to file
    
    Returns:
        File contents
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


def write_file(filepath: str, content: str) -> str:
    """
    Write content to a file.
    
    Args:
        filepath: Path to file
        content: Content to write
    
    Returns:
        Success message
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


# ============= Tool Registry =============

BUILTIN_TOOLS: Dict[str, Tool] = {
    "search": Tool(
        name="web_search",
        func=web_search,
        description="Search the web for information about any topic",
    ),
    "summarize": Tool(
        name="summarize_text",
        func=summarize_text,
        description="Summarize long text into a shorter version",
    ),
    "calculator": Tool(
        name="calculator",
        func=calculator,
        description="Evaluate mathematical expressions and perform calculations",
    ),
    "http_request": Tool(
        name="http_request",
        func=http_request,
        description="Make HTTP requests to APIs or websites",
    ),
    "read_file": Tool(
        name="read_file",
        func=read_file,
        description="Read contents from a file",
    ),
    "write_file": Tool(
        name="write_file",
        func=write_file,
        description="Write content to a file",
    ),
}


def get_builtin_tool(name: str) -> Optional[Tool]:
    """Get a built-in tool by name."""
    return BUILTIN_TOOLS.get(name)


def list_builtin_tools() -> List[str]:
    """List all available built-in tools."""
    return list(BUILTIN_TOOLS.keys())

