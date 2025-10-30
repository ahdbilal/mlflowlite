"""Tests for Agent class."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from mlflowlite.agent import Agent, AgentResult
from mlflowlite.llm.base import LLMResponse, MessageRole
from mlflowlite.tools.base import Tool


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    return LLMResponse(
        content="This is a test response",
        role=MessageRole.ASSISTANT,
        usage={"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50},
        finish_reason="stop",
    )


def test_agent_initialization():
    """Test agent initialization."""
    with patch('mlflowlite.agent.get_provider'):
        agent = Agent(
            name="test_agent",
            model="gpt-4o",
            tools=["calculator"],
            temperature=0.7,
        )
        
        assert agent.name == "test_agent"
        assert agent.model == "gpt-4o"
        assert agent.temperature == 0.7
        assert "calculator" in agent.tools


def test_agent_with_custom_tool():
    """Test agent with custom tool."""
    def custom_func(x: int) -> int:
        return x * 2
    
    custom_tool = Tool(name="double", func=custom_func)
    
    with patch('mlflowlite.agent.get_provider'):
        agent = Agent(
            name="test_agent",
            model="gpt-4o",
            tools=[custom_tool],
        )
        
        assert "double" in agent.tools
        assert agent.tools["double"].name == "double"


@patch('mlflowlite.agent.MLflowTracer')
@patch('mlflowlite.agent.get_provider')
def test_agent_run(mock_get_provider, mock_tracer_class, mock_llm_response):
    """Test agent run."""
    # Setup mocks
    mock_provider = Mock()
    mock_provider.complete.return_value = mock_llm_response
    mock_provider.count_tokens.return_value = 50
    mock_get_provider.return_value = mock_provider
    
    mock_tracer = Mock()
    mock_tracer.start_trace.return_value = "trace_id_123"
    mock_trace = Mock()
    mock_trace.trace_id = "trace_id_123"
    mock_trace.duration = 1.0
    mock_trace.total_tokens = 100
    mock_trace.total_cost = 0.01
    mock_trace.steps = []
    mock_tracer.end_trace.return_value = mock_trace
    mock_tracer_class.return_value = mock_tracer
    
    # Create agent
    agent = Agent(
        name="test_agent",
        model="gpt-4o",
    )
    
    # Run
    result = agent.run("Test query")
    
    # Assertions
    assert isinstance(result, AgentResult)
    assert result.response == "This is a test response"
    assert result.trace == mock_trace


def test_agent_get_info():
    """Test agent get_info method."""
    with patch('mlflowlite.agent.get_provider'):
        agent = Agent(
            name="test_agent",
            model="gpt-4o",
            tools=["calculator", "search"],
        )
        
        info = agent.get_info()
        
        assert info["name"] == "test_agent"
        assert info["model"] == "gpt-4o"
        assert info["num_tools"] == 2
        assert "calculator" in info["tools"]
        assert "search" in info["tools"]


def test_agent_repr():
    """Test agent string representation."""
    with patch('mlflowlite.agent.get_provider'):
        agent = Agent(
            name="test_agent",
            model="gpt-4o",
            tools=["calculator"],
        )
        
        repr_str = repr(agent)
        assert "test_agent" in repr_str
        assert "gpt-4o" in repr_str
        assert "calculator" in repr_str


def test_agent_result_str():
    """Test AgentResult string representation."""
    mock_trace = Mock()
    result = AgentResult(
        response="Test response",
        trace=mock_trace,
    )
    
    assert str(result) == "Test response"


