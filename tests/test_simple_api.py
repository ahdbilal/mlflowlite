"""Tests for simple API."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from mlflowlite.simple_api import SimpleAPI, ai_query, QueryResult
from mlflowlite.llm.base import LLMResponse, MessageRole


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    return LLMResponse(
        content="This is a test response",
        role=MessageRole.ASSISTANT,
        usage={"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50},
    )


@pytest.fixture
def simple_api():
    """Create a SimpleAPI instance."""
    return SimpleAPI()


@patch('mlflowlite.simple_api.mlflow')
@patch('mlflowlite.simple_api.get_provider')
def test_query_basic(mock_get_provider, mock_mlflow, simple_api, mock_llm_response):
    """Test basic query functionality."""
    # Setup mocks
    mock_provider = Mock()
    mock_provider.complete.return_value = mock_llm_response
    mock_get_provider.return_value = mock_provider
    
    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_id"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
    mock_mlflow.start_span.return_value.__enter__.return_value = MagicMock()
    
    # Execute query
    result = simple_api.query(
        model="gpt-4o",
        prompt="Test prompt",
        evaluate=True,
    )
    
    # Assertions
    assert isinstance(result, QueryResult)
    assert result.response == "This is a test response"
    assert result.model == "gpt-4o"
    assert "latency_ms" in result.metrics
    assert "tokens" in result.metrics
    assert result.run_id == "test_run_id"


@patch('mlflowlite.simple_api.mlflow')
@patch('mlflowlite.simple_api.get_provider')
def test_query_with_input_text(mock_get_provider, mock_mlflow, simple_api, mock_llm_response):
    """Test query with input text."""
    mock_provider = Mock()
    mock_provider.complete.return_value = mock_llm_response
    mock_get_provider.return_value = mock_provider
    
    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_id"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
    mock_mlflow.start_span.return_value.__enter__.return_value = MagicMock()
    
    result = simple_api.query(
        model="claude-3-5-sonnet",
        prompt="Summarize this",
        input_text="This is input text to summarize",
        evaluate=True,
    )
    
    assert result.response == "This is a test response"
    assert result.model == "claude-3-5-sonnet"


def test_estimate_cost(simple_api):
    """Test cost estimation."""
    # Test GPT-4
    cost = simple_api._estimate_cost("gpt-4o", 1000, 1000)
    assert cost > 0
    
    # Test Claude
    cost = simple_api._estimate_cost("claude-3-5-sonnet", 1000, 1000)
    assert cost > 0
    
    # Test unknown model
    cost = simple_api._estimate_cost("unknown-model", 1000, 1000)
    assert cost > 0


def test_generate_intelligen_summary(simple_api):
    """Test IntelliGen-style summary generation."""
    summary = simple_api._generate_intelligen_summary(
        model="gpt-4o",
        prompt="Test prompt",
        response="Test response",
        duration=1.5,
        tokens=200,
        cost=0.01,
        evaluate=True,
    )
    
    assert "gpt-4o" in summary
    assert "1.50s" in summary or "1.5s" in summary
    assert "200 tokens" in summary
    assert "$0.01" in summary


def test_quick_evaluate(simple_api):
    """Test quick evaluation."""
    scores = simple_api._quick_evaluate(
        prompt="Test prompt",
        response="This is a test response with some content",
        duration=1.0,
        tokens=100,
    )
    
    assert "helpfulness" in scores
    assert "conciseness" in scores
    assert "efficiency" in scores
    assert "cost_efficiency" in scores
    
    # All scores should be between 0 and 1
    for score in scores.values():
        assert 0 <= score <= 1


def test_compare_models(simple_api):
    """Test model comparison."""
    result1 = QueryResult(
        response="Response 1",
        trace_id="trace1",
        model="gpt-4o",
        summary="Summary 1",
        metrics={"latency_ms": 1000, "cost_usd": 0.01, "helpfulness": 0.8},
    )
    
    result2 = QueryResult(
        response="Response 2",
        trace_id="trace2",
        model="claude-3-5-sonnet",
        summary="Summary 2",
        metrics={"latency_ms": 900, "cost_usd": 0.015, "helpfulness": 0.85},
    )
    
    comparison = simple_api.compare(result1, result2)
    
    assert "model_comparison" in comparison
    assert "performance" in comparison
    assert "quality" in comparison
    
    # Check winners
    assert comparison["performance"]["latency_ms"]["winner"] == "claude-3-5-sonnet"
    assert comparison["performance"]["cost_usd"]["winner"] == "gpt-4o"


def test_suggest_improvement(simple_api):
    """Test improvement suggestions."""
    result = QueryResult(
        response="Short response",
        trace_id="trace1",
        model="gpt-4o",
        summary="Summary",
        metrics={
            "helpfulness": 0.6,
            "conciseness": 0.5,
            "latency_ms": 5000,
            "cost_usd": 0.1,
        },
    )
    
    suggestions = simple_api.suggest_improvement(result)
    
    assert "current_metrics" in suggestions
    assert "improvements" in suggestions
    assert len(suggestions["improvements"]) > 0


def test_query_result_str():
    """Test QueryResult string representation."""
    result = QueryResult(
        response="Test response",
        trace_id="trace1",
        model="gpt-4o",
        summary="Summary",
        metrics={},
    )
    
    assert str(result) == "Test response"

