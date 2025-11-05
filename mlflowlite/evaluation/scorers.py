"""Scorer system for GenAI evaluation."""

from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json


@dataclass
class ScorerResult:
    """Result from a scorer evaluation."""
    score: Union[float, int, bool]
    rationale: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Scorer(ABC):
    """Base class for all scorers."""
    
    def __init__(self, name: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize scorer.
        
        Args:
            name: Name of the scorer
            model: Model to use for LLM-based evaluation (e.g., "openai:/gpt-4o-mini")
        """
        self.name = name or self.__class__.__name__
        self.model = model or "openai:/gpt-4o-mini"
        
    @abstractmethod
    def score(
        self,
        inputs: Union[str, Dict[str, Any]],
        outputs: str,
        expectations: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> ScorerResult:
        """
        Score the input-output pair.
        
        Args:
            inputs: Input data (can be string or dict)
            outputs: Model output to evaluate
            expectations: Expected output or ground truth
            **kwargs: Additional metadata
            
        Returns:
            ScorerResult with score and rationale
        """
        pass
    
    def _get_llm_provider(self):
        """Get LLM provider based on model string."""
        from mlflowlite.llm.providers import get_provider
        
        # Parse model string (e.g., "openai:/gpt-4o-mini")
        if ":" in self.model:
            provider_name, model_name = self.model.split(":", 1)
            model_name = model_name.lstrip("/")
        else:
            provider_name = "openai"
            model_name = self.model
            
        return get_provider(provider_name, model=model_name)
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM for evaluation."""
        from mlflowlite.llm.base import Message, MessageRole
        
        try:
            provider = self._get_llm_provider()
            messages = [Message(role=MessageRole.USER, content=prompt)]
            response = provider.complete(messages)
            return response.content
        except Exception as e:
            return f"Error calling LLM: {str(e)}"


class Correctness(Scorer):
    """
    Evaluates if the answer is factually correct using LLM-as-a-Judge.
    
    This scorer uses an LLM to compare the output against the expected response
    and determine if it's factually accurate.
    """
    
    def __init__(self, model: Optional[str] = None):
        super().__init__(name="correctness", model=model)
    
    def score(
        self,
        inputs: Union[str, Dict[str, Any]],
        outputs: str,
        expectations: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> ScorerResult:
        """Score correctness of the output."""
        if not expectations:
            return ScorerResult(
                score=0.5,
                rationale="No expected response provided for comparison"
            )
        
        # Extract expected response
        if isinstance(expectations, dict):
            expected = expectations.get("expected_response", str(expectations))
        else:
            expected = expectations
            
        # Extract input question
        if isinstance(inputs, dict):
            question = inputs.get("question", str(inputs))
        else:
            question = inputs
        
        prompt = f"""Evaluate if the following answer is factually correct compared to the expected answer.

Question: {question}

Expected Answer: {expected}

Actual Answer: {outputs}

Evaluate the correctness on a scale of 1-5 where:
- 5: Completely correct, matches expected answer
- 4: Mostly correct with minor differences
- 3: Partially correct
- 2: Somewhat incorrect
- 1: Completely incorrect

Respond in JSON format:
{{
    "score": <1-5>,
    "rationale": "<brief explanation>"
}}"""
        
        response = self._call_llm(prompt)
        
        try:
            # Parse JSON response
            result = json.loads(response)
            score = result.get("score", 3)
            rationale = result.get("rationale", "LLM evaluation completed")
            
            return ScorerResult(score=score, rationale=rationale)
        except json.JSONDecodeError:
            # Fallback: try to extract score from text
            for i in range(5, 0, -1):
                if str(i) in response:
                    return ScorerResult(
                        score=i,
                        rationale=f"Extracted from response: {response[:100]}"
                    )
            
            return ScorerResult(score=3, rationale="Could not parse LLM response")


class Guidelines(Scorer):
    """
    Evaluates if the output meets specified guidelines using LLM-as-a-Judge.
    
    This scorer checks if the response adheres to natural language guidelines.
    """
    
    def __init__(
        self,
        name: str = "guidelines",
        guidelines: str = "",
        model: Optional[str] = None
    ):
        """
        Initialize Guidelines scorer.
        
        Args:
            name: Name of the scorer
            guidelines: Natural language guidelines to check
            model: Model to use for evaluation
        """
        super().__init__(name=name, model=model)
        self.guidelines = guidelines
    
    def score(
        self,
        inputs: Union[str, Dict[str, Any]],
        outputs: str,
        expectations: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> ScorerResult:
        """Score adherence to guidelines."""
        prompt = f"""Evaluate if the following response meets these guidelines:

Guidelines: {self.guidelines}

Response: {outputs}

Does the response meet the guidelines? Respond in JSON format:
{{
    "score": <true or false>,
    "rationale": "<brief explanation>"
}}"""
        
        response = self._call_llm(prompt)
        
        try:
            result = json.loads(response)
            score = result.get("score", False)
            rationale = result.get("rationale", "Guideline evaluation completed")
            
            # Convert to boolean if needed
            if isinstance(score, str):
                score = score.lower() in ["true", "yes", "1"]
            
            return ScorerResult(score=bool(score), rationale=rationale)
        except json.JSONDecodeError:
            # Fallback: check for yes/no/true/false
            response_lower = response.lower()
            if "true" in response_lower or "yes" in response_lower:
                score = True
            elif "false" in response_lower or "no" in response_lower:
                score = False
            else:
                score = False
            
            return ScorerResult(
                score=score,
                rationale=f"Extracted from response: {response[:100]}"
            )


class Relevance(Scorer):
    """
    Evaluates if the output is relevant to the input using LLM-as-a-Judge.
    """
    
    def __init__(self, model: Optional[str] = None):
        super().__init__(name="relevance", model=model)
    
    def score(
        self,
        inputs: Union[str, Dict[str, Any]],
        outputs: str,
        expectations: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> ScorerResult:
        """Score relevance of output to input."""
        if isinstance(inputs, dict):
            question = inputs.get("question", str(inputs))
        else:
            question = inputs
        
        prompt = f"""Evaluate if the following answer is relevant to the question.

Question: {question}

Answer: {outputs}

Rate relevance on a scale of 1-5 where:
- 5: Highly relevant, directly answers the question
- 4: Mostly relevant with some extra information
- 3: Partially relevant
- 2: Somewhat relevant but off-topic
- 1: Not relevant at all

Respond in JSON format:
{{
    "score": <1-5>,
    "rationale": "<brief explanation>"
}}"""
        
        response = self._call_llm(prompt)
        
        try:
            result = json.loads(response)
            score = result.get("score", 3)
            rationale = result.get("rationale", "Relevance evaluation completed")
            
            return ScorerResult(score=score, rationale=rationale)
        except json.JSONDecodeError:
            for i in range(5, 0, -1):
                if str(i) in response:
                    return ScorerResult(
                        score=i,
                        rationale=f"Extracted from response: {response[:100]}"
                    )
            
            return ScorerResult(score=3, rationale="Could not parse LLM response")


class Conciseness(Scorer):
    """
    Evaluates if the output is concise (based on word count).
    """
    
    def __init__(self, max_words: int = 100, name: str = "conciseness"):
        super().__init__(name=name)
        self.max_words = max_words
    
    def score(
        self,
        inputs: Union[str, Dict[str, Any]],
        outputs: str,
        expectations: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> ScorerResult:
        """Score conciseness based on word count."""
        word_count = len(outputs.split())
        
        if word_count <= self.max_words:
            return ScorerResult(
                score=True,
                rationale=f"Output is concise ({word_count} words, max {self.max_words})"
            )
        else:
            return ScorerResult(
                score=False,
                rationale=f"Output is too verbose ({word_count} words, max {self.max_words})"
            )


class Faithfulness(Scorer):
    """
    Evaluates if the output is faithful to provided context (for RAG applications).
    """
    
    def __init__(self, model: Optional[str] = None):
        super().__init__(name="faithfulness", model=model)
    
    def score(
        self,
        inputs: Union[str, Dict[str, Any]],
        outputs: str,
        expectations: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> ScorerResult:
        """Score faithfulness to context."""
        context = kwargs.get("context", "")
        
        if not context:
            return ScorerResult(
                score=0.5,
                rationale="No context provided for faithfulness evaluation"
            )
        
        prompt = f"""Evaluate if the answer is faithful to the provided context. 
The answer should only contain information present in the context.

Context: {context}

Answer: {outputs}

Rate faithfulness on a scale of 1-5 where:
- 5: Completely faithful, no hallucinations
- 4: Mostly faithful with minor inference
- 3: Somewhat faithful
- 2: Contains some unfaithful information
- 1: Contains significant hallucinations

Respond in JSON format:
{{
    "score": <1-5>,
    "rationale": "<brief explanation>"
}}"""
        
        response = self._call_llm(prompt)
        
        try:
            result = json.loads(response)
            score = result.get("score", 3)
            rationale = result.get("rationale", "Faithfulness evaluation completed")
            
            return ScorerResult(score=score, rationale=rationale)
        except json.JSONDecodeError:
            for i in range(5, 0, -1):
                if str(i) in response:
                    return ScorerResult(
                        score=i,
                        rationale=f"Extracted from response: {response[:100]}"
                    )
            
            return ScorerResult(score=3, rationale="Could not parse LLM response")


def scorer(func: Callable) -> Scorer:
    """
    Decorator to create a custom scorer from a function.
    
    Example:
        @scorer
        def is_concise(outputs: str) -> bool:
            return len(outputs.split()) <= 5
    """
    class CustomScorer(Scorer):
        def __init__(self):
            super().__init__(name=func.__name__)
            self.func = func
        
        def score(
            self,
            inputs: Union[str, Dict[str, Any]],
            outputs: str,
            expectations: Optional[Union[str, Dict[str, Any]]] = None,
            **kwargs
        ) -> ScorerResult:
            """Call the custom scorer function."""
            try:
                # Try to call with different argument combinations
                import inspect
                sig = inspect.signature(self.func)
                params = list(sig.parameters.keys())
                
                # Determine what arguments to pass
                if len(params) == 1:
                    if params[0] == "outputs":
                        result = self.func(outputs)
                    elif params[0] == "inputs":
                        result = self.func(inputs)
                    else:
                        result = self.func(outputs)
                elif len(params) == 2:
                    result = self.func(inputs, outputs)
                elif len(params) >= 3:
                    result = self.func(inputs, outputs, expectations)
                else:
                    result = self.func()
                
                # Handle different return types
                if isinstance(result, ScorerResult):
                    return result
                elif isinstance(result, tuple) and len(result) == 2:
                    return ScorerResult(score=result[0], rationale=result[1])
                else:
                    return ScorerResult(
                        score=result,
                        rationale=f"Custom scorer '{func.__name__}' evaluation"
                    )
            except Exception as e:
                return ScorerResult(
                    score=False,
                    rationale=f"Error in custom scorer: {str(e)}"
                )
    
    return CustomScorer()

