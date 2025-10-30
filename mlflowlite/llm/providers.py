"""Concrete LLM provider implementations."""

import os
from typing import List, Dict, Any, Optional
import litellm
from mlflowlite.llm.base import LLMProvider, LLMResponse, Message, MessageRole


class LiteLLMProvider(LLMProvider):
    """Base provider using LiteLLM for unified interface."""
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.api_key = api_key
        
        # Set API key if provided
        if api_key:
            self._set_api_key(api_key)
    
    def _set_api_key(self, api_key: str):
        """Set API key for the provider."""
        # Override in subclasses if needed
        pass
    
    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert Message objects to LiteLLM format."""
        formatted = []
        for msg in messages:
            formatted_msg = {
                "role": msg.role.value,
                "content": msg.content,
            }
            if msg.name:
                formatted_msg["name"] = msg.name
            if msg.tool_calls:
                formatted_msg["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                formatted_msg["tool_call_id"] = msg.tool_call_id
            formatted.append(formatted_msg)
        return formatted
    
    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a completion using LiteLLM."""
        formatted_messages = self._format_messages(messages)
        
        completion_kwargs = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.temperature),
        }
        
        if self.max_tokens:
            completion_kwargs["max_tokens"] = self.max_tokens
        
        if tools:
            completion_kwargs["tools"] = tools
            completion_kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")
        
        # Add any additional kwargs
        completion_kwargs.update(self.kwargs)
        
        try:
            response = litellm.completion(**completion_kwargs)
            
            message = response.choices[0].message
            
            return LLMResponse(
                content=message.content or "",
                role=MessageRole.ASSISTANT,
                tool_calls=message.tool_calls if hasattr(message, 'tool_calls') else None,
                finish_reason=response.choices[0].finish_reason,
                usage=dict(response.usage) if hasattr(response, 'usage') else None,
                model=response.model,
            )
        except Exception as e:
            raise RuntimeError(f"LLM completion failed: {str(e)}") from e
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using LiteLLM."""
        try:
            return litellm.token_counter(model=self.model, text=text)
        except Exception:
            # Fallback to rough estimation
            return len(text) // 4
    
    @property
    def provider_name(self) -> str:
        return "litellm"


class OpenAIProvider(LiteLLMProvider):
    """OpenAI provider."""
    
    def _set_api_key(self, api_key: str):
        os.environ["OPENAI_API_KEY"] = api_key
    
    @property
    def provider_name(self) -> str:
        return "openai"


class AnthropicProvider(LiteLLMProvider):
    """Anthropic provider."""
    
    def _set_api_key(self, api_key: str):
        os.environ["ANTHROPIC_API_KEY"] = api_key
    
    @property
    def provider_name(self) -> str:
        return "anthropic"


class GeminiProvider(LiteLLMProvider):
    """Google Gemini provider."""
    
    def _set_api_key(self, api_key: str):
        os.environ["GEMINI_API_KEY"] = api_key
    
    @property
    def provider_name(self) -> str:
        return "gemini"


class MistralProvider(LiteLLMProvider):
    """Mistral provider."""
    
    def _set_api_key(self, api_key: str):
        os.environ["MISTRAL_API_KEY"] = api_key
    
    @property
    def provider_name(self) -> str:
        return "mistral"


def get_provider(
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> LLMProvider:
    """Get the appropriate provider based on model name."""
    model_lower = model.lower()
    
    # Normalize model names for LiteLLM compatibility
    normalized_model = model
    
    # Map common model names to LiteLLM format
    if any(prefix in model_lower for prefix in ["gpt", "o1", "chatgpt"]):
        # OpenAI models are fine as-is
        return OpenAIProvider(normalized_model, temperature, max_tokens, api_key, **kwargs)
    elif any(prefix in model_lower for prefix in ["claude", "anthropic"]):
        # Ensure Claude models have proper prefix for LiteLLM
        if not model.startswith("claude-"):
            normalized_model = model
        # Map common shortcuts to full model names
        if "claude-3-5-sonnet" in model_lower and "20" not in model:
            normalized_model = "claude-3-5-sonnet-20240620"  # Latest stable version
        elif "claude-3-opus" in model_lower and "20" not in model:
            normalized_model = "claude-3-opus-20240229"
        elif "claude-3-sonnet" in model_lower and "20" not in model:
            normalized_model = "claude-3-sonnet-20240229"
        elif "claude-3-haiku" in model_lower and "20" not in model:
            normalized_model = "claude-3-haiku-20240307"
        return AnthropicProvider(normalized_model, temperature, max_tokens, api_key, **kwargs)
    elif any(prefix in model_lower for prefix in ["gemini", "palm"]):
        return GeminiProvider(normalized_model, temperature, max_tokens, api_key, **kwargs)
    elif "mistral" in model_lower:
        return MistralProvider(normalized_model, temperature, max_tokens, api_key, **kwargs)
    else:
        # Default to LiteLLM for other models (local, etc.)
        return LiteLLMProvider(normalized_model, temperature, max_tokens, api_key, **kwargs)

