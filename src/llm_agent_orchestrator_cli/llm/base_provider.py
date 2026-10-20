import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Literal

from pydantic import BaseModel, Field

from llm_agent_orchestrator_cli.core.exceptions import APIKeyNotFoundError

# Configure a logger for the LLM module
logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    """Represents a single message in a chat conversation."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str

class ChatCompletionResponse(BaseModel):
    """Represents the response from a chat completion request."""
    id: str
    model: str
    content: str
    usage: Dict[str, int] = Field(..., description="Token usage statistics for the completion.")
    finish_reason: Optional[str] = None

class LLMProvider(ABC):
    """Abstract Base Class for all Large Language Model providers."""

    def __init__(self, api_key_env_var: str, api_base_env_var: Optional[str] = None):
        """
        Initializes the provider, loading the API key from environment variables.

        Args:
            api_key_env_var: The name of the environment variable for the API key.
            api_base_env_var: The optional name of the environment variable for the API base URL.

        Raises:
            APIKeyNotFoundError: If the API key environment variable is not set.
        """
        self.api_key = os.getenv(api_key_env_var)
        if not self.api_key:
            logger.error(f"API key environment variable '{api_key_env_var}' not found.")
            raise APIKeyNotFoundError(api_key_env_var)

        self.base_url = os.getenv(api_base_env_var) if api_base_env_var else None
        logger.info(f"Initialized {self.__class__.__name__} provider.")

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any
    ) -> ChatCompletionResponse:
        """
        Asynchronously sends a chat completion request to the LLM provider.

        Args:
            messages: A list of ChatMessage objects representing the conversation history.
            model: The identifier of the model to use for the completion.
            temperature: The sampling temperature.
            max_tokens: The maximum number of tokens to generate.
            **kwargs: Additional provider-specific parameters.

        Returns:
            A ChatCompletionResponse object with the result.
        """
        pass

    @property
    def provider_name(self) -> str:
        """Returns the name of the provider class."""
        return self.__class__.__name__
