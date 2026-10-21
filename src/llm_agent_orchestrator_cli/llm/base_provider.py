import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """Data model for a single message in a chat conversation."""
    role: str = Field(..., description="The role of the message sender (e.g., 'user', 'assistant', 'system').")
    content: str = Field(..., description="The content of the message.")


class TokenUsage(BaseModel):
    """Data model for tracking token consumption."""
    input_tokens: int = Field(0, description="Number of tokens in the input/prompt.")
    output_tokens: int = Field(0, description="Number of tokens in the output/completion.")

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed."""
        return self.input_tokens + self.output_tokens


class ModelPricing(BaseModel):
    """Data model for defining the cost structure of a specific LLM model."""
    input_cost_per_million_tokens: float = Field(..., description="Cost in USD for 1 million input tokens.")
    output_cost_per_million_tokens: float = Field(..., description="Cost in USD for 1 million output tokens.")
    model_name: str = Field(..., description="The identifier for the model this pricing applies to.")


class LLMResponse(BaseModel):
    """Data model for a response from an LLM provider."""
    content: str = Field(..., description="The text content of the LLM's response.")
    model: str = Field(..., description="The model that generated the response.")
    token_usage: TokenUsage = Field(..., description="Token usage details for the API call.")
    cost: float = Field(..., description="Calculated cost for the API call based on token usage.")
    raw_response: Optional[Dict[str, Any]] = Field(None, description="The raw, unprocessed response from the provider's API.")


class LLMProvider(ABC):
    """Abstract Base Class for all LLM providers."""

    def __init__(self, api_key: str, model: str):
        """Initializes the provider with an API key and a default model.

        Args:
            api_key: The API key for the provider.
            model: The default model to use for invocations.
        """
        if not api_key:
            raise ValueError(f"{self.__class__.__name__} requires an API key.")
        self.api_key = api_key
        self.model = model
        self._initialize_client()

    @abstractmethod
    def _initialize_client(self) -> None:
        """Initializes the specific client for the LLM provider."""
        pass

    @abstractmethod
    def invoke(
        self, messages: List[ChatMessage], model: Optional[str] = None, temperature: float = 0.7, max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """Sends a request to the LLM and returns the response.

        Args:
            messages: A list of ChatMessage objects representing the conversation.
            model: The specific model to use for this invocation, overriding the default.
            temperature: The sampling temperature.
            max_tokens: The maximum number of tokens to generate.

        Returns:
            An LLMResponse object containing the model's output and metadata.
        """
        pass

    @abstractmethod
    def get_pricing(self, model_name: str) -> ModelPricing:
        """Retrieves the pricing information for a specific model.

        Args:
            model_name: The name of the model to get pricing for.

        Returns:
            A ModelPricing object with cost details.
        """
        pass

    def _calculate_cost(self, token_usage: TokenUsage, pricing: ModelPricing) -> float:
        """Calculates the cost of an LLM call based on token usage and pricing.

        Args:
            token_usage: The TokenUsage object with input/output token counts.
            pricing: The ModelPricing object for the specific model used.

        Returns:
            The calculated cost as a float.
        """
        input_cost = (token_usage.input_tokens / 1_000_000) * pricing.input_cost_per_million_tokens
        output_cost = (token_usage.output_tokens / 1_000_000) * pricing.output_cost_per_million_tokens
        total_cost = input_cost + output_cost
        logger.debug(
            f"Cost calculation for model '{pricing.model_name}': "
            f"Input: {token_usage.input_tokens} tokens * ${pricing.input_cost_per_million_tokens}/M = ${input_cost:.6f}, "
            f"Output: {token_usage.output_tokens} tokens * ${pricing.output_cost_per_million_tokens}/M = ${output_cost:.6f}, "
            f"Total: ${total_cost:.6f}"
        )
        return total_cost
