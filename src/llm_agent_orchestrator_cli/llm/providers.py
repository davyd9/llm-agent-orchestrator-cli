import logging
import os
from typing import List, Optional, Dict

from openai import OpenAI, APIError

from .base_provider import LLMProvider, ChatMessage, LLMResponse, TokenUsage, ModelPricing
from ..core.exceptions import LLMProviderError

logger = logging.getLogger(__name__)

# Pricing data (USD per 1 million tokens)
# This should ideally be loaded from a configuration file or a service.
OPENAI_PRICING: Dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(model_name="gpt-4o", input_cost_per_million_tokens=5.00, output_cost_per_million_tokens=15.00),
    "gpt-4-turbo": ModelPricing(model_name="gpt-4-turbo", input_cost_per_million_tokens=10.00, output_cost_per_million_tokens=30.00),
    "gpt-3.5-turbo": ModelPricing(model_name="gpt-3.5-turbo", input_cost_per_million_tokens=0.50, output_cost_per_million_tokens=1.50),
    "default": ModelPricing(model_name="default", input_cost_per_million_tokens=10.00, output_cost_per_million_tokens=30.00), # Fallback
}

DEEPSEEK_PRICING: Dict[str, ModelPricing] = {
    "deepseek-chat": ModelPricing(model_name="deepseek-chat", input_cost_per_million_tokens=0.14, output_cost_per_million_tokens=0.28),
    "deepseek-coder": ModelPricing(model_name="deepseek-coder", input_cost_per_million_tokens=0.14, output_cost_per_million_tokens=0.28),
    "default": ModelPricing(model_name="default", input_cost_per_million_tokens=0.14, output_cost_per_million_tokens=0.28), # Fallback
}


class OpenAIProvider(LLMProvider):
    """Concrete implementation for the OpenAI API."""

    def _initialize_client(self) -> None:
        self.client = OpenAI(api_key=self.api_key)

    def get_pricing(self, model_name: str) -> ModelPricing:
        return OPENAI_PRICING.get(model_name, OPENAI_PRICING["default"])

    def invoke(self, messages: List[ChatMessage], model: Optional[str] = None, temperature: float = 0.7, max_tokens: Optional[int] = None) -> LLMResponse:
        target_model = model or self.model
        logger.info(f"Invoking OpenAI model: {target_model} with {len(messages)} messages.")

        try:
            api_messages = [msg.model_dump() for msg in messages]
            response = self.client.chat.completions.create(
                model=target_model,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content
            if not content:
                raise LLMProviderError("OpenAI API returned an empty response content.")

            usage = response.usage
            token_usage = TokenUsage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            )

            pricing = self.get_pricing(target_model)
            cost = self._calculate_cost(token_usage, pricing)

            return LLMResponse(
                content=content,
                model=target_model,
                token_usage=token_usage,
                cost=cost,
                raw_response=response.model_dump()
            )

        except APIError as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            raise LLMProviderError(f"OpenAI API request failed: {e.message}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred while calling OpenAI: {e}", exc_info=True)
            raise LLMProviderError(f"An unexpected error occurred: {e}") from e


class DeepSeekProvider(LLMProvider):
    """Concrete implementation for the DeepSeek API (OpenAI-compatible)."""

    def _initialize_client(self) -> None:
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1"
        )

    def get_pricing(self, model_name: str) -> ModelPricing:
        return DEEPSEEK_PRICING.get(model_name, DEEPSEEK_PRICING["default"])

    def invoke(self, messages: List[ChatMessage], model: Optional[str] = None, temperature: float = 0.7, max_tokens: Optional[int] = None) -> LLMResponse:
        target_model = model or self.model
        logger.info(f"Invoking DeepSeek model: {target_model} with {len(messages)} messages.")

        try:
            api_messages = [msg.model_dump() for msg in messages]
            response = self.client.chat.completions.create(
                model=target_model,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content
            if not content:
                raise LLMProviderError("DeepSeek API returned an empty response content.")

            usage = response.usage
            token_usage = TokenUsage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            )

            pricing = self.get_pricing(target_model)
            cost = self._calculate_cost(token_usage, pricing)

            return LLMResponse(
                content=content,
                model=target_model,
                token_usage=token_usage,
                cost=cost,
                raw_response=response.model_dump()
            )

        except APIError as e:
            logger.error(f"DeepSeek API error: {e}", exc_info=True)
            raise LLMProviderError(f"DeepSeek API request failed: {e.message}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred while calling DeepSeek: {e}", exc_info=True)
            raise LLMProviderError(f"An unexpected error occurred: {e}") from e


class LLMProviderFactory:
    """Factory for creating LLM provider instances."""
    _providers = {
        "openai": OpenAIProvider,
        "deepseek": DeepSeekProvider,
    }

    @staticmethod
    def create(provider_name: str, model: str) -> LLMProvider:
        """Creates an LLM provider instance.

        Args:
            provider_name: The name of the provider (e.g., 'openai', 'deepseek').
            model: The default model to be used by the provider.

        Returns:
            An instance of a class that inherits from LLMProvider.

        Raises:
            ValueError: If the provider name or required API key is invalid.
        """
        provider_name = provider_name.lower()
        if provider_name not in LLMProviderFactory._providers:
            raise ValueError(f"Unknown provider: {provider_name}. Available: {list(LLMProviderFactory._providers.keys())}")

        api_key_env_var = f"{provider_name.upper()}_API_KEY"
        api_key = os.getenv(api_key_env_var)

        if not api_key:
            raise ValueError(f"API key not found. Please set the {api_key_env_var} environment variable.")

        provider_class = LLMProviderFactory._providers[provider_name]
        logger.info(f"Creating '{provider_name}' provider for model '{model}'.")
        return provider_class(api_key=api_key, model=model)
