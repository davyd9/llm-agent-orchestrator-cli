import logging
from typing import List, Any, Dict

import openai
from openai import AsyncOpenAI

from llm_agent_orchestrator_cli.core.exceptions import (
    APIConnectionError,
    AuthenticationError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
)
from llm_agent_orchestrator_cli.llm.base_provider import (
    LLMProvider,
    ChatMessage,
    ChatCompletionResponse,
)

logger = logging.getLogger(__name__)

class OpenAIProvider(LLMProvider):
    """LLM Provider implementation for OpenAI APIs."""

    def __init__(self):
        super().__init__(api_key_env_var="OPENAI_API_KEY", api_base_env_var="OPENAI_API_BASE")
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        """Sends a chat completion request to the OpenAI API."""
        logger.debug(f"Sending request to OpenAI model {model} with {len(messages)} messages.")
        try:
            chat_messages = [msg.model_dump() for msg in messages]
            response = await self.client.chat.completions.create(
                messages=chat_messages, # type: ignore
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            choice = response.choices[0]
            logger.info(f"Received response from OpenAI model {model}. Finish reason: {choice.finish_reason}")
            return ChatCompletionResponse(
                id=response.id,
                model=response.model,
                content=choice.message.content or "",
                usage=dict(response.usage) if response.usage else {},
                finish_reason=choice.finish_reason,
            )
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI authentication error: {e}")
            raise AuthenticationError("Invalid OpenAI API key or organization.", e)
        except openai.RateLimitError as e:
            logger.warning(f"OpenAI rate limit exceeded: {e}")
            raise RateLimitError("OpenAI API rate limit exceeded.", e)
        except openai.APIConnectionError as e:
            logger.error(f"OpenAI API connection error: {e}")
            raise APIConnectionError("Could not connect to OpenAI API.", e)
        except openai.BadRequestError as e:
            logger.error(f"OpenAI invalid request error: {e}")
            raise InvalidRequestError(f"Invalid request to OpenAI API: {e.body}", e)
        except Exception as e:
            logger.exception("An unexpected error occurred with the OpenAI provider.")
            raise ProviderError("An unexpected error occurred.", e)

class DeepSeekProvider(LLMProvider):
    """LLM Provider implementation for DeepSeek APIs (OpenAI compatible)."""

    def __init__(self):
        super().__init__(api_key_env_var="DEEPSEEK_API_KEY")
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1"
        )

    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        """Sends a chat completion request to the DeepSeek API."""
        logger.debug(f"Sending request to DeepSeek model {model} with {len(messages)} messages.")
        # The implementation is identical to OpenAI's due to API compatibility
        # We use a helper method to avoid code duplication
        return await self._execute_openai_compatible_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def _execute_openai_compatible_completion(
        self, **kwargs: Any
    ) -> ChatCompletionResponse:
        try:
            chat_messages = [msg.model_dump() for msg in kwargs.pop("messages")]
            response = await self.client.chat.completions.create(
                messages=chat_messages, # type: ignore
                **kwargs,
            )
            choice = response.choices[0]
            logger.info(f"Received response from {self.provider_name} model {kwargs.get('model')}. Finish reason: {choice.finish_reason}")
            return ChatCompletionResponse(
                id=response.id,
                model=response.model,
                content=choice.message.content or "",
                usage=dict(response.usage) if response.usage else {},
                finish_reason=choice.finish_reason,
            )
        except openai.AuthenticationError as e:
            logger.error(f"{self.provider_name} authentication error: {e}")
            raise AuthenticationError(f"Invalid {self.provider_name} API key.", e)
        except openai.RateLimitError as e:
            logger.warning(f"{self.provider_name} rate limit exceeded: {e}")
            raise RateLimitError(f"{self.provider_name} API rate limit exceeded.", e)
        except openai.APIConnectionError as e:
            logger.error(f"{self.provider_name} API connection error: {e}")
            raise APIConnectionError(f"Could not connect to {self.provider_name} API.", e)
        except openai.BadRequestError as e:
            logger.error(f"{self.provider_name} invalid request error: {e}")
            raise InvalidRequestError(f"Invalid request to {self.provider_name} API: {e.body}", e)
        except Exception as e:
            logger.exception(f"An unexpected error occurred with the {self.provider_name} provider.")
            raise ProviderError("An unexpected error occurred.", e)

class LLMProviderFactory:
    """Factory for creating LLM provider instances."""

    _providers: Dict[str, type[LLMProvider]] = {
        "openai": OpenAIProvider,
        "deepseek": DeepSeekProvider,
    }

    @staticmethod
    def get_provider(name: str) -> LLMProvider:
        """
        Gets an instance of a registered LLM provider.

        Args:
            name: The name of the provider (e.g., 'openai', 'deepseek').

        Returns:
            An instance of the requested LLMProvider.

        Raises:
            ValueError: If the provider name is not registered.
        """
        provider_class = LLMProviderFactory._providers.get(name.lower())
        if not provider_class:
            logger.error(f"Attempted to get an unknown provider: '{name}'")
            raise ValueError(f"Unknown provider: '{name}'. Available: {list(LLMProviderFactory._providers.keys())}")
        
        logger.info(f"Creating instance of provider: '{name}'")
        return provider_class()
