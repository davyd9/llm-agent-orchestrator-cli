import os
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from openai import APIConnectionError as OpenAIAPIConnectionError, AuthenticationError as OpenAIAuthError

from llm_agent_orchestrator_cli.llm.providers import LLMProviderFactory, OpenAIProvider, DeepSeekProvider
from llm_agent_orchestrator_cli.llm.base_provider import ChatMessage
from llm_agent_orchestrator_cli.core.exceptions import APIKeyNotFoundError, APIConnectionError, AuthenticationError

@pytest.fixture
def mock_env_openai():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_openai_key"}) as patched_env:
        yield patched_env

@pytest.fixture
def mock_env_deepseek():
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test_deepseek_key"}) as patched_env:
        yield patched_env

@pytest.fixture
def mock_openai_client():
    with patch("openai.AsyncOpenAI") as mock_client_class:
        mock_client_instance = AsyncMock()
        mock_client_class.return_value = mock_client_instance
        yield mock_client_instance

class TestLLMProviderFactory:
    def test_get_openai_provider(self, mock_env_openai, mock_openai_client):
        provider = LLMProviderFactory.get_provider("openai")
        assert isinstance(provider, OpenAIProvider)
        mock_openai_client.chat.completions.create.assert_not_called() # Ensure it's just instantiation

    def test_get_deepseek_provider(self, mock_env_deepseek, mock_openai_client):
        provider = LLMProviderFactory.get_provider("deepseek")
        assert isinstance(provider, DeepSeekProvider)

    def test_get_unknown_provider_raises_error(self):
        with pytest.raises(ValueError, match="Unknown provider: 'unknown'"):
            LLMProviderFactory.get_provider("unknown")

    def test_get_provider_is_case_insensitive(self, mock_env_openai, mock_openai_client):
        provider = LLMProviderFactory.get_provider("OpenAI")
        assert isinstance(provider, OpenAIProvider)

    def test_api_key_not_found_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(APIKeyNotFoundError, match="'OPENAI_API_KEY' not found"):
                LLMProviderFactory.get_provider("openai")

@pytest.mark.asyncio
class TestOpenAIProvider:
    @pytest.fixture
    def provider(self, mock_env_openai, mock_openai_client):
        return OpenAIProvider()

    async def test_chat_completion_success(self, provider, mock_openai_client):
        mock_response = MagicMock()
        mock_response.id = "chatcmpl-123"
        mock_response.model = "gpt-4-turbo"
        mock_response.usage.to_dict.return_value = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_response.choices[0].message.content = "Hello, world!"
        mock_response.choices[0].finish_reason = "stop"
        mock_openai_client.chat.completions.create.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello")]
        response = await provider.chat_completion(messages=messages, model="gpt-4-turbo")

        mock_openai_client.chat.completions.create.assert_awaited_once()
        args, kwargs = mock_openai_client.chat.completions.create.call_args
        assert kwargs["model"] == "gpt-4-turbo"
        assert kwargs["messages"][0]["content"] == "Hello"

        assert response.content == "Hello, world!"
        assert response.model == "gpt-4-turbo"
        assert response.usage["total_tokens"] == 30

    async def test_chat_completion_api_connection_error(self, provider, mock_openai_client):
        mock_openai_client.chat.completions.create.side_effect = OpenAIAPIConnectionError(request=MagicMock())
        
        with pytest.raises(APIConnectionError, match="Could not connect to OpenAI API."):
            await provider.chat_completion(messages=[ChatMessage(role="user", content="Hi")], model="gpt-4")

    async def test_chat_completion_auth_error(self, provider, mock_openai_client):
        mock_openai_client.chat.completions.create.side_effect = OpenAIAuthError("Invalid key", response=MagicMock(), body=None)

        with pytest.raises(AuthenticationError, match="Invalid OpenAI API key or organization."):
            await provider.chat_completion(messages=[ChatMessage(role="user", content="Hi")], model="gpt-4")

@pytest.mark.asyncio
class TestDeepSeekProvider:
    def test_client_instantiation(self, mock_env_deepseek):
        with patch("openai.AsyncOpenAI") as mock_client_class:
            provider = DeepSeekProvider()
            mock_client_class.assert_called_once_with(
                api_key="test_deepseek_key",
                base_url="https://api.deepseek.com/v1"
            )

    @pytest.fixture
    def provider(self, mock_env_deepseek, mock_openai_client):
        return DeepSeekProvider()

    async def test_chat_completion_success(self, provider, mock_openai_client):
        mock_response = MagicMock()
        mock_response.id = "chatcmpl-deep-123"
        mock_response.model = "deepseek-chat"
        mock_response.usage.to_dict.return_value = {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40}
        mock_response.choices[0].message.content = "Deep response"
        mock_response.choices[0].finish_reason = "stop"
        mock_openai_client.chat.completions.create.return_value = mock_response

        messages = [ChatMessage(role="user", content="Tell me something deep")]
        response = await provider.chat_completion(messages=messages, model="deepseek-chat")

        mock_openai_client.chat.completions.create.assert_awaited_once_with(
            messages=[{"role": "user", "content": "Tell me something deep"}],
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=1024
        )

        assert response.content == "Deep response"
        assert response.model == "deepseek-chat"
        assert response.usage["total_tokens"] == 40
