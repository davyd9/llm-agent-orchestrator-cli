import pytest
from unittest.mock import MagicMock, create_autospec

from src.llm_agent_orchestrator_cli.llm.base_provider import (
    LLMProvider,
    LLMResponse,
    TokenUsage,
)


@pytest.fixture
def mock_llm_provider() -> MagicMock:
    """
    Provides a mocked LLM provider that returns predictable responses
    for integration testing agent flows.
    """
    # Use create_autospec to ensure the mock has the same interface as the real class
    mock_provider = create_autospec(LLMProvider, instance=True)

    def invoke_side_effect(prompt: str, **kwargs) -> LLMResponse:
        """Simulates LLM responses based on prompt content."""
        prompt_lower = prompt.lower()
        if "research the key features of crewai" in prompt_lower:
            return LLMResponse(
                content="CrewAI facilitates orchestrating role-playing, autonomous AI agents. Key features include flexible task management, agent collaboration, and tool integration.",
                model="test-model-v1",
                token_usage=TokenUsage(prompt_tokens=15, completion_tokens=35, total_tokens=50),
                cost=0.00015,
            )
        elif "write a short marketing post" in prompt_lower and "crewai" in prompt_lower:
            return LLMResponse(
                content="🚀 Supercharge your workflows with CrewAI! Orchestrate autonomous AI agents for complex tasks. Unlock seamless collaboration and powerful tool integration. #AI #Automation #CrewAI",
                model="test-model-v1",
                token_usage=TokenUsage(prompt_tokens=55, completion_tokens=45, total_tokens=100),
                cost=0.00025,
            )
        else:
            return LLMResponse(
                content="Default mock response for unknown prompt.",
                model="test-model-v1",
                token_usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
                cost=0.00005,
            )

    mock_provider.invoke.side_effect = invoke_side_effect
    mock_provider.provider_name = "mock_provider"

    return mock_provider
