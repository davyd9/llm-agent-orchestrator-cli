import pytest
from unittest.mock import MagicMock, call

from llm_agent_orchestrator_cli.agents.models import Agent, Task
from llm_agent_orchestrator_cli.agents.crew import Crew
from llm_agent_orchestrator_cli.core.state_manager import StateManager
from llm_agent_orchestrator_cli.llm.base_provider import LLMResponse, ChatMessage
from llm_agent_orchestrator_cli.core.exceptions import OrchestrationError

@pytest.fixture
def mock_llm_provider():
    """Fixture for a mocked LLMProvider."""
    provider = MagicMock()
    provider.chat.side_effect = [
        LLMResponse(content="Analysis complete. Key insight: A."),
        LLMResponse(content="Draft written based on insight A.")
    ]
    return provider

@pytest.fixture
def mock_llm_provider_error():
    """Fixture for a mocked LLMProvider that raises an error."""
    provider = MagicMock()
    provider.chat.side_effect = RuntimeError("API limit reached")
    return provider

@pytest.fixture
def analyst_agent(mock_llm_provider):
    """Fixture for a researcher agent."""
    return Agent(
        role="Data Analyst",
        goal="Analyze data for key insights",
        backstory="You are a meticulous data analyst.",
        llm_provider=mock_llm_provider
    )

@pytest.fixture
def writer_agent(mock_llm_provider):
    """Fixture for a writer agent."""
    return Agent(
        role="Content Writer",
        goal="Write compelling content based on analysis",
        backstory="You are a creative content writer.",
        llm_provider=mock_llm_provider
    )

@pytest.fixture
def simple_tasks(analyst_agent, writer_agent):
    """Fixture for a list of tasks."""
    return [
        Task(
            description="Analyze the provided dataset.",
            expected_output="A summary of key insights.",
            agent=analyst_agent
        ),
        Task(
            description="Write a blog post based on the analysis.",
            expected_output="A 500-word blog post.",
            agent=writer_agent
        )
    ]


def test_crew_kickoff_success(analyst_agent, writer_agent, simple_tasks, mock_llm_provider):
    """Test successful execution of a sequence of tasks by the crew."""
    state_manager = StateManager()
    crew = Crew(agents=[analyst_agent, writer_agent], tasks=simple_tasks, state_manager=state_manager)

    final_result = crew.kickoff()

    assert final_result == "Draft written based on insight A."
    assert len(state_manager.history) == 2
    assert state_manager.history[0][1] == "Analysis complete. Key insight: A."
    assert state_manager.history[1][1] == "Draft written based on insight A."

    assert mock_llm_provider.chat.call_count == 2
    
    # Verify first call (analyst)
    first_call_args = mock_llm_provider.chat.call_args_list[0]
    first_prompt_content = first_call_args[0][0][0].content
    assert "You are Data Analyst." in first_prompt_content
    assert "Analyze the provided dataset." in first_prompt_content
    assert "No previous tasks have been completed." in first_prompt_content

    # Verify second call (writer)
    second_call_args = mock_llm_provider.chat.call_args_list[1]
    second_prompt_content = second_call_args[0][0][0].content
    assert "You are Content Writer." in second_prompt_content
    assert "Write a blog post based on the analysis." in second_prompt_content
    assert "Task Description: Analyze the provided dataset." in second_prompt_content
    assert "Task Output:\nAnalysis complete. Key insight: A." in second_prompt_content

def test_crew_initialization_with_invalid_agent(analyst_agent, writer_agent, simple_tasks):
    """Test that crew initialization fails if a task's agent is not in the crew's agent list."""
    # writer_agent is assigned to task 2, but not provided in the crew's agent list
    with pytest.raises(OrchestrationError, match="agent that is not part of this crew"):
        Crew(agents=[analyst_agent], tasks=simple_tasks, state_manager=StateManager())

def test_crew_kickoff_llm_error(analyst_agent, writer_agent, simple_tasks, mock_llm_provider_error):
    """Test that kickoff raises an OrchestrationError if the LLM provider fails."""
    # Both agents will use the error-producing mock
    analyst_agent.llm_provider = mock_llm_provider_error
    writer_agent.llm_provider = mock_llm_provider_error

    state_manager = StateManager()
    crew = Crew(agents=[analyst_agent, writer_agent], tasks=simple_tasks, state_manager=state_manager)

    with pytest.raises(OrchestrationError, match="Failed to execute task"):
        crew.kickoff()
    
    # Ensure the process stopped after the first failure
    assert len(state_manager.history) == 0
    mock_llm_provider_error.chat.assert_called_once()

def test_state_manager_context_building():
    """Test the context building logic of the StateManager."""
    sm = StateManager()
    assert "No previous tasks" in sm.get_full_context()

    sm.add_task_output("Task A", "Output A")
    context1 = sm.get_full_context()
    assert "Task Description: Task A" in context1
    assert "Task Output:\nOutput A" in context1

    sm.add_task_output("Task B", "Output B")
    context2 = sm.get_full_context()
    assert "Task Description: Task A" in context2
    assert "Task Output:\nOutput A" in context2
    assert "Task Description: Task B" in context2
    assert "Task Output:\nOutput B" in context2

    sm.clear()
    assert "No previous tasks" in sm.get_full_context()
