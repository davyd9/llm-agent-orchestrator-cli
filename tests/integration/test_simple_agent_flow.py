import pytest
import logging

from src.llm_agent_orchestrator_cli.crew.models import Agent, Task
from src.llm_agent_orchestrator_cli.crew.orchestrator import Orchestrator
from src.llm_agent_orchestrator_cli.core.state import OrchestrationState

# Ensure logs are captured during tests
logging.basicConfig(level=logging.INFO)


class TestSimpleAgentFlow:
    """
    Integration tests for a simple, two-step agent workflow.

    This test class verifies that the Orchestrator can correctly execute a sequence
    of tasks, pass context between them, and accurately track state, including
    token usage and cost, using a mocked LLM provider.
    """

    @pytest.fixture
    def research_and_write_crew(self, mock_llm_provider):
        """
        Sets up a crew with a researcher and a writer agent, along with their tasks.
        """
        researcher = Agent(
            role="Senior Tech Researcher",
            goal="Find the key features of a given technology.",
            backstory="You are an expert at distilling complex technical information into concise points.",
            llm_provider=mock_llm_provider,
        )

        writer = Agent(
            role="Marketing Content Writer",
            goal="Write a compelling marketing post based on research findings.",
            backstory="You craft engaging social media posts that drive interest in new tech.",
            llm_provider=mock_llm_provider,
        )

        research_task = Task(
            name="Research CrewAI Features",
            description="Research the key features of CrewAI and provide a summary.",
            agent=researcher,
        )

        write_task = Task(
            name="Write Marketing Post",
            description="Write a short marketing post for Twitter about CrewAI, using the research findings. The post should include relevant hashtags.",
            agent=writer,
            context_tasks=[research_task],  # This task depends on the research_task
        )

        return {
            "agents": [researcher, writer],
            "tasks": [research_task, write_task]
        }

    def test_sequential_flow_execution_and_state_tracking(self, research_and_write_crew, mock_llm_provider):
        """
        Tests a sequential agent flow, verifying context passing, final output,
        and correct aggregation of token and cost data.
        """
        # Arrange
        agents = research_and_write_crew["agents"]
        tasks = research_and_write_crew["tasks"]

        orchestrator = Orchestrator(agents=agents, tasks=tasks, verbose=False)

        # Act
        final_result = orchestrator.run()
        final_state: OrchestrationState = orchestrator.state

        # Assert Final Output
        expected_final_output = "🚀 Supercharge your workflows with CrewAI! Orchestrate autonomous AI agents for complex tasks. Unlock seamless collaboration and powerful tool integration. #AI #Automation #CrewAI"
        assert final_result == expected_final_output
        assert final_state.get_final_output() == expected_final_output

        # Assert State and Task Execution
        assert len(final_state.execution_records) == 2

        # Verify Research Task
        research_record = final_state.execution_records[0]
        assert research_record.task_name == "Research CrewAI Features"
        assert research_record.success is True
        expected_research_output = "CrewAI facilitates orchestrating role-playing, autonomous AI agents. Key features include flexible task management, agent collaboration, and tool integration."
        assert research_record.output == expected_research_output
        assert research_record.token_usage.total_tokens == 50
        assert research_record.cost == 0.00015

        # Verify Write Task
        write_record = final_state.execution_records[1]
        assert write_record.task_name == "Write Marketing Post"
        assert write_record.success is True
        assert write_record.output == expected_final_output
        assert write_record.token_usage.total_tokens == 100
        assert write_record.cost == 0.00025

        # Assert that the context from the first task was used in the second
        # We can check the calls made to the mock provider
        second_call_args = mock_llm_provider.invoke.call_args_list[1]
        prompt_for_writer = second_call_args.args[0]
        assert expected_research_output in prompt_for_writer

        # Assert Total Cost and Tokens
        assert final_state.total_tokens.total_tokens == 150  # 50 + 100
        assert final_state.total_tokens.prompt_tokens == 70  # 15 + 55
        assert final_state.total_tokens.completion_tokens == 80  # 35 + 45
        assert final_state.total_cost == pytest.approx(0.00040)  # 0.00015 + 0.00025
