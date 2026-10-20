import unittest
from unittest.mock import patch, MagicMock, ANY

from llm_agent_orchestrator_cli.crew.models import Agent, Task
from llm_agent_orchestrator_cli.crew.orchestrator import Orchestrator, OrchestrationError
from llm_agent_orchestrator_cli.llm.base_provider import ChatResponse, Usage
from llm_agent_orchestrator_cli.core.exceptions import ProviderNotFoundError

class TestOrchestrator(unittest.TestCase):

    def setUp(self):
        """Set up common test resources."""
        self.mock_provider = MagicMock()
        self.mock_provider.chat.return_value = ChatResponse(
            content="Default mock response",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        )

        # Patch the factory to return our mock provider
        self.factory_patcher = patch(
            'llm_agent_orchestrator_cli.crew.orchestrator.LLMProviderFactory'
        )
        self.mock_factory_class = self.factory_patcher.start()
        self.mock_factory_instance = self.mock_factory_class.return_value
        self.mock_factory_instance.get_provider.return_value = self.mock_provider

        self.researcher_agent = Agent(
            role="Senior Research Analyst",
            goal="Uncover cutting-edge advancements in AI",
            backstory="You are an expert researcher with a knack for finding hidden gems.",
            llm_provider_name="openai",
            llm_config={"model": "gpt-4"}
        )
        self.writer_agent = Agent(
            role="Tech Content Strategist",
            goal="Craft compelling content about AI discoveries",
            backstory="You are a skilled writer who can simplify complex topics.",
            llm_provider_name="deepseek",
            llm_config={"model": "deepseek-coder"}
        )

    def tearDown(self):
        """Clean up after tests."""
        self.factory_patcher.stop()

    def test_orchestrator_initialization(self):
        """Test that the orchestrator initializes correctly."""
        task1 = Task(description="Research AI trends", agent=self.researcher_agent, expected_output="A summary report")
        orchestrator = Orchestrator(tasks=[task1])
        self.assertEqual(len(orchestrator.tasks), 1)
        self.assertIsNotNone(orchestrator.state)
        self.assertIn("orch_", orchestrator.orchestration_id)

    def test_orchestrator_init_with_empty_tasks(self):
        """Test that initialization fails with an empty task list."""
        with self.assertRaises(ValueError):
            Orchestrator(tasks=[])

    def test_run_single_task_with_initial_context(self):
        """Test running a single task with initial context."""
        self.mock_provider.chat.return_value = ChatResponse(
            content="AI trends report.",
            usage=Usage(prompt_tokens=50, completion_tokens=10, total_tokens=60)
        )
        task1 = Task(
            description="Research AI trends based on a topic",
            agent=self.researcher_agent,
            expected_output="A summary report",
            context={"topic": "Quantum Computing"}
        )
        orchestrator = Orchestrator(tasks=[task1])
        result = orchestrator.run()

        self.assertEqual(result, "AI trends report.")
        self.mock_factory_instance.get_provider.assert_called_once_with("openai", model="gpt-4")
        self.mock_provider.chat.assert_called_once()
        
        # Check if the prompt contains the initial context
        call_args = self.mock_provider.chat.call_args[0][0]
        prompt_content = call_args[0].content
        self.assertIn("topic: Quantum Computing", prompt_content)
        self.assertIn("You are Senior Research Analyst", prompt_content)

        # Check state management
        self.assertEqual(len(orchestrator.state.history), 1)
        record = orchestrator.state.history[0]
        self.assertEqual(record.output, "AI trends report.")
        self.assertEqual(record.token_usage['total_tokens'], 60)

    def test_run_multi_task_sequential_context_passing(self):
        """Test that context is passed between two sequential tasks."""
        # Mock different responses for each call
        self.mock_provider.chat.side_effect = [
            ChatResponse(content="Research result: AI is growing.", usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)),
            ChatResponse(content="Blog post about AI growth.", usage=Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30))
        ]

        task1 = Task(description="Research AI trends", agent=self.researcher_agent, expected_output="A summary report")
        task2 = Task(description="Write a blog post based on the research", agent=self.writer_agent, expected_output="A blog post")
        
        orchestrator = Orchestrator(tasks=[task1, task2])
        final_result = orchestrator.run()

        self.assertEqual(final_result, "Blog post about AI growth.")
        self.assertEqual(self.mock_provider.chat.call_count, 2)

        # Verify first call (researcher)
        first_call_args = self.mock_provider.chat.call_args_list[0][0][0]
        first_prompt = first_call_args[0].content
        self.assertIn("You are Senior Research Analyst", first_prompt)
        self.assertIn("No initial context provided", first_prompt)

        # Verify second call (writer)
        second_call_args = self.mock_provider.chat.call_args_list[1][0][0]
        second_prompt = second_call_args[0].content
        self.assertIn("You are Tech Content Strategist", second_prompt)
        self.assertIn("Research result: AI is growing.", second_prompt) # Crucial check for context passing

        # Check state
        self.assertEqual(len(orchestrator.state.history), 2)
        self.assertEqual(orchestrator.state.history[0].output, "Research result: AI is growing.")
        self.assertEqual(orchestrator.state.history[1].output, "Blog post about AI growth.")

    def test_run_with_provider_error(self):
        """Test that OrchestrationError is raised on provider failure."""
        self.mock_factory_instance.get_provider.side_effect = ProviderNotFoundError("Provider 'fail_provider' not found.")
        
        task1 = Task(description="A failing task", agent=self.researcher_agent, expected_output="wont happen")
        orchestrator = Orchestrator(tasks=[task1])

        with self.assertRaises(OrchestrationError) as cm:
            orchestrator.run()
        
        self.assertIn("Provider 'fail_provider' not found", str(cm.exception))
        
        # Check that the failed task is recorded in the state
        self.assertEqual(len(orchestrator.state.history), 1)
        record = orchestrator.state.history[0]
        self.assertIsNone(record.output)
        self.assertIsNotNone(record.error)
        self.assertIn("Provider 'fail_provider' not found", record.error)

if __name__ == '__main__':
    unittest.main()
