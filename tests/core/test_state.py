import pytest
from pydantic import ValidationError

from src.llm_agent_orchestrator_cli.core.state import (
    OrchestrationState,
    TaskExecutionRecord,
)
from src.llm_agent_orchestrator_cli.llm.base_provider import TokenUsage


class TestStateModels:
    """Unit tests for the state management Pydantic models."""

    def test_task_execution_record_creation(self):
        """Tests successful creation of a TaskExecutionRecord."""
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        record = TaskExecutionRecord(
            task_name="test_task",
            agent_role="tester",
            output="test output",
            success=True,
            token_usage=usage,
            cost=0.001,
        )
        assert record.task_name == "test_task"
        assert record.cost == 0.001
        assert record.token_usage.total_tokens == 30

    def test_task_execution_record_missing_fields(self):
        """Tests that required fields raise validation errors."""
        with pytest.raises(ValidationError):
            TaskExecutionRecord(task_name="incomplete_task", agent_role="test")

    def test_orchestration_state_initialization(self):
        """Tests that OrchestrationState initializes correctly with no records."""
        state = OrchestrationState()
        assert state.execution_records == []
        assert state.total_tokens.total_tokens == 0
        assert state.total_cost == 0.0
        assert state.get_final_output() is None

    def test_orchestration_state_aggregation(self):
        """
        Tests the aggregation properties (total_tokens, total_cost) of OrchestrationState.
        """
        state = OrchestrationState()

        # Add first record
        state.add_record(TaskExecutionRecord(
            task_name="task_1",
            agent_role="agent_1",
            output="output 1",
            success=True,
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            cost=0.001
        ))

        assert state.total_tokens.prompt_tokens == 10
        assert state.total_tokens.completion_tokens == 20
        assert state.total_tokens.total_tokens == 30
        assert state.total_cost == pytest.approx(0.001)
        assert state.get_final_output() == "output 1"

        # Add second record
        state.add_record(TaskExecutionRecord(
            task_name="task_2",
            agent_role="agent_2",
            output="output 2",
            success=True,
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=200, total_tokens=300),
            cost=0.01
        ))

        assert state.total_tokens.prompt_tokens == 110  # 10 + 100
        assert state.total_tokens.completion_tokens == 220  # 20 + 200
        assert state.total_tokens.total_tokens == 330  # 30 + 300
        assert state.total_cost == pytest.approx(0.011)  # 0.001 + 0.01
        assert state.get_final_output() == "output 2"  # Final output is the last one

    def test_orchestration_state_with_failed_task(self):
        """
        Tests that aggregation works correctly even with failed tasks
        that might have no token usage or cost.
        """
        state = OrchestrationState()

        state.add_record(TaskExecutionRecord(
            task_name="task_1",
            agent_role="agent_1",
            output="output 1",
            success=True,
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            cost=0.001
        ))

        # Add a failed task record with no token/cost info
        state.add_record(TaskExecutionRecord(
            task_name="task_failed",
            agent_role="agent_failed",
            output="Error occurred",
            success=False,
            # token_usage and cost are optional, default to None
        ))

        assert len(state.execution_records) == 2
        assert state.total_tokens.total_tokens == 30  # Unchanged from task_1
        assert state.total_cost == pytest.approx(0.001)  # Unchanged from task_1
        # Final output should still be from the last successful task
        assert state.get_final_output() == "output 1"

    def test_get_final_output_no_successful_records(self):
        """Tests get_final_output returns None if all tasks failed."""
        state = OrchestrationState()
        state.add_record(TaskExecutionRecord(
            task_name="task_failed",
            agent_role="agent_failed",
            output="Error occurred",
            success=False,
        ))
        assert state.get_final_output() is None
