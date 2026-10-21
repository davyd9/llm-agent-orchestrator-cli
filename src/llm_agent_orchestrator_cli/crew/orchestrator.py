import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional

from .models import Agent, Task
from ..core.state import OrchestrationState, TaskExecutionRecord
from ..core.exceptions import OrchestrationError
from ..llm.base_provider import LLMProvider, ChatMessage

logger = logging.getLogger(__name__)


class Orchestrator:
    """Orchestrates the execution of tasks by a crew of agents."""

    def __init__(self, agents: List[Agent], tasks: List[Task], llm_provider: LLMProvider):
        if not agents:
            raise OrchestrationError("Orchestrator requires at least one agent.")
        if not tasks:
            raise OrchestrationError("Orchestrator requires at least one task.")

        self.agents = {agent.agent_id: agent for agent in agents}
        self.tasks = tasks
        self.llm_provider = llm_provider
        self.state = OrchestrationState()
        logger.info(f"Orchestrator initialized with {len(self.agents)} agents and {len(self.tasks)} tasks.")

    def run(self, initial_context: Optional[Dict] = None) -> str:
        """Runs the orchestration, executing all tasks in sequence."""
        logger.info(f"Starting orchestration run: {self.state.orchestration_id}")
        if initial_context:
            self.state.shared_context.update(initial_context)

        final_result = ""
        for i, task in enumerate(self.tasks):
            logger.info(f"Executing task {i+1}/{len(self.tasks)}: {task.description}")
            try:
                result = self._execute_task(task)
                final_result = result
                logger.info(f"Task '{task.task_id}' completed successfully.")
            except Exception as e:
                logger.error(f"Orchestration failed during task '{task.task_id}': {e}", exc_info=True)
                raise OrchestrationError(f"Failed to execute task '{task.task_id}'. Reason: {e}") from e

        self._log_summary()
        return final_result

    def _execute_task(self, task: Task) -> str:
        """Executes a single task."""
        agent = self.agents.get(task.agent_id)
        if not agent:
            raise OrchestrationError(f"Agent '{task.agent_id}' not found for task '{task.task_id}'.")

        record = TaskExecutionRecord(task_id=task.task_id, agent_id=agent.agent_id, input_data=task.model_dump())
        self.state.add_record(record)
        self.state.update_record(record.record_id, status="RUNNING")

        try:
            prompt = self._construct_prompt(task, agent)
            messages = [
                ChatMessage(role="system", content=agent.backstory),
                ChatMessage(role="user", content=prompt)
            ]

            llm_response = self.llm_provider.invoke(messages, model=agent.model, temperature=agent.temperature)

            self.state.update_record(
                record.record_id,
                status="COMPLETED",
                output_data=llm_response.content,
                token_usage=llm_response.token_usage,
                cost=llm_response.cost,
                end_time=datetime.now(timezone.utc)
            )
            return llm_response.content

        except Exception as e:
            error_message = f"Error executing task '{task.task_id}' with agent '{agent.agent_id}': {e}"
            logger.error(error_message, exc_info=True)
            self.state.update_record(
                record.record_id,
                status="FAILED",
                error=error_message,
                end_time=datetime.now(timezone.utc)
            )
            raise

    def _construct_prompt(self, task: Task, agent: Agent) -> str:
        """Constructs the full prompt for the LLM, including context."""
        context_str = "No previous context available."
        last_output = self.state.get_last_output()
        if last_output:
            context_str = f"Here is the result from the previous task, use it as context for your current work:\n---BEGIN CONTEXT---\n{last_output}\n---END CONTEXT---"

        prompt = f"""
        You are {agent.name}, a {agent.role}.
        Your goal is: {agent.goal}

        You are working on the following task:
        Task Description: {task.description}
        Expected Output: {task.expected_output}

        {context_str}

        Now, perform the task and provide only the final output as described.
        """
        return prompt

    def _log_summary(self) -> None:
        """Logs the final summary of the orchestration run."""
        logger.info("--- Orchestration Run Summary ---")
        logger.info(f"Run ID: {self.state.orchestration_id}")
        logger.info(f"Total Tasks Executed: {len(self.state.execution_records)}")
        logger.info(f"Total Input Tokens: {self.state.total_input_tokens}")
        logger.info(f"Total Output Tokens: {self.state.total_output_tokens}")
        logger.info(f"Total Tokens Consumed: {self.state.total_tokens}")
        logger.info(f"Estimated Total Cost: ${self.state.total_cost:.6f}")
        logger.info("---------------------------------")
