import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from ..core.exceptions import OrchestrationError
from ..core.state import OrchestrationState, TaskExecutionRecord
from ..llm.base_provider import ChatMessage
from ..llm.providers import LLMProviderFactory
from .models import Task

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Manages a series of tasks, orchestrating agents to execute them sequentially.
    It maintains state, passes context between tasks, and handles the overall workflow.
    """

    def __init__(self, tasks: List[Task], orchestration_id: Optional[str] = None):
        if not tasks:
            raise ValueError("The list of tasks cannot be empty.")
        self.tasks = tasks
        self.orchestration_id = orchestration_id or f"orch_{uuid.uuid4()}"
        self.state = OrchestrationState(self.orchestration_id)
        self.provider_factory = LLMProviderFactory()
        logger.info(f"Orchestrator initialized for ID: {self.orchestration_id} with {len(tasks)} tasks.")

    def _construct_prompt(self, task: Task, context: str) -> str:
        """Constructs the full prompt for the agent's LLM call."""
        agent = task.agent
        prompt = f"""
You are {agent.role}.
Your Goal: {agent.goal}
Your Backstory: {agent.backstory}

You are working on the following task:
- Task Description: {task.description}
- Expected Output: {task.expected_output}

Here is the context from the previous step:
---
{context}
---

Based on the context and your expertise, perform the task.
Provide only the final output as described in 'Expected Output', without any extra commentary, introductions, or conversational text.
"""
        return prompt.strip()

    def run(self) -> str:
        """
        Executes the sequence of tasks, passing the output of each task as context to the next.

        Returns:
            The final output from the last task in the sequence.

        Raises:
            OrchestrationError: If any task fails during execution.
        """
        logger.info(f"Starting orchestration run for ID: {self.orchestration_id}")
        current_context: Optional[str] = None

        for i, task in enumerate(self.tasks):
            logger.info(f"Starting task {i+1}/{len(self.tasks)}: '{task.description[:50]}...'")
            
            # The first task uses its own context, subsequent tasks use the previous output
            if i == 0:
                initial_input = "\n".join(f"{k}: {v}" for k, v in task.context.items())
                input_context = initial_input if initial_input else "No initial context provided. Start from scratch."
            else:
                input_context = current_context
            
            if input_context is None:
                # This should not happen after the first task
                msg = f"Critical error: Context is None for task {i+1}."
                logger.error(msg)
                raise OrchestrationError(msg)

            record = TaskExecutionRecord(
                task_id=task.id,
                task_description=task.description,
                agent_role=task.agent.role,
                input_context=input_context
            )

            try:
                provider = self.provider_factory.get_provider(
                    task.agent.llm_provider_name,
                    **task.agent.llm_config
                )
                
                prompt = self._construct_prompt(task, input_context)
                messages = [ChatMessage(role="user", content=prompt)]
                
                response = provider.chat(messages)
                
                current_context = response.content
                record.output = current_context
                record.token_usage = response.usage.dict()

            except Exception as e:
                logger.error(f"Error executing task {task.id} with agent {task.agent.role}: {e}", exc_info=True)
                record.error = str(e)
                self.state.add_task_record(record)
                raise OrchestrationError(f"Task failed for agent {task.agent.role}: {e}") from e
            finally:
                record.end_time = datetime.now(timezone.utc)
                self.state.add_task_record(record)

            logger.info(f"Finished task {i+1}/{len(self.tasks)}. Output length: {len(current_context or '')}")

        if current_context is None:
            final_msg = "Orchestration completed, but the final output is empty."
            logger.warning(final_msg)
            return final_msg
        
        logger.info(f"Orchestration run {self.orchestration_id} completed successfully.")
        return current_context
