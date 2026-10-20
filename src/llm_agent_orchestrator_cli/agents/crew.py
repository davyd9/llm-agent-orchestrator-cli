import logging
from typing import List, Dict

from llm_agent_orchestrator_cli.agents.models import Agent, Task
from llm_agent_orchestrator_cli.core.state_manager import StateManager
from llm_agent_orchestrator_cli.llm.base_provider import ChatMessage
from llm_agent_orchestrator_cli.core.exceptions import OrchestrationError

logger = logging.getLogger(__name__)

class Crew:
    """Orchestrates a collection of agents to perform a series of tasks sequentially."""

    def __init__(
        self, 
        agents: List[Agent], 
        tasks: List[Task],
        state_manager: StateManager,
        verbose: bool = False
    ):
        """Initializes the Crew.

        Args:
            agents: A list of Agent objects that are part of this crew.
            tasks: A list of Task objects to be executed.
            state_manager: The state manager instance for context and history.
            verbose: If True, enables detailed logging of the orchestration process.
        """
        self.agents = agents
        self.tasks = tasks
        self.state_manager = state_manager
        self.verbose = verbose
        self._validate_crew()
        logger.info(f"Crew initialized with {len(self.agents)} agents and {len(self.tasks)} tasks.")

    def _validate_crew(self) -> None:
        """Validates that all tasks have a valid, assigned agent within the crew."""
        agent_set = set(self.agents)
        for task in self.tasks:
            if task.agent not in agent_set:
                raise OrchestrationError(
                    f"Task '{task.description[:50]}...' is assigned to an agent that is not part of this crew."
                )

    def _construct_prompt(self, agent: Agent, task: Task) -> str:
        """Constructs the full prompt for the LLM, including context and persona."""
        context = self.state_manager.get_full_context()
        
        prompt = f"""
        You are {agent.role}. {agent.backstory}
        Your personal goal is: {agent.goal}

        CURRENT TASK:
        Description: {task.description}
        Expected Output: {task.expected_output}
        
        CONTEXT FROM PREVIOUS TASKS:
        {context}

        Now, execute the task. Provide only the output for the current task, adhering to the 'Expected Output' format.
        Your response should be a direct fulfillment of the task, without any additional commentary or conversational fluff.
        """
        if task.context:
            prompt += f"\n\nADDITIONAL CONTEXT FOR THIS TASK:\n{task.context}"
            
        return prompt

    def kickoff(self) -> str:
        """Starts the execution of the tasks by the crew.

        Iterates through each task, assigns it to the corresponding agent, and executes it.
        The state is managed across tasks.

        Returns:
            The result of the final task in the sequence.
        """
        logger.info("Crew kickoff initiated. Starting task execution...")
        self.state_manager.clear()

        for i, task in enumerate(self.tasks, 1):
            agent = task.agent
            logger.info(f"Executing Task {i}/{len(self.tasks)}: '{task.description}' with Agent: '{agent.role}'")

            prompt = self._construct_prompt(agent, task)
            if self.verbose or agent.verbose:
                logger.info(f"---\n[VERBOSE] Prompt for {agent.role}:\n{prompt}\n---")

            try:
                messages = [ChatMessage(role="user", content=prompt)]
                response = agent.llm_provider.chat(messages)
                output = response.content

                if self.verbose or agent.verbose:
                    logger.info(f"---\n[VERBOSE] Raw Output from {agent.role}:\n{output}\n---")

                self.state_manager.add_task_output(task.description, output)
                logger.info(f"Task {i} completed by Agent '{agent.role}'.")

            except Exception as e:
                logger.error(f"An error occurred during task execution by '{agent.role}': {e}", exc_info=True)
                raise OrchestrationError(f"Failed to execute task: {task.description}") from e

        final_result = self.state_manager.get_final_result()
        logger.info("All tasks completed. Crew execution finished.")
        return final_result
