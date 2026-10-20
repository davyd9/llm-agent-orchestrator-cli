import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class StateManager:
    """Manages the state of a task-oriented process, tracking outputs and context.

    This is a simple in-memory implementation designed to be easily swappable
    with more persistent storage solutions like Redis or a database.
    """

    def __init__(self):
        """Initializes the state manager with an empty history."""
        self._history: List[Tuple[str, str]] = []
        logger.info("StateManager initialized.")

    def add_task_output(self, task_description: str, output: str) -> None:
        """Adds the output of a completed task to the state history.

        Args:
            task_description: The description of the task that was executed.
            output: The result or output from the task execution.
        """
        if not isinstance(task_description, str) or not isinstance(output, str):
            raise TypeError("Task description and output must be strings.")
        
        self._history.append((task_description, output))
        logger.debug(f"Added output for task: '{task_description[:50]}...' to state.")

    def get_full_context(self) -> str:
        """Assembles the entire history of tasks and outputs into a single context string.

        This context can be passed to subsequent tasks to provide them with the
        history of what has been accomplished so far.

        Returns:
            A formatted string containing all previous task descriptions and their outputs.
        """
        if not self._history:
            return "No previous tasks have been completed. You are the first one."

        context_parts = [
            "This is the summary of the work done so far:"
        ]
        for i, (desc, out) in enumerate(self._history, 1):
            context_parts.append(f"\n----- Task {i} -----")
            context_parts.append(f"Task Description: {desc}")
            context_parts.append(f"Task Output:\n{out}")
        
        context_string = "\n".join(context_parts)
        logger.debug(f"Assembled context of length {len(context_string)}.")
        return context_string

    def get_final_result(self) -> str:
        """Retrieves the output of the very last task executed.

        Returns:
            The output string of the last task, or an empty string if no tasks were run.
        """
        if not self._history:
            return ""
        return self._history[-1][1]

    def clear(self) -> None:
        """Resets the state, clearing all history."""
        self._history = []
        logger.info("StateManager state has been cleared.")

    @property
    def history(self) -> List[Tuple[str, str]]:
        """Provides read-only access to the execution history."""
        return self._history
