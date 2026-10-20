import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class TaskExecutionRecord(BaseModel):
    """
    A record of a single task's execution within the orchestration.
    """
    task_id: str
    task_description: str
    agent_role: str
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    input_context: str
    output: Optional[str] = None
    error: Optional[str] = None
    token_usage: Dict[str, int] = Field(default_factory=dict)

class OrchestrationState:
    """
    Manages the state of the entire orchestration process, including task history and context.
    This provides an in-memory "short-term memory" for the crew of agents.
    """
    def __init__(self, orchestration_id: str):
        self.orchestration_id: str = orchestration_id
        self._history: List[TaskExecutionRecord] = []
        self._shared_context: Dict[str, Any] = {}
        logger.info(f"OrchestrationState initialized for ID: {self.orchestration_id}")

    def add_task_record(self, record: TaskExecutionRecord):
        """Adds a completed task execution record to the history."""
        self._history.append(record)
        logger.debug(f"Added record for task {record.task_id} to state.")

    def get_last_output(self) -> Optional[str]:
        """
        Retrieves the output from the most recently completed successful task.
        """
        for record in reversed(self._history):
            if record.output and not record.error:
                return record.output
        return None

    def get_full_history_as_text(self) -> str:
        """
        Serializes the entire task history into a single string for context.
        """
        if not self._history:
            return "No tasks have been executed yet."

        history_str = "## Orchestration History ##\n\n"
        for i, record in enumerate(self._history):
            history_str += f"### Task {i+1}: {record.agent_role} ###\n"
            history_str += f"- Description: {record.task_description}\n"
            history_str += f"- Input Context Provided: {record.input_context[:150]}...\n"
            if record.output:
                history_str += f"- Result: {record.output}\n\n"
            elif record.error:
                history_str += f"- Error: {record.error}\n\n"
        return history_str

    def update_shared_context(self, data: Dict[str, Any]):
        """Updates the shared context with new data."""
        self._shared_context.update(data)
        logger.debug(f"Shared context updated with keys: {list(data.keys())}")

    @property
    def history(self) -> List[TaskExecutionRecord]:
        """Provides read-only access to the execution history."""
        return self._history

    @property
    def shared_context(self) -> Dict[str, Any]:
        """Provides read-only access to the shared context."""
        return self._shared_context
