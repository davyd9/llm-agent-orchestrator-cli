import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from ..llm.base_provider import TokenUsage

class TaskExecutionRecord(BaseModel):
    """Records the execution details of a single task."""
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the record.")
    task_id: str = Field(..., description="ID of the task being executed.")
    agent_id: str = Field(..., description="ID of the agent executing the task.")
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    status: str = Field("PENDING", description="Execution status (e.g., PENDING, RUNNING, COMPLETED, FAILED).")
    input_data: Dict[str, Any] = Field(..., description="The input data provided to the task.")
    output_data: Optional[str] = Field(None, description="The final output or result from the task execution.")
    error: Optional[str] = Field(None, description="Error message if the task failed.")
    token_usage: Optional[TokenUsage] = Field(None, description="Token usage for the LLM call within this task.")
    cost: Optional[float] = Field(None, description="Cost of the LLM call within this task.")


class OrchestrationState(BaseModel):
    """Manages the overall state of an orchestration run, including context and history."""
    orchestration_id: str = Field(default_factory=lambda: f"run-{uuid.uuid4()}", description="Unique ID for the entire orchestration run.")
    shared_context: Dict[str, Any] = Field(default_factory=dict, description="Context shared across all tasks.")
    execution_records: List[TaskExecutionRecord] = Field(default_factory=list, description="A log of all task executions.")

    def add_record(self, record: TaskExecutionRecord):
        """Adds a new task execution record to the history."""
        self.execution_records.append(record)

    def get_record(self, record_id: str) -> Optional[TaskExecutionRecord]:
        """Retrieves a specific record by its ID."""
        for record in self.execution_records:
            if record.record_id == record_id:
                return record
        return None

    def update_record(self, record_id: str, **updates: Any):
        """Updates an existing record with new data."""
        record = self.get_record(record_id)
        if record:
            for key, value in updates.items():
                if hasattr(record, key):
                    setattr(record, key, value)
                else:
                    # This could be a log warning or an error depending on strictness
                    pass
            record.end_time = datetime.now(timezone.utc)

    def get_last_output(self) -> Optional[str]:
        """Gets the output from the most recently completed task."""
        completed_records = sorted(
            [rec for rec in self.execution_records if rec.status == 'COMPLETED' and rec.end_time],
            key=lambda r: r.end_time, # type: ignore
            reverse=True
        )
        return completed_records[0].output_data if completed_records else None

    @property
    def total_input_tokens(self) -> int:
        """Calculates the total number of input tokens used in the orchestration."""
        return sum(
            record.token_usage.input_tokens
            for record in self.execution_records
            if record.token_usage
        )

    @property
    def total_output_tokens(self) -> int:
        """Calculates the total number of output tokens used in the orchestration."""
        return sum(
            record.token_usage.output_tokens
            for record in self.execution_records
            if record.token_usage
        )

    @property
    def total_tokens(self) -> int:
        """Calculates the grand total of tokens used in the orchestration."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost(self) -> float:
        """Calculates the total cost of the orchestration."""
        return sum(
            record.cost
            for record in self.execution_records
            if record.cost is not None
        )
