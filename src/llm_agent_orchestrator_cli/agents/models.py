from typing import Optional
from pydantic import BaseModel, Field

from llm_agent_orchestrator_cli.llm.base_provider import LLMProvider

class Agent(BaseModel):
    """Represents an autonomous agent with a specific role, goal, and LLM configuration."""
    role: str = Field(..., description="The role of the agent (e.g., 'Senior Software Engineer').")
    goal: str = Field(..., description="The primary objective of the agent.")
    backstory: str = Field(..., description="A backstory that defines the agent's personality and expertise.")
    llm_provider: LLMProvider = Field(..., description="The LLM provider instance this agent will use.")
    verbose: bool = Field(False, description="Enable verbose logging for the agent's operations.")

    class Config:
        arbitrary_types_allowed = True

class Task(BaseModel):
    """Represents a single, actionable task to be performed by an agent."""
    description: str = Field(..., description="A clear and concise description of the task.")
    expected_output: str = Field(..., description="A description of the expected output format or content.")
    agent: Agent = Field(..., description="The agent assigned to perform this task.")
    context: Optional[str] = Field(None, description="Additional context or data required to perform the task.")

    class Config:
        arbitrary_types_allowed = True
