import uuid
from typing import Dict, Any
from pydantic import BaseModel, Field

class Agent(BaseModel):
    """
    Represents an autonomous agent with a specific role, goal, and backstory.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the agent.")
    role: str = Field(..., description="The role of the agent (e.g., 'Senior Software Engineer').")
    goal: str = Field(..., description="The primary objective of the agent.")
    backstory: str = Field(..., description="A narrative describing the agent's background and expertise.")
    llm_provider_name: str = Field(..., description="The name of the LLM provider to use (e.g., 'openai', 'deepseek').")
    llm_config: Dict[str, Any] = Field(default_factory=dict, description="Configuration for the LLM provider, including model name.")

    class Config:
        frozen = True # Agents are immutable definitions

class Task(BaseModel):
    """
    Represents a specific unit of work to be performed by an agent.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the task.")
    description: str = Field(..., description="A detailed description of the task to be performed.")
    agent: Agent = Field(..., description="The agent assigned to perform this task.")
    expected_output: str = Field(..., description="A clear description of the expected output format or content.")
    context: Dict[str, Any] = Field(default_factory=dict, description="Initial context or data required to start the task.")

    class Config:
        arbitrary_types_allowed = True
