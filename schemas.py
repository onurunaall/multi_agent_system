from typing import TypedDict, Annotated, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph.message import AnyMessage
from langgraph.managed import RemainingSteps

class State(TypedDict):
    """Central state schema that flows through the entire agent graph."""
    customer_id: str
    messages: List[AnyMessage]
    loaded_memory: List[str]
    remaining_steps: Annotated[int, RemainingSteps]

class UserInput(BaseModel):
    """Schema for extracting customer identifiers during verification."""
    identifier: str = Field(description="The customer ID or name provided by the user")

class UserProfile(BaseModel):
    """Schema for storing user preferences in long-term memory."""
    customer_id: str = Field(description="The customer’s ID")
    music_preferences: List[str] = Field(description="Customer’s music preferences and listening history")
