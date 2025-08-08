from typing import TypedDict, Annotated, List
from pydantic import BaseModel, Field
from langgraph.graph.message import AnyMessage, add_messages

class State(TypedDict):
    """Central state schema that flows through the entire agent graph."""
    customer_id: str
    messages: Annotated[List[AnyMessage], add_messages]
    remaining_steps: int

class UserInput(BaseModel):
    """Schema for extracting customer identifiers during verification."""
    identifier: str = Field(description="The customer ID or name provided by the user")
