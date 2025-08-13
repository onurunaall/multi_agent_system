from typing import TypedDict, Annotated, List, Optional
from pydantic import BaseModel, Field
from langgraph.graph.message import AnyMessage, add_messages

class State(TypedDict):
    """Central state schema that flows through the entire agent graph."""
    customer_id: Optional[str]
    messages: Annotated[List[AnyMessage], add_messages]

class UserInput(BaseModel):
    """Schema for extracting customer identifiers during verification."""
    identifier: str = Field(description="The customer ID or name provided by the user")