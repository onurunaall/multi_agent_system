import pytest
from schemas import State, UserInput
from langchain_core.messages import HumanMessage, AIMessage


class TestSchemas:
    """Test cases for schema classes."""

    def test_state_typing(self):
        """Test State TypedDict structure."""
        # Test that State can be created with required fields
        state = State(
            customer_id="123",
            messages=[HumanMessage(content="test")],
            remaining_steps=10
        )

        assert state["customer_id"] == "123"
        assert len(state["messages"]) == 1
        assert state["remaining_steps"] == 10
        assert isinstance(state["messages"][0], HumanMessage)

    def test_user_input_valid(self):
        """Test UserInput with valid identifier."""
        user_input = UserInput(identifier="test@example.com")
        assert user_input.identifier == "test@example.com"

    def test_user_input_empty(self):
        """Test UserInput with empty identifier."""
        user_input = UserInput(identifier="")
        assert user_input.identifier == ""
