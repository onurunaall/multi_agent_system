import pytest
from pydantic import ValidationError
from schemas import State, UserInput, UserProfile
from langchain_core.messages import HumanMessage


class TestSchemas:
    """Test cases for Pydantic schemas."""
    
    def test_user_input_valid(self):
        """Test valid UserInput creation."""
        user_input = UserInput(identifier="test@example.com")
        assert user_input.identifier == "test@example.com"
    
    def test_user_input_empty(self):
        """Test UserInput with empty identifier."""
        user_input = UserInput(identifier="")
        assert user_input.identifier == ""
    
    def test_user_profile_valid(self):
        """Test valid UserProfile creation."""
        profile = UserProfile(
            customer_id="123",
            music_preferences=["Rock", "Jazz"]
        )
        assert profile.customer_id == "123"
        assert len(profile.music_preferences) == 2
    
    def test_user_profile_empty_preferences(self):
        """Test UserProfile with empty preferences."""
        profile = UserProfile(
            customer_id="123",
            music_preferences=[]
        )
        assert profile.music_preferences == []
    
    def test_state_typing(self):
        """Test State TypedDict structure."""
        state = State(
            customer_id="123",
            messages=[HumanMessage(content="Hello")],
            loaded_memory="preference1",
            remaining_steps=10
        )
        assert state["customer_id"] == "123"
        assert len(state["messages"]) == 1
        assert state["remaining_steps"] == 10