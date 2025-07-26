import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from workflow import (
    get_customer_id_from_identifier,
    verify_info,
    human_input,
    should_interrupt,
    format_user_memory,
    load_memory,
    create_memory
)
from schemas import State, UserProfile
from langchain_core.messages import HumanMessage, SystemMessage


class TestWorkflowHelpers:
    """Test cases for workflow helper functions."""
    
    @patch('workflow.db')
    def test_get_customer_id_from_identifier_numeric(self, mock_db):
        """Test getting customer ID from numeric identifier."""
        result = get_customer_id_from_identifier("123")
        assert result == 123
        mock_db.run.assert_not_called()
    
    @patch('workflow.db')
    def test_get_customer_id_from_identifier_phone(self, mock_db):
        """Test getting customer ID from phone number."""
        mock_db.run.return_value = "[(456,)]"
        
        result = get_customer_id_from_identifier("+1234567890")
        
        assert result == 456
        assert "Phone = '+1234567890'" in mock_db.run.call_args[0][0]
    
    @patch('workflow.db')
    def test_get_customer_id_from_identifier_email(self, mock_db):
        """Test getting customer ID from email."""
        mock_db.run.return_value = "[(789,)]"
        
        result = get_customer_id_from_identifier("test@example.com")
        
        assert result == 789
        assert "Email = 'test@example.com'" in mock_db.run.call_args[0][0]
    
    @patch('workflow.db')
    def test_get_customer_id_from_identifier_not_found(self, mock_db):
        """Test getting customer ID when not found."""
        mock_db.run.return_value = "[]"
        
        result = get_customer_id_from_identifier("unknown@example.com")
        
        assert result is None
    
    def test_should_interrupt_with_customer_id(self):
        """Test should_interrupt when customer ID exists."""
        state = State(customer_id="123",
                      messages=[],
                      loaded_memory=[],
                      remaining_steps=10)
      
        assert should_interrupt(state, None) == "continue"
    
    def test_should_interrupt_without_customer_id(self):
        """Test should_interrupt when customer ID is missing."""
        state = State(customer_id=None,
                      messages=[],
                      loaded_memory=[],
                      remaining_steps=10)
      
        assert should_interrupt(state, None) == "interrupt"
    
    def test_format_user_memory_with_preferences(self):
        """Test formatting user memory with preferences."""
        user_data = {"memory": UserProfile(customer_id="123",
                                           music_preferences=["Rock", "Jazz", "Blues"])}
      
        result = format_user_memory(user_data)
        assert result == "Music Preferences: Rock, Jazz, Blues"
    
    def test_format_user_memory_empty_preferences(self):
        """Test formatting user memory with empty preferences."""
        user_data = {
            "memory": UserProfile(customer_id="123",
                                  music_preferences=[])}
      
        result = format_user_memory(user_data)
        assert result == ""
