import pytest
from unittest.mock import patch, MagicMock
from workflow import get_customer_id_from_identifier, verify_info, should_interrupt
from schemas import State
from langchain_core.messages import HumanMessage, AIMessage

class TestWorkflowHelpers:
    @patch('workflow.engine')
    def test_get_customer_id_from_identifier_email(self, mock_engine):
        """Test getting customer ID from email."""
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_connection.execute.return_value.scalar_one_or_none.return_value = 789
    
        result = get_customer_id_from_identifier("test@example.com")
        assert result == 789
        mock_connection.execute.assert_called_once()
        assert 'WHERE "Email" = :identifier' in str(mock_connection.execute.call_args.args[0])
    
    @patch('workflow.engine')
    def test_get_customer_id_from_identifier_not_found(self, mock_engine):
        """Test getting customer ID when not found."""
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_connection.execute.return_value.scalar_one_or_none.return_value = None
    
        result = get_customer_id_from_identifier("unknown@example.com")
        assert result is None
    
    def test_should_interrupt_with_customer_id(self):
        """Test should_interrupt when customer ID exists."""
        state = State(customer_id="123", messages=[], remaining_steps=10)
        assert should_interrupt(state, None) == "continue"
    
    def test_should_interrupt_without_customer_id(self):
        """Test should_interrupt when customer ID is missing."""
        state = State(customer_id=None, messages=[], remaining_steps=10)
        assert should_interrupt(state, None) == "interrupt"

class TestWorkflowNodes:
    @patch('workflow.get_customer_id_from_identifier')
    def test_verify_info_finds_customer(self, mock_lookup):
        """verify_info returns customer_id when identifier matches."""
        mock_lookup.return_value = 123
        state = State(messages=[HumanMessage(content='my email is user@example.com')], customer_id=None, remaining_steps=10)
    
        result = verify_info(state, None)
    
        mock_lookup.assert_called_with("user@example.com")
        assert result['customer_id'] == "123"
        assert any('verified' in str(m.content).lower() for m in result['messages'])
    
    @patch('workflow.get_customer_id_from_identifier')
    def test_verify_info_prompts_for_id_on_failure(self, mock_lookup):
        """verify_info prompts for more info when lookup fails."""
        mock_lookup.return_value = None
        state = State(messages=[HumanMessage(content='my email is missing@example.com')], customer_id=None, remaining_steps=10)
    
        result = verify_info(state, None)
    
        assert 'customer_id' not in result
        assert isinstance(result['messages'][-1], AIMessage)
        assert "couldn't find an account" in result['messages'][-1].content.lower()
