import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from workflow import get_customer_id_from_identifier, verify_info, should_interrupt, format_user_memory, load_memory, \
    create_memory
from schemas import State, UserProfile
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import json


class TestWorkflowHelpers:
    """Test cases for workflow helper functions."""

    def test_get_customer_id_from_identifier_numeric(self):
        """Test getting customer ID from numeric identifier."""
        result = get_customer_id_from_identifier("123")
        assert result == 123

    @patch('workflow.db')
    def test_get_customer_id_from_identifier_phone(self, mock_db):
        """Test getting customer ID from phone number."""
        mock_db.run.return_value = "[(456,)]"

        result = get_customer_id_from_identifier("+1234567890")

        assert result == 456
        query, params = mock_db.run.call_args[0]

        assert "WHERE Phone = ?" in query
        assert params == ("+1234567890",)

    @patch('workflow.db')
    def test_get_customer_id_from_identifier_email(self, mock_db):
        """Test getting customer ID from email."""
        mock_db.run.return_value = "[(789,)]"

        result = get_customer_id_from_identifier("test@example.com")

        assert result == 789
        query, params = mock_db.run.call_args[0]
        assert "WHERE Email = ?" in query
        assert params == ("test@example.com",)

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
                      loaded_memory="",
                      remaining_steps=10)

        assert should_interrupt(state, None) == "continue"

    def test_should_interrupt_without_customer_id(self):
        """Test should_interrupt when customer ID is missing."""
        state = State(customer_id=None,
                      messages=[],
                      loaded_memory="",
                      remaining_steps=10)

        assert should_interrupt(state, None) == "interrupt"

    def test_format_user_memory_with_preferences(self):
        """Test formatting user memory with preferences."""
        user_data = {"memory": UserProfile(customer_id="123", music_preferences=["Rock", "Jazz", "Blues"]).model_dump()}
        user_data_bytes = json.dumps(user_data).encode('utf-8')
        result = format_user_memory(user_data_bytes)
        assert result == "Music Preferences: Rock, Jazz, Blues"

    def test_format_user_memory_empty_preferences(self):
        """Test formatting user memory with empty preferences."""
        user_data = {"memory": UserProfile(customer_id="123", music_preferences=[]).model_dump()}
        user_data_bytes = json.dumps(user_data).encode('utf-8')
        result = format_user_memory(user_data_bytes)
        assert result == ""


class TestWorkflowNodes:
    """Unit tests for the primary workflow nodes."""

    @patch('workflow.get_customer_id_from_identifier')
    @patch('workflow.structured_llm')
    def test_verify_info_finds_customer(self, mock_llm, mock_lookup):
        """verify_info returns customer_id when identifier matches."""
        # Mock identifier extraction
        identifier = 'user@example.com'
        mock_llm.invoke.return_value = MagicMock(identifier=identifier)
        mock_lookup.return_value = 123

        state = State(messages=[HumanMessage(content='my email is user@example.com')],
                      customer_id=None,
                      loaded_memory="",
                      remaining_steps=10)

        result = verify_info(state, None)

        assert result['customer_id'] == 123
        assert any('verified' in str(m.content).lower()
                   for m in result['messages'] if hasattr(m, 'content'))

    @patch('workflow.get_customer_id_from_identifier')
    @patch('workflow.structured_llm')
    @patch('workflow.llm')
    def test_verify_info_prompts_for_id(self, mock_llm, mock_structured_llm, mock_lookup):
        """verify_info prompts for more info when lookup fails."""
        mock_structured_llm.invoke.return_value = MagicMock(identifier='missing@example.com')
        mock_lookup.return_value = None
        mock_llm.invoke.return_value = AIMessage(content="Please provide your customer ID")

        state = State(messages=[HumanMessage(content='my email is missing@example.com')],
                      customer_id=None,
                      loaded_memory="",
                      remaining_steps=10)

        result = verify_info(state, None)

        assert isinstance(result['messages'][-1], AIMessage)
        assert 'provide' in result['messages'][-1].content.lower()

    def _dummy_state(self):
        return State(customer_id='123', messages=[], loaded_memory="", remaining_steps=10)

    def test_load_memory_finds_profile(self):
        """load_memory returns formatted user preferences string."""
        mock_store = MagicMock()
        mock_profile = UserProfile(customer_id='123', music_preferences=['Rock', 'Jazz'])
        data_to_load = {'memory': mock_profile.model_dump()}
        bytes_to_load = json.dumps(data_to_load).encode('utf-8')

        mock_store.mget.return_value = [bytes_to_load]

        result = load_memory(self._dummy_state(), None, mock_store)
        assert result['loaded_memory'] == 'Music Preferences: Rock, Jazz'

    @patch('workflow.llm')
    def test_create_memory_updates_profile(self, mock_llm):
        """create_memory stores new user profile in the store."""
        mock_store = MagicMock()
        mock_store.mget.return_value = [None]  # No existing profile
        new_profile = UserProfile(customer_id='123', music_preferences=['Classical'])

        mock_llm.with_structured_output.return_value.invoke.return_value = new_profile

        create_memory(self._dummy_state(), None, mock_store)

        mock_store.mset.assert_called_once()

        # The first argument to mset() is a list of tuples
        args_list = mock_store.mset.call_args[0][0]
        key, data_to_store_bytes = args_list[0]
        
        data_to_store = json.loads(data_to_store_bytes.decode('utf-8'))

        assert key == "memory_profile_123"
        assert data_to_store['memory']['music_preferences'] == ['Classical']