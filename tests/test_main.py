import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from io import StringIO
from langchain_core.messages import AIMessage


class TestMain:
    """Test cases for main application."""

    @pytest.mark.asyncio
    @patch('main.multi_agent_final_graph')
    @patch('builtins.input')
    @patch('builtins.print')
    async def test_main_exit_command(self, mock_print, mock_input, mock_graph):
        """Test that 'exit' command properly exits the application."""
        mock_input.return_value = 'exit'

        from main import main
        await main()

        mock_graph.astream_events.assert_not_called()
        mock_print.assert_any_call("Thank you for using Customer Support. Goodbye!")

    @pytest.mark.asyncio
    @patch('main.multi_agent_final_graph')
    @patch('builtins.input')
    @patch('builtins.print')
    async def test_main_normal_flow(self, mock_print, mock_input, mock_graph):
        """Test normal conversation flow."""
        mock_input.side_effect = ['Hello', 'exit']

        async def mock_stream(*args, **kwargs):
            yield {
                "event": "on_chain_end",
                "data": {"output": {"messages": [AIMessage(content="Hello! How can I help you?")]}}
            }

        mock_graph.astream_events.return_value = mock_stream()

        from main import main
        await main()

        mock_graph.astream_events.assert_called_once()
        mock_print.assert_any_call("\nAssistant: Hello! How can I help you?\n")

    @pytest.mark.asyncio
    @patch('main.multi_agent_final_graph')
    @patch('builtins.input')
    @patch('builtins.print')
    async def test_main_interrupt_and_resume_flow(self, mock_print, mock_input, mock_graph):
        """Test interrupt-and-resume flow with two turns."""
        mock_input.side_effect = ['I need my last invoice', 'some-email@example.com', 'exit']

        async def first_stream(*args, **kwargs):
            yield {
                'event': 'on_chain_stream',
                'name': 'human_input',
                'data': {'chunk': {'messages': [AIMessage(content='May I have your e-mail?')]}}
            }

        async def second_stream(*args, **kwargs):
            yield {
                'event': 'on_chain_end',
                'data': {'output': {'messages': [AIMessage(content='Your last invoice total is $42.00')]}}
            }

        mock_graph.astream_events.side_effect = [first_stream(), second_stream()]

        from main import main
        await main()

        assert mock_graph.astream_events.call_count == 2
        mock_print.assert_any_call('\nAssistant: May I have your e-mail?\n')
        mock_print.assert_any_call('\nAssistant: Your last invoice total is $42.00\n')