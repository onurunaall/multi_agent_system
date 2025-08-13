import pytest
import asyncio
import sys
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
    @patch('sys.stdout')
    async def test_main_normal_flow(self, mock_stdout, mock_input, mock_graph):
        """Test normal conversation flow."""
        mock_input.side_effect = ['Hello', 'exit']
        mock_graph.ainvoke.return_value = {
            "messages": [AIMessage(content="Hello! How can I help you?")]
        }

        from main import main
        await main()

        mock_graph.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    @patch('main.multi_agent_final_graph')  
    @patch('builtins.input')
    @patch('sys.stdout')
    async def test_main_interrupt_and_resume_flow(self, mock_stdout, mock_input, mock_graph):
        """Test multiple conversation turns."""
        mock_input.side_effect = ['First message', 'Second message', 'exit']
        mock_graph.ainvoke.side_effect = [
            {"messages": [AIMessage(content="First response")]},
            {"messages": [AIMessage(content="Second response")]}
        ]

        from main import main
        await main()

        assert mock_graph.ainvoke.call_count == 2