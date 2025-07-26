import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from io import StringIO


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
        
        mock_print.assert_any_call("Thank you. Shutting Down!")
        mock_graph.astream_events.assert_not_called()
    
    @pytest.mark.asyncio
    @patch('main.multi_agent_final_graph')
    @patch('builtins.input')
    @patch('builtins.print')
    async def test_main_normal_flow(self, mock_print, mock_input, mock_graph):
        """Test normal conversation flow."""
        # Setup input sequence
        mock_input.side_effect = ['Hello', 'exit']
        
        # Mock astream_events
        async def mock_stream():
            yield {
                "event": "on_chain_end",
                "data": {
                    "output": {
                        "messages": [AIMessage(content="Hello! How can I help you?")]
                    }
                }
            }
        
        mock_graph.astream_events.return_value = mock_stream()
        
        from main import main
        await main()
        
        mock_graph.astream_events.assert_called_once()
        mock_print.assert_any_call("\nAssistant: Hello! How can I help you?\n")
