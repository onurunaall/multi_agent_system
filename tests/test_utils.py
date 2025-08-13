import pytest
from unittest.mock import patch, MagicMock, mock_open
from utils import save_graph_diagram
import importlib
import sys


class TestUtils:
    """Test cases for utility functions."""

    @patch('builtins.open', new_callable=mock_open)
    @patch('utils.print')
    def test_save_graph_diagram_success(self, mock_print, mock_file):
        """Test successful graph diagram save."""
        mock_graph = MagicMock()
        mock_graph.get_graph().draw_mermaid_png.return_value = b"PNG_DATA"

        save_graph_diagram(mock_graph, "test.png")

        mock_file.assert_called_once_with("test.png", "wb")
        mock_file().write.assert_called_once_with(b"PNG_DATA")
        mock_print.assert_called_with("Graph diagram saved to test.png")

    @patch('builtins.open', new_callable=mock_open)
    @patch('utils.print')
    def test_save_graph_diagram_fallback(self, mock_print, mock_file):
        """Test fallback method when primary fails."""
        mock_graph = MagicMock()
        mock_graph.get_graph().draw_mermaid_png.side_effect = [Exception("Primary failed"),
                                                               b"PNG_DATA"]

        # Mock the nest_asyncio module by inserting it into sys.modules
        mock_nest_asyncio = MagicMock()
        sys.modules['nest_asyncio'] = mock_nest_asyncio

        # Reload the utils module to ensure it picks up the mocked import
        import utils
        importlib.reload(utils)

        try:
            utils.save_graph_diagram(mock_graph, "test.png")

            mock_nest_asyncio.apply.assert_called_once()
            mock_file.assert_called_with("test.png", "wb")
            assert any("fallback method" in str(call) for call in mock_print.call_args_list)
        finally:
            # Clean up by removing the mocked module
            if 'nest_asyncio' in sys.modules:
                del sys.modules['nest_asyncio']