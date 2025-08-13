import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

class TestMusicTools:
    """Test cases for the modern music agent tools."""
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("config.create_engine")
    @patch("agents.music_agent.vector_retriever.similarity_search")
    def test_search_for_music_valid(self, mock_similarity_search, mock_engine):
        """Test the semantic search tool for music."""
        mock_engine.return_value = MagicMock()
        from agents.music_agent import search_for_music
        
        mock_similarity_search.return_value = [Document(page_content="Track: Smells Like Teen Spirit")]

        result = search_for_music.invoke({"query": "songs by nirvana"})
        
        mock_similarity_search.assert_called_once_with("songs by nirvana", k=5)
        assert "Smells Like Teen Spirit" in result

class TestMusicAgentGraph:
    """Test cases for the music agent graph itself."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("langchain_openai.ChatOpenAI.ainvoke", new_callable=AsyncMock)
    def test_should_continue_music_with_tool_calls(self, mock_llm_ainvoke):
        """Test the conditional edge for continuing with a tool call."""
        from agents.music_agent import should_continue_music
        
        mock_llm_ainvoke.return_value = AIMessage(
            content="",
            tool_calls=[{"name": "search_for_music", "args": {"query": "test"}, "id": "123"}]
        )

        state = {"messages": [mock_llm_ainvoke.return_value]}
        decision = should_continue_music(state)
        assert decision == "continue"

    def test_should_continue_music_without_tool_calls(self):
        """Test the conditional edge for ending the graph."""
        from agents.music_agent import should_continue_music
        state = {"messages": [AIMessage(content="Final Answer")]}
        decision = should_continue_music(state)
        assert decision == "end"
