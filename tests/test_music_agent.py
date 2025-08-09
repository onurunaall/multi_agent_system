import pytest
from unittest.mock import patch, MagicMock
from agents.music_agent import get_albums_by_artist, get_tracks_by_artist, get_songs_by_genre, check_for_songs, generate_music_assistant_prompt, should_continue, create_music_agent_graph
from schemas import State
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_core.messages.tool import ToolCall

class TestMusicTools:
    @patch('agents.music_agent.db')
    def test_get_albums_by_artist_valid(self, mock_db):
        """Test getting albums for a valid artist."""
        mock_db.run.return_value = "[('Album1', 'Artist1'), ('Album2', 'Artist1')]"
    
        result = get_albums_by_artist("Artist1")
    
        mock_db.run.assert_called_once()
        query = mock_db.run.call_args.args[0]
        params = mock_db.run.call_args.kwargs['parameters']
        assert 'WHERE "Artist"."Name" ILIKE %(artist_name)s' in query
        assert params == {"artist_name": "%Artist1%"}
    
    @patch('agents.music_agent.db')
    def test_get_tracks_by_artist_valid(self, mock_db):
        """Test getting tracks for a valid artist."""
        mock_db.run.return_value = "[('Song1', 'Artist1'), ('Song2', 'Artist1')]"
    
        result = get_tracks_by_artist("Artist1")
    
        mock_db.run.assert_called_once()
        query = mock_db.run.call_args.args[0]
        params = mock_db.run.call_args.kwargs['parameters']
        assert 'WHERE "Artist"."Name" ILIKE %(artist_name)s' in query
        assert params == {"artist_name": "%Artist1%"}
    
    @patch('agents.music_agent.db')
    def test_get_songs_by_genre_valid(self, mock_db):
        """Test getting songs for a valid genre."""
        mock_db.run.return_value = "[{'Song': 'Song1', 'Artist': 'Artist1'}]"
        result = get_songs_by_genre("Rock")
        mock_db.run.assert_called_once()
        params = mock_db.run.call_args.kwargs['parameters']
        assert params == {"genre_name": "%Rock%"}
        assert "No songs found" not in result
    
    @patch('agents.music_agent.db')
    def test_check_for_songs_valid(self, mock_db):
        """Test checking for songs."""
        mock_db.run.return_value = "[('Song1', 'Artist1')]"
    
        result = check_for_songs("Song1")
    
        mock_db.run.assert_called_once()
        query = mock_db.run.call_args.args[0]
        params = mock_db.run.call_args.kwargs['parameters']
        assert 'WHERE "Name" ILIKE %(song_title)s' in query
        assert params == {"song_title": "%Song1%"}


class TestMusicAssistant:
    def test_generate_music_assistant_prompt_no_memory(self):
        """Test prompt generation without memory."""
        prompt = generate_music_assistant_prompt()
        assert "CORE RESPONSIBILITIES" in prompt
        assert "SEARCH GUIDELINES" in prompt
    
    def test_should_continue_with_tool_calls(self):
        """Test should_continue when tool calls exist."""
        state = State(customer_id="123",
                      messages=[AIMessage(content="", tool_calls=[ToolCall(name="test", args={}, id="test_id")])],
                      remaining_steps=10)
    
        assert should_continue(state) == "continue"
    
    def test_should_continue_without_tool_calls(self):
        """Test should_continue when no tool calls."""
        state = State(customer_id="123",
                      messages=[AIMessage(content="Here's the answer")],
                      remaining_steps=10)
    
        assert should_continue(state) == "end"
