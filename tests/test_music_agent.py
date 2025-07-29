import pytest
from unittest.mock import patch, MagicMock
from agents.music_agent import get_albums_by_artist, get_tracks_by_artist, get_songs_by_genre, check_for_songs, generate_music_assistant_prompt, should_continue, create_music_agent_graph
from schemas import State
from langchain_core.messages import AIMessage


class TestMusicTools:
    """Test cases for music agent tools."""

    @patch('agents.music_agent.db')
    def test_get_albums_by_artist_valid(self, mock_db):
        """Test getting albums for a valid artist."""
        mock_db.run.return_value = "[('Album1', 'Artist1'), ('Album2', 'Artist1')]"

        result = get_albums_by_artist("Artist1")

        mock_db.run.assert_called_once()
        query, params = mock_db.run.call_args[0]
        assert '?' in query
        assert params == ["%Artist1%"]

    def test_get_albums_by_artist_empty(self):
        """Test getting albums with empty artist name."""
        with pytest.raises(ValueError, match="Artist name required"):
            get_albums_by_artist("")

    def test_get_albums_by_artist_whitespace(self):
        """Test getting albums with whitespace artist name."""
        with pytest.raises(ValueError, match="Artist name required"):
            get_albums_by_artist("   ")

    @patch('agents.music_agent.db')
    def test_get_tracks_by_artist_valid(self, mock_db):
        """Test getting tracks for a valid artist."""
        mock_db.run.return_value = "[('Song1', 'Artist1'), ('Song2', 'Artist1')]"

        result = get_tracks_by_artist("Artist1")

        mock_db.run.assert_called_once()
        query, params = mock_db.run.call_args[0]
        
        assert '?' in query
        assert params == ["%Artist1%"]

    @patch('agents.music_agent.db')
    def test_get_songs_by_genre_valid(self, mock_db):
        """Test getting songs for a valid genre."""
        # Mock genre ID query then songs query
        mock_db.run.side_effect = ["[(1,)]",
                                   "[{'Song': 'Song1', 'Artist': 'Artist1'}]"]

        result = get_songs_by_genre("Rock")

        assert mock_db.run.call_count == 2
        for call_args in mock_db.run.call_args_list:
            query, params = call_args[0]
            assert '?' in query
        
        assert any(call_args[0][1][0] == "%Rock%" for call_args in mock_db.run.call_args_list)
        assert isinstance(result, list)

    @patch('agents.music_agent.db')
    def test_get_songs_by_genre_not_found(self, mock_db):
        """Test getting songs for non-existent genre."""
        mock_db.run.return_value = ""

        result = get_songs_by_genre("NonExistentGenre")

        assert "No songs found" in result

    @patch('agents.music_agent.db')
    def test_check_for_songs_valid(self, mock_db):
        """Test checking for songs."""
        mock_db.run.return_value = "[('Song1', 'Artist1')]"

        result = check_for_songs("Song1")

        mock_db.run.assert_called_once()
        query, params = mock_db.run.call_args[0]
        assert '?' in query
        assert params == ["%Song1%"]


class TestMusicAssistant:
    """Test cases for music assistant functionality."""

    def test_generate_music_assistant_prompt_no_memory(self):
        """Test prompt generation without memory."""
        prompt = generate_music_assistant_prompt()
        assert "Prior saved user preferences: None" in prompt
        assert "CORE RESPONSIBILITIES" in prompt
        assert "SEARCH GUIDELINES" in prompt

    def test_generate_music_assistant_prompt_with_memory(self):
        """Test prompt generation with memory."""
        prompt = generate_music_assistant_prompt("Likes Rock and Jazz")
        assert "Prior saved user preferences: Likes Rock and Jazz" in prompt

    def test_should_continue_with_tool_calls(self):
        """Test should_continue when tool calls exist."""
        state = State(customer_id="123",
                      messages=[AIMessage(content="", tool_calls=[{"name": "test"}])],
                      loaded_memory="")

        assert should_continue(state) == "continue"

    def test_should_continue_without_tool_calls(self):
        """Test should_continue when no tool calls."""
        state = State(customer_id="123",
                      messages=[AIMessage(content="Here's the answer")],
                      loaded_memory="")

        assert should_continue(state) == "end"