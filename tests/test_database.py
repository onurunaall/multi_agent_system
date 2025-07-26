import pytest
from unittest.mock import patch, MagicMock
import sqlite3
from database import get_engine_for_chinook_db, db


class TestDatabase:
    """Test cases for database functionality."""
    
    @patch('database.requests.get')
    def test_get_engine_for_chinook_db_success(self, mock_get):
        """Test successful database engine creation."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.text = "CREATE TABLE Test (id INT); INSERT INTO Test VALUES (1);"
        mock_get.return_value = mock_response
        
        # Call the function
        engine = get_engine_for_chinook_db()
        
        # Assertions
        assert engine is not None
        mock_get.assert_called_once_with(
            "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
        )
    
    @patch('database.requests.get')
    def test_get_engine_for_chinook_db_request_failure(self, mock_get):
        """Test handling of request failure."""
        mock_get.side_effect = Exception("Network error")
        
        with pytest.raises(Exception):
            get_engine_for_chinook_db()
    
    def test_db_object_exists(self):
        """Test that db object is properly initialized."""
        assert db is not None
