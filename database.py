"""
Creates an in-memory SQLite database with sample data.
"""

import sqlite3
import requests
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool


def get_engine_for_chinook_db():
    """
    Downloads the Chinook sample database SQL script and creates 
    an in-memory SQLite database populated with the data.
    
    Returns:
        sqlalchemy.Engine: The database engine object
    """
    sql_file_url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(sql_file_url)
    response.raise_for_status()
    sql_script = response.text
    
    # Creatimg in-memory SQLite database using sqlite3
    conn = sqlite3.connect(":memory:")
    conn.executescript(sql_script)
    conn.commit()
    
    # Creating SQLAlchemy engine from the existing connection
    engine = create_engine("sqlite:///:memory:",
                           creator=lambda: conn,
                           poolclass=StaticPool,
                           connect_args={"check_same_thread": False})
    
    return engine


# Create the database engine
engine = get_engine_for_chinook_db()

# Initialize the LangChain SQLDatabase wrapper
db = SQLDatabase(engine)
