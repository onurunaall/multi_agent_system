from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import ast

from config import llm, checkpointer, store
from database import db
from schemas import State

# Find all albums by a specific artist.
@tool
def get_albums_by_artist(artist: str):
    """Get albums by an artist."""
    if not artist or not artist.strip():
        raise ValueError("Artist name required")

    artist = artist.strip()

    return db.run(
        """
        SELECT Album.Title, Artist.Name
        FROM Album
        JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Artist.Name LIKE ?
        """,
        [f"%{artist}%"],
        include_columns=True
    )

# Get all the songs by a given artist.
@tool
def get_tracks_by_artist(artist: str):
    """Get songs by an artist (or similar artists)."""
    if not artist or not artist.strip():
        raise ValueError("Artist name required")
    
    artist = artist.strip()
    
    return db.run(
        """
        SELECT Track.Name AS SongName, Artist.Name AS ArtistName
        FROM Album
        LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId
        LEFT JOIN Track ON Track.AlbumId = Album.AlbumId
        WHERE Artist.Name LIKE ?
        """,
        [f"%{artist}%"],
        include_columns=True
    )

# Find songs that belong to a specific genre.
@tool
def get_songs_by_genre(genre: str):
    """
    Fetch songs that match a specific genre.
    Args:
        genre (str): The genre of the songs to fetch.
    Returns:
        list[dict]: A list of songs that match the specified genre.
    """
    if not genre or not genre.strip():
        return "Genre name is required"
        
    genre = genre.strip()
    genre_ids_raw = db.run("SELECT GenreId FROM Genre WHERE Name LIKE ?",
                           [f"%{genre}%"])
    
    if not genre_ids_raw:
        return f"No songs found for the genre: {genre}"

    genre_ids = ast.literal_eval(genre_ids_raw)
    genre_id_values = [gid[0] for gid in genre_ids]
    placeholders = ", ".join("?" * len(genre_id_values))

    songs_query = f"""
        SELECT Track.Name AS Song, Artist.Name AS Artist
        FROM Track
        LEFT JOIN Album  ON Track.AlbumId  = Album.AlbumId
        LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Track.GenreId IN ({placeholders})
        GROUP BY Artist.Name;
    """
    songs_raw = db.run(songs_query, genre_id_values, include_columns=True)

    if not songs_raw:
        return f"No songs found for the genre: {genre}"

    return ast.literal_eval(songs_raw)

@tool
def check_for_songs(song_title: str):
    """Check if a song exists by its name."""
    return db.run(
        """
        SELECT * FROM Track
        WHERE Name LIKE ?;
        """,
        [f"%{song_title}%"],
        include_columns=True,
    )

music_tools = [get_albums_by_artist, get_tracks_by_artist, get_songs_by_genre, check_for_songs]

llm_with_music_tools = llm.bind_tools(music_tools)
music_tool_node = ToolNode(music_tools)

def generate_music_assistant_prompt(memory: str = "None") -> str:
    music_assistant_prompt =  f"""
You are one of several specialised assistants; your focus is the music-catalog. If the catalog is missing an artist’s material, simply say so.  
Prior saved user preferences: {memory}

CORE RESPONSIBILITIES
- Search for songs, albums, artists, playlists
- Recommend music based on user interests
- Answer only music-related questions

SEARCH GUIDELINES
1. Always search thoroughly before concluding something is unavailable.
2. If no exact match, try alt spellings, partial matches, remixes.
3. When listing songs include artist, album, playlist if relevant.

Message history follows.
"""
    return music_assistant_prompt

def music_assistant(state: State):
    """The reasoning node for the music assistant. Generates tool calls or a final answer."""
    memory = state.get("loaded_memory", "None")
    music_assistant_prompt = generate_music_assistant_prompt(memory)

    response = llm_with_music_tools.invoke([SystemMessage(content=music_assistant_prompt)] + state["messages"])
    return {"messages": [response]}

def should_continue(state: State):
    """Return 'continue' if the LLM asked for a tool call, else 'end'."""
    last_message = state["messages"][-1]
    return "continue" if last_message.tool_calls else "end"

def create_music_agent_graph():
    """Build and compile the music-catalog sub-agent graph."""
    graph = StateGraph(State)

    graph.add_node("music_assistant", music_assistant)
    graph.add_node("music_tool_node", music_tool_node)

    graph.add_edge(START, "music_assistant")
    graph.add_conditional_edges("music_assistant",
                                should_continue,
                                {"continue": "music_tool_node", "end": END})
    graph.add_edge("music_tool_node", "music_assistant")
	
    grp = graph.compile(name="music_catalog_subagent", checkpointer=checkpointer, store=store).with_config({"messages_key": "messages"})
    
    return grp 
