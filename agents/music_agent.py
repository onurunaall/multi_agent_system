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
def get_albums_by_artist(artist: str) -> str:
    """Get albums by an artist."""
    if not artist or not artist.strip():
        return "Artist name required"

    artist = artist.strip()
    # Escape single quotes for SQL
    safe_artist = artist.replace("'", "''")

    query = f"""
        SELECT "Album"."Title", "Artist"."Name"
        FROM "Album"
        JOIN "Artist" ON "Album"."ArtistId" = "Artist"."ArtistId"
        WHERE "Artist"."Name" LIKE '%{safe_artist}%'
    """
    return db.run(query)

# Get all the songs by a given artist.
@tool
def get_tracks_by_artist(artist: str) -> str:
    """Get songs by an artist (or similar artists)."""
    if not artist or not artist.strip():
        return "Artist name required"
    
    artist = artist.strip()
    # Escape single quotes for SQL
    safe_artist = artist.replace("'", "''")
    
    query = f"""
        SELECT "Track"."Name" AS SongName, "Artist"."Name" AS ArtistName
        FROM "Album"
        LEFT JOIN "Artist" ON "Album"."ArtistId" = "Artist"."ArtistId"
        LEFT JOIN "Track" ON "Track"."AlbumId" = "Album"."AlbumId"
        WHERE "Artist"."Name" LIKE '%{safe_artist}%'
    """
    return db.run(query)

# Find songs that belong to a specific genre.
@tool
def get_songs_by_genre(genre: str) -> str:
    """
    Fetch songs that match a specific genre.
    Args:
        genre (str): The genre of the songs to fetch.
    Returns:
        str: A list of songs that match the specified genre.
    """
    if not genre or not genre.strip():
        return "Genre name is required"
        
    genre = genre.strip()
    # Escape single quotes for SQL
    safe_genre = genre.replace("'", "''")
    
    # First get genre IDs
    genre_query = f"""SELECT "GenreId" FROM "Genre" WHERE "Name" LIKE '%{safe_genre}%'"""
    genre_ids_raw = db.run(genre_query)
    
    if not genre_ids_raw or genre_ids_raw == "[]":
        return f"No songs found for the genre: {genre}"

    try:
        genre_ids = ast.literal_eval(genre_ids_raw)
        if not genre_ids:
            return f"No songs found for the genre: {genre}"
        
        # Build the IN clause with genre IDs
        genre_id_values = [str(gid[0]) for gid in genre_ids]
        genre_ids_str = ", ".join(genre_id_values)

        songs_query = f"""
            SELECT "Track"."Name" AS Song, "Artist"."Name" AS Artist
            FROM "Track"
            LEFT JOIN "Album"  ON "Track"."AlbumId"  = "Album"."AlbumId"
            LEFT JOIN "Artist" ON "Album"."ArtistId" = "Artist"."ArtistId"
            WHERE "Track"."GenreId" IN ({genre_ids_str})
        """
        songs_raw = db.run(songs_query)

        if not songs_raw or songs_raw == "[]":
            return f"No songs found for the genre: {genre}"

        return songs_raw
    except Exception as e:
        return f"Error fetching songs for genre {genre}: {str(e)}"

@tool
def check_for_songs(song_title: str) -> str:
    """Check if a song exists by its name."""
    if not song_title or not song_title.strip():
        return "Song title required"
    
    # Escape single quotes for SQL
    safe_title = song_title.strip().replace("'", "''")
    
    query = f"""
        SELECT * FROM "Track"
        WHERE "Name" LIKE '%{safe_title}%'
    """
    return db.run(query)

music_tools = [get_albums_by_artist, get_tracks_by_artist, get_songs_by_genre, check_for_songs]

llm_with_music_tools = llm.bind_tools(music_tools)
music_tool_node = ToolNode(music_tools)

def generate_music_assistant_prompt(memory: str = "None") -> str:
    music_assistant_prompt =  f"""
You are one of several specialised assistants; your focus is the music-catalog. If the catalog is missing an artist's material, simply say so.  
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
