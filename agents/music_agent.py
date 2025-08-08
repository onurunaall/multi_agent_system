from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage

from config import llm, checkpointer, store, db
from schemas import State

# Find all albums by a specific artist.
@tool
def get_albums_by_artist(artist: str) -> str:
    """Get albums by an artist."""
    if not artist or not artist.strip():
        return "Artist name required"

    query = """
        SELECT "Album"."Title", "Artist"."Name"
        FROM "Album"
        JOIN "Artist" ON "Album"."ArtistId" = "Artist"."ArtistId"
        WHERE "Artist"."Name" ILIKE %(artist_name)s
    """
    return db.run(query, parameters={"artist_name": f"%{artist.strip()}%"})

# Get all the songs by a given artist.
@tool
def get_tracks_by_artist(artist: str) -> str:
    """Get songs by an artist (or similar artists)."""
    if not artist or not artist.strip():
        return "Artist name required"
    
    query = """
        SELECT "Track"."Name" AS SongName, "Artist"."Name" AS ArtistName
        FROM "Album"
        LEFT JOIN "Artist" ON "Album"."ArtistId" = "Artist"."ArtistId"
        LEFT JOIN "Track" ON "Track"."AlbumId" = "Album"."AlbumId"
        WHERE "Artist"."Name" ILIKE %(artist_name)s
    """
    return db.run(query, parameters={"artist_name": f"%{artist.strip()}%"})

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

    genre_name = genre.strip()
    
    query = """
        SELECT "Track"."Name" AS "Song", "Artist"."Name" AS "Artist"
        FROM "Track"
        JOIN "Album" ON "Track"."AlbumId" = "Album"."AlbumId"
        JOIN "Artist" ON "Album"."ArtistId" = "Artist"."ArtistId"
        JOIN "Genre" ON "Track"."GenreId" = "Genre"."GenreId"
        WHERE "Genre"."Name" ILIKE %(genre_name)s
    """
    
    results = db.run(query, parameters={"genre_name": f"%{genre_name}%"})
    
    if not results or results == "[]":
        return f"No songs found for the genre: {genre_name}"
    return results

@tool
def check_for_songs(song_title: str) -> str:
    """Check if a song exists by its name."""
    if not song_title or not song_title.strip():
        return "Song title required"
    
    query = """
        SELECT * FROM "Track"
        WHERE "Name" ILIKE %(song_title)s
    """
    return db.run(query, parameters={"song_title": f"%{song_title.strip()}%"})

music_tools = [get_albums_by_artist, get_tracks_by_artist, get_songs_by_genre, check_for_songs]

llm_with_music_tools = llm.bind_tools(music_tools)
music_tool_node = ToolNode(music_tools)

def generate_music_assistant_prompt() -> str:
    music_assistant_prompt =  """
You are one of several specialised assistants; your focus is the music-catalog. If the catalog is missing an artist's material, simply say so.

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
    music_assistant_prompt = generate_music_assistant_prompt()
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
