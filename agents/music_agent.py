from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AIMessage
import json

from config import llm, checkpointer, store, db, vector_retriever
from schemas import State


@tool
def search_for_music(query: str) -> str:
    """Search for tracks, artists, albums, or genres in the music catalog.

    Use this for any music-related questions such as finding songs, getting recommendations,
    or checking for artists.

    Args:
        query (str): The search query for music content.

    Returns:
        str: JSON formatted search results or a message indicating no results found.
    """
    retrieved_docs = vector_retriever.similarity_search(query, k=5)
    
    # Handle case when no results are found
    if not retrieved_docs:
        return "No results found in the music catalog for that query."
    
    # Format the results for better readability
    results = [doc.page_content for doc in retrieved_docs]
    formatted_results = json.dumps(results, indent=2)
    return formatted_results


@tool
def get_albums_by_artist(artist: str) -> str:
    """Get all albums by a specific artist.

    Use this when a user explicitly asks for an artist's albums.

    Args:
        artist (str): The name of the artist to search for.

    Returns:
        str: List of album titles or an error message.
    """
    # Validate artist name is provided
    if not artist or not artist.strip():
        return "Artist name required"

    query = """
            SELECT "Album"."Title"
            FROM "Album"
            JOIN "Artist" ON "Album"."ArtistId" = "Artist"."ArtistId"
            WHERE "Artist"."Name" ILIKE %(artist_name)s
            """
    
    search_pattern = f"%{artist.strip()}%"
    parameters = {"artist_name": search_pattern}
    return db.run(query, parameters=parameters)


# Available tools for the music agent
music_tools = [search_for_music, get_albums_by_artist]

# Create tool node and bind tools to LLM
music_tool_node = ToolNode(music_tools)
llm_with_music_tools = llm.bind_tools(music_tools)


async def music_assistant_agent(state: State):
    """The reasoning node for the music assistant. Generates tool calls or a final answer.

    Args:
        state (State): The current conversation state containing messages.

    Returns:
        dict: Updated state with the assistant's response message.
    """
    music_assistant_prompt = """
You are a specialized assistant for a digital music store. Your focus is the music catalog.
Your goal is to answer music-related questions by using the available tools.
If the catalog is missing an artist's material, simply say so.

Use the `search_for_music` tool for general queries, recommendations, or finding songs.
Only use `get_albums_by_artist` when a user specifically asks for an artist's discography or albums.
    """
    
    messages = [SystemMessage(content=music_assistant_prompt)] + state["messages"]
    response = await llm_with_music_tools.ainvoke(messages)
    return {"messages": [response]}


def should_continue_music(state: State):
    """Determine if the music agent should continue processing or end.

    Args:
        state (State): The current conversation state.

    Returns:
        str: "continue" if tool calls are needed, "end" if processing is complete.
    """
    last_message = state["messages"][-1]
    
    # Check if the last message requires tool calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue"
    return "end"


def create_music_agent_graph():
    """Build and compile the music-catalog sub-agent graph.

    Returns:
        The compiled state graph for the music agent.
    """
    graph = StateGraph(State)
    
    # Add nodes to the graph
    graph.add_node("agent", music_assistant_agent)
    graph.add_node("tools", music_tool_node)
    
    # Define the flow of the graph
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent",
                                should_continue_music,
                                {"continue": "tools", "end": END})
    graph.add_edge("tools", "agent")
    
    return graph.compile(checkpointer=checkpointer)
