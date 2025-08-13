import pytest
import os
from unittest.mock import AsyncMock
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

@pytest.mark.asyncio
async def test_router_to_music_agent_flow(mocker):
    """Test the full graph flow from the router to the music agent."""
    from workflow import multi_agent_final_graph

    mock_llm_ainvoke = mocker.patch(
        "langchain_openai.ChatOpenAI.ainvoke",
        new_callable=AsyncMock,
        side_effect=[
            AIMessage(content="", tool_calls=[{"name": "search_for_music", "args": {"query": "classic rock"}, "id": "call_xyz"}]),
            AIMessage(content="Here is a classic rock song I found.")
        ]
    )

    mock_retriever = mocker.patch("agents.music_agent.vector_retriever.similarity_search",
                                 return_value=[Document(page_content="Track: Stairway to Heaven by Led Zeppelin")])

    config = {"configurable": {"thread_id": "integration-test"}}
    input_message = {"messages": [HumanMessage(content="Find me some classic rock")]}

    final_state = await multi_agent_final_graph.ainvoke(input_message, config)

    assert mock_llm_ainvoke.call_count == 2
    last_message = final_state["messages"][-1]
    assert "Here is a classic rock song I found" in last_message.content
