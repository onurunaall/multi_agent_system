from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END

from config import llm, checkpointer
from schemas import State
from agents.invoice_agent import create_invoice_agent
from agents.music_agent import create_music_agent_graph

# Create agents
invoice_agent_runnable = create_invoice_agent()
music_agent_runnable = create_music_agent_graph()

class RouteQuery(BaseModel):
    """Route a user query to the most relevant agent."""
    destination: Literal["invoice", "music", "end"] = Field(
        description="Given the user's query, route them to the most relevant agent: 'invoice' for invoice questions, 'music' for music catalog questions. If the query is a simple greeting or farewell, route to 'end'."
    )

# Create the router with a specific prompt
structured_llm_router = llm.with_structured_output(RouteQuery)
router_prompt = (
    "You are an expert at routing user queries to a specialized agent or a final response handler. "
    "Based on the user's request, choose the appropriate destination."
)

async def router(state: State) -> Literal["invoice", "music", "end"]:
    """The router node. Decides where to send the user's query next."""
    route = await structured_llm_router.ainvoke([
        ("system", router_prompt),
        ("human", state["messages"][-1].content),
    ])
    if isinstance(route, AIMessage) and route.tool_calls:
        return route.tool_calls[0]['args']['destination']
    return route.destination


async def final_answer(state: State):
    """Generates a final response for simple queries that don't need an agent."""
    last_message = state["messages"][-1].content
    response = await llm.ainvoke([
        ("system", "You are a friendly customer support assistant."),
        ("human", last_message)
    ])
    return {"messages": [response]}


# --- Corrected Graph Definition ---

# Build the graph
workflow = StateGraph(State)

# Add the nodes for the agents and the final answer handler
workflow.add_node("invoice_agent", invoice_agent_runnable)
workflow.add_node("music_agent", music_agent_runnable)
workflow.add_node("final_answer", final_answer)

# The START node will now directly use the router function to decide the first step
workflow.add_conditional_edges(
    "__start__",
    router,
    {
        "invoice": "invoice_agent",
        "music": "music_agent",
        "end": "final_answer",
    },
)

# Add the edges that lead to the end of the conversation
workflow.add_edge("invoice_agent", END)
workflow.add_edge("music_agent", END)
workflow.add_edge("final_answer", END)

# Compile the final graph
multi_agent_final_graph = workflow.compile(checkpointer=checkpointer)
