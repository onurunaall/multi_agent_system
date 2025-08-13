from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool

from config import llm, checkpointer, db
from schemas import State, UserInput
from agents.invoice_agent import create_invoice_agent
from agents.music_agent import create_music_agent_graph

# Create agents
invoice_agent_runnable = create_invoice_agent()
music_agent_runnable = create_music_agent_graph()

class RouteQuery(BaseModel):
    """Route a user query to the most relevant agent."""
    destination: Literal["verify_customer", "invoice", "music", "end"] = Field(
        description="Route to 'verify_customer' for invoice-related queries when customer_id is missing, "
                    "'invoice' for invoice questions with valid customer, 'music' for music catalog questions, "
                    "'end' for greetings/farewells."
    )

@tool
def verify_customer_identity(email_or_name: str) -> str:
    """Verify customer identity by email or name and return customer ID."""
    if not email_or_name or not email_or_name.strip():
        return "Please provide your email address or full name."
    
    query = """
        SELECT "CustomerId", "FirstName", "LastName", "Email"
        FROM "Customer"
        WHERE "Email" ILIKE %(identifier)s 
        OR (CONCAT("FirstName", ' ', "LastName")) ILIKE %(identifier)s
        LIMIT 1
    """
    
    result = db.run(query, parameters={"identifier": f"%{email_or_name.strip()}%"})
    
    if result and result != "[]":
        import json
        customer_data = json.loads(result)[0]
        return f"CUSTOMER_VERIFIED:{customer_data['CustomerId']}"
    
    return "Customer not found. Please check your email address or full name."

structured_llm_router = llm.with_structured_output(RouteQuery)
structured_llm_input = llm.with_structured_output(UserInput)

async def router(state: State) -> Literal["verify_customer", "invoice", "music", "end"]:
    """Route user queries to appropriate handlers."""
    last_message = state["messages"][-1].content.lower()
    
    # Direct routing for invoice queries
    invoice_keywords = ["invoice", "bill", "payment", "charge", "receipt", "purchase", "buy", "order"]
    needs_invoice = any(keyword in last_message for keyword in invoice_keywords)
    
    if needs_invoice and not state.get("customer_id"):
        return "verify_customer"
    elif needs_invoice and state.get("customer_id"):
        return "invoice"
    
    # Use LLM for other queries
    try:
        route = await structured_llm_router.ainvoke([
            ("system", "Route user queries: 'music' for songs/artists/albums, 'end' for greetings/farewells."),
            ("human", state["messages"][-1].content),
        ])
        
        if isinstance(route, AIMessage) and route.tool_calls:
            return route.tool_calls[0]['args']['destination']
        elif hasattr(route, 'destination'):
            return route.destination
        else:
            # Fallback routing
            music_keywords = ["music", "song", "artist", "album", "band", "track"]
            if any(keyword in last_message for keyword in music_keywords):
                return "music"
            return "end"
    except Exception:
        # Fallback routing on error
        music_keywords = ["music", "song", "artist", "album", "band", "track"]
        if any(keyword in last_message for keyword in music_keywords):
            return "music"
        return "end"

async def customer_verification(state: State) -> dict:
    """Handle customer identity verification for invoice queries."""
    verification_prompt = """You are a customer service agent. The user needs invoice information but must be verified first.

Ask for their email address or full name politely. If they provide it, use the verify_customer_identity tool to look them up."""

    messages = [
        ("system", verification_prompt),
        *[(msg.type, msg.content) for msg in state["messages"]]
    ]
    
    tools = [verify_customer_identity]
    llm_with_tools = llm.bind_tools(tools)
    
    response = await llm_with_tools.ainvoke(messages)
    
    updated_state = {"messages": [response]}
    
    # Only process tool calls if they exist
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_result = verify_customer_identity.invoke({
            "email_or_name": response.tool_calls[0]["args"]["email_or_name"]
        })
        
        if tool_result.startswith("CUSTOMER_VERIFIED:"):
            customer_id = tool_result.split(":")[1]
            updated_state["customer_id"] = customer_id
            success_msg = AIMessage(content="Great! I found your account. How can I help you with your invoices?")
            updated_state["messages"] = [success_msg]
    
    return updated_state

async def final_answer(state: State) -> dict:
    """Generate responses for simple queries."""
    response = await llm.ainvoke([
        ("system", "You are a friendly customer support assistant. Keep responses brief and helpful."),
        ("human", state["messages"][-1].content)
    ])
    return {"messages": [response]}

# Build workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("verify_customer", customer_verification)
workflow.add_node("invoice_agent", invoice_agent_runnable)
workflow.add_node("music_agent", music_agent_runnable)
workflow.add_node("final_answer", final_answer)

# Routing from start
workflow.add_conditional_edges(
    "__start__",
    router,
    {
        "verify_customer": "verify_customer",
        "invoice": "invoice_agent", 
        "music": "music_agent",
        "end": "final_answer",
    },
)

# After verification, route to invoice agent if customer_id exists
def route_after_verification(state: State) -> Literal["invoice_agent", "verify_customer"]:
    return "invoice_agent" if state.get("customer_id") else "verify_customer"

workflow.add_conditional_edges("verify_customer",
                               route_after_verification,
                               {
                                   "invoice_agent": "invoice_agent",
                                   "verify_customer": "verify_customer"})

# End edges
workflow.add_edge("invoice_agent", END)
workflow.add_edge("music_agent", END) 
workflow.add_edge("final_answer", END)

# Compile
multi_agent_final_graph = workflow.compile(checkpointer=checkpointer)