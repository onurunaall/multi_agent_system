from typing import Optional
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph_supervisor import create_supervisor
from langchain_core.runnables import RunnableConfig
from sqlalchemy import text
import re

from config import llm, checkpointer, store, db, engine
from schemas import State, UserInput
from agents.invoice_agent import create_invoice_agent
from agents.music_agent import create_music_agent_graph

# Create agents
invoice_agent = create_invoice_agent()
music_agent = create_music_agent_graph()

# Supervisor prompt
supervisor_prompt = """
You are an expert customer-support assistant for a digital music store. You supervise two specialised sub-agents:
1. music_agent which looks up catalogue data and user music preferences.
2. invoice_agent which retrieves customer invoices and purchase history.

Given the conversation so far, decide which sub-agent should act next.
Multiple steps may be required to fully satisfy a request.
"""

# Create the supervisor
supervisor_graph = create_supervisor(
    agents=[invoice_agent, music_agent],
    output_mode="last_message",
    model=llm,
    prompt=supervisor_prompt,
    state_schema=State
)
supervisor_prebuilt_workflow = supervisor_graph.compile()


# Wrap the supervisor to pass the full state
def run_supervisor(state: State):
    return supervisor_prebuilt_workflow.invoke(state)


supervisor_node = RunnableLambda(run_supervisor)


def get_customer_id_from_identifier(identifier: str) -> Optional[int]:
    """Resolve a customer identifier (ID, e-mail, phone) to CustomerId."""
    if not identifier:
        return None

    identifier = identifier.strip()
    query = None
    params = {}

    # Prioritize email format
    if "@" in identifier:
        query = text('SELECT "CustomerId" FROM "Customer" WHERE "Email" = :identifier')
        params = {"identifier": identifier}
    # Check for phone-like patterns before assuming it's a numeric ID
    elif identifier.startswith("+") or (len(identifier) >= 10 and identifier.replace("-", "").isdigit()):
        query = text('SELECT "CustomerId" FROM "Customer" WHERE "Phone" = :identifier')
        params = {"identifier": identifier}
    # Finally, check if it's a simple numeric ID
    elif identifier.isdigit():
        query = text('SELECT "CustomerId" FROM "Customer" WHERE "CustomerId" = :id')
        params = {"id": int(identifier)}
    
    if query is None:
        return None

    try:
        with engine.connect() as connection:
            customer_id = connection.execute(query, params).scalar_one_or_none()
        return customer_id
    except Exception as e:
        print(f"Database error while looking up identifier '{identifier}': {e}")
        return None


def extract_identifier_from_message(message_content: str) -> str:
    """
    Extract customer identifier from message content.
    """
    # Look for email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, message_content)
    if email_match:
        return email_match.group(0)
    
    # Look for phone pattern
    phone_pattern = r'\+?[\d\s\-\(\)]{10,}'
    phone_match = re.search(phone_pattern, message_content)
    if phone_match:
        phone = phone_match.group(0).strip()
        phone = re.sub(r'[\s\-\(\)]', '', phone)
        if len(phone) >= 10:
            return phone
    
    # Look for customer ID
    id_patterns = [
        r'\b(?:my )?(?:id|ID|customer id|customer ID)[\s:]+(\d+)\b',
        r'\b(\d{1,4})\b'  # Just numbers
    ]
    
    for pattern in id_patterns:
        match = re.search(pattern, message_content, re.IGNORECASE)
        if match:
            return match.group(1) if match.groups() else match.group(0)
    
    return ""


def verify_info(state: State, config: RunnableConfig = None):
    """Verify customer identity or prompt for it."""
    # Handle state properly - it should be a dictionary
    if isinstance(state, str):
        print(f"ERROR: State is string instead of dict: {state}")
        return {"messages": [AIMessage(content="I'm experiencing a technical issue. Please try again.")]}
    
    # Access state as dictionary
    customer_id = state.get("customer_id", None) if hasattr(state, 'get') else state["customer_id"] if "customer_id" in state else None
    messages = state.get("messages", []) if hasattr(state, 'get') else state["messages"] if "messages" in state else []
    
    if customer_id is None or customer_id == "":
        # Get the most recent user message
        if messages:
            last_message = messages[-1]
            message_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
            
            # Try to extract identifier
            identifier = extract_identifier_from_message(message_content)
            
            if identifier:
                # Try to find customer
                found_customer_id = get_customer_id_from_identifier(identifier)
                
                if found_customer_id:
                    confirm_msg = f"Great! I've verified your account (Customer ID: {found_customer_id}). How can I help you today?"
                    return {
                        "customer_id": str(found_customer_id),
                        "messages": [AIMessage(content=confirm_msg)]
                    }
                else:
                    # Identifier provided but not found
                    error_msg = f"I couldn't find an account with '{identifier}'. Could you please double-check and provide your email address, phone number, or customer ID again?"
                    return {
                        "messages": [AIMessage(content=error_msg)]
                    }
            else:
                # No identifier found in message
                prompt_msg = "Hello! I'm here to help you with your music store account. To get started, I'll need to verify your identity. Could you please provide your email address, phone number, or customer ID?"
                return {
                    "messages": [AIMessage(content=prompt_msg)]
                }
        else:
            # No messages at all
            return {
                "messages": [AIMessage(content="Hello! Please provide your email, phone, or customer ID to get started.")]
            }
    
    # Already verified - must return a dict, not None
    return {}


def human_input(state: State, config: RunnableConfig = None):
    """Interrupt the graph and await fresh user input."""
    return interrupt("Please provide input.")


def should_interrupt(state: State, config: RunnableConfig = None):
    """Branch: continue if verified else interrupt."""
    if isinstance(state, str):
        return "interrupt"
    
    customer_id = state.get("customer_id", None) if hasattr(state, 'get') else state["customer_id"] if "customer_id" in state else None
    return "continue" if customer_id is not None and customer_id != "" else "interrupt"


# Build the graph
multi_agent_final = StateGraph(State)
multi_agent_final.add_node("verify_info", verify_info)
multi_agent_final.add_node("human_input", human_input)
multi_agent_final.add_node("supervisor", supervisor_node)

multi_agent_final.add_edge(START, "verify_info")
multi_agent_final.add_conditional_edges(
    "verify_info",
    should_interrupt,
    {"continue": "supervisor", "interrupt": "human_input"}
)
multi_agent_final.add_edge("human_input", "verify_info")
multi_agent_final.add_edge("supervisor", END)

# Compile the graph
multi_agent_final_graph = multi_agent_final.compile(
    name="multi_agent_verify",
    checkpointer=checkpointer,
    store=store,
)
