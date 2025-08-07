from typing import Optional
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph_supervisor import create_supervisor
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
import ast
import json

from config import llm, checkpointer, store
from database import db
from schemas import State, UserInput, UserProfile
from agents.invoice_agent import create_invoice_agent
from agents.music_agent import create_music_agent_graph

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

supervisor_graph = create_supervisor(agents=[invoice_agent, music_agent],
                                     output_mode="last_message",
                                     model=llm,
                                     prompt=supervisor_prompt,
                                     state_schema=State)

supervisor_prebuilt_workflow = supervisor_graph.compile()


def run_supervisor(state: State):
    """Wrapper function to run the supervisor with full state context."""
    return supervisor_prebuilt_workflow.invoke(state)

# Converting the supervisor function to a LangGraph node
supervisor_node = RunnableLambda(run_supervisor)


def get_customer_id_from_identifier(identifier: str) -> Optional[int]:
    """
    Resolve a customer identifier (ID, email, phone) to internal CustomerId.
    
    Supports three identifier types:
    - Numeric string: treated as direct customer ID
    - Phone number: starts with '+', looked up in Customer.Phone
    - Email address: contains '@', looked up in Customer.Email
    """
    if not identifier:
        return None

    # Direct customer ID - convert string to integer
    if identifier.isdigit():
        return int(identifier)

    try:
        # Phone number lookup - must start with '+'
        if identifier.startswith("+"):
            row = db.run("SELECT CustomerId FROM Customer WHERE Phone = ?;", (identifier,))
            if row:
                parsed = ast.literal_eval(row)
                if parsed:
                    return parsed[0][0] # Extract customer ID from result tuple

        # Email lookup - must contain '@' symbol
        if "@" in identifier:
            row = db.run("SELECT CustomerId FROM Customer WHERE Email = ?;", (identifier,))
            if row:
                parsed = ast.literal_eval(row)
                if parsed:
                    return parsed[0][0] # Extract customer ID from result tuple
    
    except Exception as e:
        print(f"Error looking up customer: {e}")
    return None


# Forced to output data matching the UserInput schema.
structured_llm = llm.with_structured_output(schema=UserInput)
structured_system_prompt = ("Extract the customer's identifier (ID, e-mail, phone) from the messages. If none is present, return an empty string.")


# Checks if a user is verified before letting the workflow continue.
def verify_info(state: State, config: RunnableConfig):
    """
    Verify customer identity or prompt for missing information.
    
    This is the entry point for all conversations - ensures we have a valid
    customer ID before proceeding to any support functionality.
    """
    # Skip verification if customer is already identified
    if state.get("customer_id") is None:
        # Guidance for the LLM when customer info is missing/invalid
        guidance = ("Before helping, you must verify the customer's identity "
                    "(ID, e-mail, or phone). If it's missing, ask. "
                    "If the supplied identifier is invalid, ask for a correction.")

        # Get the most recent message from the conversation history
        user_msg = state["messages"][-1]

        # Extract identifier from the most recent user message
        parsed = structured_llm.invoke([SystemMessage(content=structured_system_prompt), user_msg])
        identifier = parsed.identifier

        # Attempt to resolve the identifier to a customer ID
        customer_id = (get_customer_id_from_identifier(identifier) if identifier else None)

        if customer_id:
            confirm = SystemMessage(content=f"Account verified. Customer ID: {customer_id}.")
            return {"customer_id": customer_id, "messages": [confirm]}
        else:
            follow_up = llm.invoke([SystemMessage(content=guidance)] + state["messages"])
            return {"messages": [follow_up]}

    # Already verified
    return None


def human_input(state: State, config: RunnableConfig):
    """
    Interrupt the graph execution to await fresh user input.
    
    This creates a breakpoint in the workflow where we wait for the user
    to provide additional information (typically their customer identifier).
    """
    return interrupt("Please provide input.")


def should_interrupt(state: State, config: RunnableConfig):
    """Branch: continue if verified else interrupt."""
    action = "continue" if state.get("customer_id") is not None else "interrupt"
    return action


def format_user_memory(user_data_bytes: Optional[bytes]) -> str:
    if not user_data_bytes:
        return ""
    
    # Decode bytes back to dictionary structure
    user_data = json.loads(user_data_bytes.decode())
    
    # Extract and format music preferences if they exist
    if "memory" in user_data:
        profile = UserProfile.model_validate(user_data["memory"])
        if profile.music_preferences:
            return f"Music Preferences: {', '.join(profile.music_preferences)}"
    
    return ""


def load_memory(state: State, config: RunnableConfig, store: BaseStore):
    """
    Load any existing user preferences from persistent storage.
    
    Retrieves the customer's saved profile and formats it for use
    in subsequent agent interactions.
    """
    uid = state["customer_id"]
    key = f"memory_profile_{uid}"
    
    # Attempt to retrieve existing profile from store
    entries = store.mget([key])
    entry_value = entries[0] if entries else None
    
    formatted = format_user_memory(entry_value) if entry_value else ""
    return {"loaded_memory": formatted}


create_memory_prompt = """
Analyse the conversation and update the customer's memory profile.

Fields:
- customer_id
- music_preferences

If no new info, keep existing values unchanged.

Conversation:
{conversation}

Existing profile:
{memory_profile}
"""


def create_memory(state: State, config: RunnableConfig, store: BaseStore):
    uid = str(state["customer_id"])
    key = f"memory_profile_{uid}"

    entries = store.mget([key])
    entry_value_bytes = entries[0] if entries else None

    current_pref = ""
    if entry_value_bytes:
        entry_value = json.loads(entry_value_bytes.decode())
        if "memory" in entry_value:
            prof = UserProfile.model_validate(entry_value["memory"])
            current_pref = ", ".join(prof.music_preferences or [])

    # Generate updated profile using LLM analysis
    sys = SystemMessage(content=create_memory_prompt.format(conversation=state["messages"], memory_profile=current_pref))
    new_profile = llm.with_structured_output(UserProfile).invoke([sys])

    # Serialize the dictionary to a JSON string and then encode to bytes
    data_to_store = {"memory": new_profile.model_dump()}
    bytes_to_store = json.dumps(data_to_store).encode('utf-8')
    
    store.mset([(key, bytes_to_store)])


multi_agent_final = StateGraph(State)
multi_agent_final.add_node("verify_info", verify_info)
multi_agent_final.add_node("human_input", human_input)
multi_agent_final.add_node("load_memory", load_memory)
multi_agent_final.add_node("supervisor", supervisor_node)
multi_agent_final.add_node("create_memory", create_memory)

multi_agent_final.add_edge(START, "verify_info")

multi_agent_final.add_conditional_edges("verify_info",
                                        should_interrupt,
                                        {"continue": "load_memory", "interrupt": "human_input"})

multi_agent_final.add_edge("human_input", "verify_info")
multi_agent_final.add_edge("load_memory", "supervisor")
multi_agent_final.add_edge("supervisor", "create_memory")
multi_agent_final.add_edge("create_memory", END)

multi_agent_final_graph = multi_agent_final.compile(name="multi_agent_verify",
                                                    checkpointer=checkpointer,
                                                    store=store)