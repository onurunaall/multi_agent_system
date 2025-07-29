from typing import Optional
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph_supervisor import create_supervisor
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
import ast

from config import llm, checkpointer, store
from database import db
from schemas import State, UserInput, UserProfile
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
# The create_supervisor function returns a graph, which must be compiled.
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

    if identifier.isdigit():
        return int(identifier)

    try:
        if identifier.startswith("+"):
            row = db.run("SELECT CustomerId FROM Customer WHERE Phone = ?;", (identifier,))
            if row:
                parsed = ast.literal_eval(row)
                if parsed:
                    return parsed[0][0]

        if "@" in identifier:
            row = db.run("SELECT CustomerId FROM Customer WHERE Email = ?;", (identifier,))
            if row:
                parsed = ast.literal_eval(row)
                if parsed:
                    return parsed[0][0]
    except Exception as e:
        print(f"Error looking up customer: {e}")
    return None

# Forced to output data matching the UserInput schema.
structured_llm = llm.with_structured_output(schema=UserInput)
structured_system_prompt = ("Extract the customer's identifier (ID, e-mail, phone) from the messages. If none is present, return an empty string.")

# Checks if a user is verified before letting the workflow continue.
def verify_info(state: State, config: RunnableConfig):
    """Verify customer identity or prompt for it."""
    if state.get("customer_id") is None:
        guidance = (
          "Before helping, you must verify the customer's identity "
          "(ID, e-mail, or phone). If it's missing, ask. "
          "If the supplied identifier is invalid, ask for a correction."
        )

        # Get the most recent message from the conversation history
        user_msg = state["messages"][-1]
        parsed = structured_llm.invoke([SystemMessage(content=structured_system_prompt), user_msg])
        identifier = parsed.identifier

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
    """Interrupt the graph and await fresh user input."""
    return interrupt("Please provide input.")

def should_interrupt(state: State, config: RunnableConfig):
    """Branch: continue if verified else interrupt."""
    action = "continue" if state.get("customer_id") is not None else "interrupt"
    return action

def format_user_memory(user_data):
    if user_data and "memory" in user_data:
        profile: UserProfile = user_data["memory"]
        if profile.music_preferences:
            return f"Music Preferences: {', '.join(profile.music_preferences)}"
    return ""

def load_memory(state: State, config: RunnableConfig, store: BaseStore):
    """Load saved user preferences (if any)."""
    uid = state["customer_id"]
    key = f"memory_profile_{uid}"
    # The 'mget' method returns a list of values for the given keys
    entries = store.mget([key])
    # We are only getting one key, so we take the first element
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
    entry_value = entries[0] if entries else None
    current_pref = ""
    if entry_value and "memory" in entry_value:
        prof: UserProfile = entry_value["memory"]
        current_pref = ", ".join(prof.music_preferences or [])

    sys = SystemMessage(content=create_memory_prompt.format(conversation=state["messages"], memory_profile=current_pref))
    new_profile = llm.with_structured_output(UserProfile).invoke([sys])
    store.mset([(key, {"memory": new_profile})])

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

multi_agent_final_graph = multi_agent_final.compile(
    name="multi_agent_verify",
    checkpointer=checkpointer,
    store=store,
)