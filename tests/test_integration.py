import pytest
from unittest.mock import patch
from langchain_core.messages import HumanMessage, AIMessage
from workflow import multi_agent_final_graph

@pytest.mark.asyncio
async def test_verification_and_supervisor_flow():
    """Test the full graph flow from verification to supervisor handoff."""
    config = {“configurable”: {“thread_id”: “test-integration-thread”}}
    
    # 1. First invocation with an identifier
    input1 = {"messages": [HumanMessage(content="Hello my email is found@example.com")]}
    
    # Mock the call to the database lookup
    with patch('workflow.get_customer_id_from_identifier', return_value=123) as mock_lookup:
        # Mock the supervisor so it just returns a message instead of calling an agent
        with patch('workflow.supervisor_prebuilt_workflow.invoke', return_value={"messages": [AIMessage(content="Supervisor speaking.")]}) as mock_supervisor:
            final_state = await multi_agent_final_graph.ainvoke(input1, config)
    
            # Assertions
            mock_lookup.assert_called_once_with("found@example.com")
            mock_supervisor.assert_called_once()
    
            # Check that the final message is from the supervisor
            last_message = final_state['messages'][-1]
            assert isinstance(last_message, AIMessage)
            assert last_message.content == "Supervisor speaking."
