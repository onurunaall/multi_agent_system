import pytest
import asyncio
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from workflow import multi_agent_final_graph


class TestIntegration:
    """Integration test cases."""
    
    @pytest.mark.asyncio
    @patch('workflow.get_customer_id_from_identifier')
    @patch('workflow.structured_llm')
    async def test_verification_flow(self, mock_structured_llm, mock_get_customer):
        """Test the verification flow end-to-end."""
        # Mock the structured LLM to extract identifier
        mock_parsed = MagicMock()
        mock_parsed.identifier = "test@example.com"
        mock_structured_llm.invoke.return_value = mock_parsed
        
        # Mock customer lookup
        mock_get_customer.return_value = 123
        
        # Create initial state
        config = {"configurable": {"thread_id": "test-thread"}}
        
        # Invoke the graph
        result = await multi_agent_final_graph.ainvoke(
            {"messages": [HumanMessage(content="My email is test@example.com")]},
            config
        )
        
        assert result["customer_id"] == 123
        assert any("verified" in str(msg.content).lower() for msg in result["messages"] if hasattr(msg, 'content'))
