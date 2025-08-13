import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage
from workflow import RouteQuery

class TestRouter:
    """Test cases for the main workflow router."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("message_content, destination, should_call_llm",
                             [("I need my last invoice.", "verify_customer", False),
                              ("Can you find songs by Queen?", "music", True),
                              ("Thanks!", "end", True)])
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("langchain_openai.ChatOpenAI.ainvoke", new_callable=AsyncMock)
    async def test_router_logic(self, mock_llm_ainvoke, message_content, destination, should_call_llm):
        """Test that the router correctly directs queries based on content."""
        from workflow import router

        if should_call_llm:
            mock_llm_ainvoke.return_value = AIMessage(
                content="",
                tool_calls=[{
                    "name": "RouteQuery",
                    "args": {"destination": destination},
                    "id": "test_call"
                }]
            )

        state = {"messages": [HumanMessage(content=message_content)]}
        result = await router(state)

        assert result == destination
        
        if should_call_llm:
            mock_llm_ainvoke.assert_called_once()
        else:
            mock_llm_ainvoke.assert_not_called()
