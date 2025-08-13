import pytest
import asyncio
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage

@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="Test response")
    llm.with_structured_output.return_value = llm
    return llm


@pytest.fixture
def mock_store():
    """Mock store for testing."""
    store = MagicMock()
    store.get.return_value = None
    store.put.return_value = None
    return store


@pytest.fixture
def mock_checkpointer():
    """Mock checkpointer for testing."""
    return MagicMock()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
