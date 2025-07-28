from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Initialize memory components
checkpointer = MemorySaver()  # For short-term memory/state management
store = InMemoryStore()  # For long-term memory
