from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.storage import LocalFileStore
from dotenv import load_dotenv
import os

load_dotenv()

# Create the storage directory if it doesn't exist
if not os.path.exists("./storage"):
    os.makedirs("./storage")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Initialize memory components
checkpointer = MemorySaver()
store = LocalFileStore("./storage")