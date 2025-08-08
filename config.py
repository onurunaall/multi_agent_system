from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.storage import LocalFileStore
from sqlalchemy import create_engine
from langchain_community.utilities.sql_database import SQLDatabase
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

# Database Configuration
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
    raise ValueError("One or more database environment variables are not set.")

# Construct the PostgreSQL connection string
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create the SQLAlchemy engine and LangChain SQLDatabase wrapper
engine = create_engine(DATABASE_URL)
db = SQLDatabase(engine)
