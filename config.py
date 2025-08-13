from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.storage import LocalFileStore
from sqlalchemy import create_engine
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

# Create the storage directory if it doesn't exist
STORAGE_DIR = "./storage"
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

# Initialize memory components
checkpointer = MemorySaver()
store = LocalFileStore(STORAGE_DIR)

# Database Configuration
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
    raise ValueError("One or more database environment variables are not set.")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)
db = SQLDatabase(engine)

# Vector Store Configuration
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory=f"{STORAGE_DIR}/chroma_db",
    embedding_function=embedding_function
)
vector_retriever = vector_store
