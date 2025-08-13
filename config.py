from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.storage import LocalFileStore
from sqlalchemy import create_engine
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
import sys

load_dotenv()

# Validate required environment variables
REQUIRED_ENV_VARS = {
    "OPENAI_API_KEY": "OpenAI API key is required",
    "DB_USER": "Database user is required", 
    "DB_PASSWORD": "Database password is required",
    "DB_HOST": "Database host is required",
    "DB_PORT": "Database port is required", 
    "DB_NAME": "Database name is required"
}

missing_vars = []
for var, description in REQUIRED_ENV_VARS.items():
    if not os.getenv(var):
        missing_vars.append(f"  - {var}: {description}")

if missing_vars:
    print("ERROR: Missing required environment variables:")
    print("\n".join(missing_vars))
    print("\nPlease set these variables in your .env file.")
    sys.exit(1)

# Create storage directory
STORAGE_DIR = "./storage"
if not os.path.exists(STORAGE_DIR):
    try:
        os.makedirs(STORAGE_DIR)
    except OSError as e:
        print(f"ERROR: Could not create storage directory: {e}")
        sys.exit(1)

# Initialize LLM with error handling
try:
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
except Exception as e:
    print(f"ERROR: Failed to initialize OpenAI client: {e}")
    sys.exit(1)

# Initialize memory components
checkpointer = MemorySaver()
store = LocalFileStore(STORAGE_DIR)

# Database Configuration with validation
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD") 
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

try:
    from sqlalchemy import text
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
    db = SQLDatabase(engine)
    
    # Test connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("Database connection established successfully")
    
except Exception as e:
    print(f"ERROR: Database connection failed: {e}")
    print("Please check your database configuration and ensure the database is running.")
    sys.exit(1)

# Vector Store Configuration with error handling
try:
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory=f"{STORAGE_DIR}/chroma_db",
        embedding_function=embedding_function
    )
    vector_retriever = vector_store
    print("Vector store initialized successfully")
    
except Exception as e:
    print(f"ERROR: Vector store initialization failed: {e}")
    sys.exit(1)