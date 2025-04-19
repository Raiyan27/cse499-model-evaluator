import os
from dotenv import load_dotenv

load_dotenv()

# --- Constants ---
DATA_DIR = "./data"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "Legal_data_test" # Use a different name for testing maybe
RAG_GRAPH_IMAGE_PATH = "rag_agent_graph.png"
CLARIFICATION_GRAPH_IMAGE_PATH = "clarification_agent_graph.png"

# Embedding & Vector Store
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
EMBEDDING_BATCH_SIZE = 100 # Adjust if needed for testing environment
MAX_DOCS_PER_QUERY = 5
MINIMUN_RETRIVAL_SCORE = 0.4 # Adjust based on testing

# LLM Models
CLARIFICATION_ASSESSMENT_MODEL = "gpt-4o-mini"
CLARIFICATION_QUESTION_MODEL = "gpt-4o-mini"
RAG_LLM_MODEL = "gpt-4o-mini"
REWRITER_LLM_MODEL = "gpt-4o-mini" # Can be same or different

# Agent Settings
MAX_QUERY_CLARIFICATION_TURNS = 5 # Lower for testing? Or keep original
TAVILY_MAX_RESULTS = 3
TAVILY_SEARCH_DEPTH = 'advanced'
TAVILY_MAX_TOKENS = 10000 # Max tokens for Tavily search result content


# API Keys (Loaded from .env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Basic validation
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY environment variable not set.")