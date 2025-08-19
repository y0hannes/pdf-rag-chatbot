import os

# Project Root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Data and ChromaDB Directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PERSIST_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
PROCESSED_FILES_LOG = os.path.join(PERSIST_DIR, "processed_files.log")

# Models
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "llama3-8b-8192"
