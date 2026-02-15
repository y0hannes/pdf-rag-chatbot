import os

# Project Root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data and ChromaDB Directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PERSIST_DIR = os.path.join(PROJECT_ROOT, "chroma_db")

# Ephemeral Session Directories
SESSION_DATA_BASE = os.path.join(DATA_DIR, "sessions")
SESSION_PERSIST_BASE = os.path.join(PERSIST_DIR, "sessions")

def get_conversation_paths(session_id: str, conversation_id: str):
    """Returns (data_dir, persist_dir, log_file) for a given conversation."""
    # Each conversation gets its own isolated storage
    conv_data_dir = os.path.join(SESSION_DATA_BASE, session_id, conversation_id)
    conv_persist_dir = os.path.join(SESSION_PERSIST_BASE, session_id, conversation_id)
    conv_log_file = os.path.join(conv_persist_dir, "processed_files.log")
    return conv_data_dir, conv_persist_dir, conv_log_file

def get_session_base_paths(session_id: str):
    """Returns base paths for a session (for cleanup)."""
    s_data_dir = os.path.join(SESSION_DATA_BASE, session_id)
    s_persist_dir = os.path.join(SESSION_PERSIST_BASE, session_id)
    return s_data_dir, s_persist_dir

# Models
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "llama-3.1-8b-instant"
