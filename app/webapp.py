import streamlit as st
import asyncio
import os
import shutil
from datetime import datetime
import uuid
import shutil
import threading
import time as time_module
from shared import create_qa_chain
from ingest import main as ingest_main
from config import (
    DATA_DIR, PERSIST_DIR, SESSION_DATA_BASE, SESSION_PERSIST_BASE, 
    get_conversation_paths, get_session_base_paths
)
from keep_alive import start_keep_alive
from cleanup import cleanup_old_sessions

# Initialize Session ID (persists across reruns in same browser session)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize conversation ID tracker
if "conversation_ids" not in st.session_state:
    st.session_state.conversation_ids = {}

# Helper function to get conversation ID
def get_conversation_id(conversation_name):
    """Get or create a unique ID for a conversation."""
    if conversation_name not in st.session_state.conversation_ids:
        st.session_state.conversation_ids[conversation_name] = str(uuid.uuid4())
    return st.session_state.conversation_ids[conversation_name]

# Background cleanup thread (runs once per app instance)
if "cleanup_thread_started" not in st.session_state:
    def periodic_cleanup():
        """Run cleanup every 30 minutes."""
        while True:
            time_module.sleep(1800)  # 30 minutes
            try:
                cleanup_old_sessions()
            except Exception as e:
                print(f"Background cleanup error: {e}")
    
    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()
    st.session_state.cleanup_thread_started = True

# Start keep-alive pinger
start_keep_alive()


# Event Loop 
def get_or_create_eventloop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


get_or_create_eventloop()


# Caching 
@st.cache_resource
def load_qa_chain(_persist_dir):
    """Load QA chain for a specific conversation. Use _ prefix to exclude from hash."""
    return create_qa_chain(_persist_dir)


def generate_title(query):
    return query[:50] + "..." if len(query) > 50 else query


# Page Config
st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
st.title("📄 PDF RAG Chatbot")


# Sidebar
with st.sidebar:
    # Conversation Management (moved to top)
    st.header("💬 Conversations")
    if "conversations" not in st.session_state:
        st.session_state.conversations = {"Conversation 1": []}
        st.session_state.active_conversation = "Conversation 1"

    conversation_list = list(st.session_state.conversations.keys())
    selected_conv = st.radio(
        "Select a conversation",
        conversation_list,
        index=conversation_list.index(st.session_state.active_conversation),
    )
    st.session_state.active_conversation = selected_conv
    
    # Get current conversation paths
    current_conv_id = get_conversation_id(st.session_state.active_conversation)
    current_data_dir, current_persist_dir, current_log_file = get_conversation_paths(
        st.session_state.session_id, current_conv_id
    )
    
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if not os.path.exists(current_data_dir):
            os.makedirs(current_data_dir)

        file_path = os.path.join(current_data_dir, uploaded_file.name)

        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Save upload timestamp in session
            st.session_state[f"{uploaded_file.name}_uploaded"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            with st.spinner("Ingesting PDF..."):
                ingest_main(data_dir=current_data_dir, persist_dir=current_persist_dir, log_file=current_log_file)
                st.cache_resource.clear()
            st.success("PDF ingested successfully!")
        else:
            st.info("File already exists. Using existing data.")

    # PDF List
    st.subheader("📂 Uploaded PDFs")
    if os.path.exists(current_data_dir) and os.listdir(current_data_dir):
        for filename in os.listdir(current_data_dir):
            file_path = os.path.join(current_data_dir, filename)
            if filename.lower().endswith(".pdf"):
                file_stats = os.stat(file_path)
                file_size = round(file_stats.st_size / 1024, 2)  # KB
                uploaded_date = st.session_state.get(f"{filename}_uploaded", None)

                with st.expander(f"{filename} ({file_size} KB)"):
                    if uploaded_date:
                        st.caption(f"Uploaded on: {uploaded_date}")
                    if st.button(f"🗑 Delete {filename}", key=f"delete_{filename}"):
                        os.remove(file_path)
                        # also remove from processed log if needed
                        if os.path.exists(current_log_file):
                            with open(current_log_file, "r") as f:
                                lines = f.readlines()
                            with open(current_log_file, "w") as f:
                                f.writelines([l for l in lines if filename not in l])
                        st.success(f"Deleted {filename}")
                        st.rerun()
    else:
        st.info("No PDFs uploaded yet.")

    # Conversation Actions
    if st.button("➕ New Conversation"):
        new_name = f"Conversation {len(st.session_state.conversations) + 1}"
        st.session_state.conversations[new_name] = []
        st.session_state.active_conversation = new_name
        st.rerun()

    if st.button("🗑 Clear Current Conversation"):
        # Clear chat history
        st.session_state.conversations[st.session_state.active_conversation] = []
        # Delete conversation data
        if os.path.exists(current_data_dir):
            shutil.rmtree(current_data_dir)
        if os.path.exists(current_persist_dir):
            shutil.rmtree(current_persist_dir)
        st.cache_resource.clear()
        st.rerun()

    if st.button("♻️ Reset All Conversations"):
        # Clear all session data
        session_data_base, session_persist_base = get_session_base_paths(st.session_state.session_id)
        if os.path.exists(session_data_base):
            shutil.rmtree(session_data_base)
        if os.path.exists(session_persist_base):
            shutil.rmtree(session_persist_base)
        st.session_state.clear()
        st.cache_resource.clear()
        st.rerun()




# Main Chat Section
qa_chain = load_qa_chain(current_persist_dir)
messages = st.session_state.get("conversations", {"Conversation 1": []}).get(
    st.session_state.get("active_conversation", "Conversation 1"), []
)

# Display chat history with timestamps
if len(messages) > 10:
    with st.expander("📜 Conversation History (collapsed)", expanded=False):
        for message in messages:
            with st.chat_message(message["role"]):
                ts = message.get("timestamp", None)
                if ts:
                    st.caption(f"🕒 {ts}")
                st.markdown(message["content"])
else:
    for message in messages:
        with st.chat_message(message["role"]):
            ts = message.get("timestamp", None)
            if ts:
                st.caption(f"🕒 {ts}")
            st.markdown(message["content"])

# Chat input
if query := st.chat_input("Ask a question about your PDF:"):
    # Guard: No PDFs uploaded yet
    if not os.path.exists(current_data_dir) or not any(
        f.endswith(".pdf") for f in os.listdir(current_data_dir)
    ):
        st.warning("⚠️ Please upload a PDF before asking questions.")
    else:
        if not messages:
            new_title = generate_title(query)
            if "conversations" not in st.session_state:
                st.session_state.conversations = {}
            st.session_state.conversations[new_title] = st.session_state.conversations.pop(
                st.session_state.get("active_conversation", "Conversation 1"), []
            )
            st.session_state.active_conversation = new_title

        messages.append({
            "role": "user",
            "content": query,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        with st.chat_message("user"):
            st.caption(f"🕒 {messages[-1]['timestamp']}")
            st.markdown(query)

        with st.spinner("Thinking..."):
            try:
                result = asyncio.run(qa_chain.ainvoke(query))
                response = result["answer"]

                messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                with st.chat_message("assistant"):
                    st.caption(f"🕒 {messages[-1]['timestamp']}")
                    st.markdown(response)

                    with st.expander("Show sources"):
                        for doc in result.get("source_documents", []):
                            metadata_str = ", ".join(
                                f"{k}: {v}" for k, v in doc.metadata.items()
                            )
                            st.markdown(f"**{metadata_str}**")
                            st.write(doc.page_content[:500] + "...")
            except Exception as e:
                st.error(f"❌ Error during query: {e}")
