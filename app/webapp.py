import streamlit as st
import asyncio
import os
import shutil
from datetime import datetime
from shared import create_qa_chain
from ingest import main as ingest_main
from config import DATA_DIR, PERSIST_DIR, PROCESSED_FILES_LOG


# ---------------- Event Loop ---------------- #
def get_or_create_eventloop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


get_or_create_eventloop()


# ---------------- Caching ---------------- #
@st.cache_resource
def load_qa_chain():
    return create_qa_chain()


def generate_title(query):
    return query[:50] + "..." if len(query) > 50 else query


# ---------------- Page Config ---------------- #
st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
st.title("📄 PDF RAG Chatbot")


# ---------------- Sidebar ---------------- #
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        file_path = os.path.join(DATA_DIR, uploaded_file.name)

        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Save upload timestamp in session
            st.session_state[f"{uploaded_file.name}_uploaded"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            with st.spinner("Ingesting PDF..."):
                ingest_main()
                st.cache_resource.clear()
            st.success("PDF ingested successfully!")
        else:
            st.info("File already exists. Using existing data.")

    # ---------------- PDF List ---------------- #
    st.subheader("📂 Uploaded PDFs")
    if os.path.exists(DATA_DIR) and os.listdir(DATA_DIR):
        for filename in os.listdir(DATA_DIR):
            file_path = os.path.join(DATA_DIR, filename)
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
                        if os.path.exists(PROCESSED_FILES_LOG):
                            with open(PROCESSED_FILES_LOG, "r") as f:
                                lines = f.readlines()
                            with open(PROCESSED_FILES_LOG, "w") as f:
                                f.writelines([l for l in lines if filename not in l])
                        st.success(f"Deleted {filename}")
                        st.rerun()
    else:
        st.info("No PDFs uploaded yet.")

    # ---------------- Conversation Management ---------------- #
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

    if st.button("➕ New Conversation"):
        new_name = f"Conversation {len(st.session_state.conversations) + 1}"
        st.session_state.conversations[new_name] = []
        st.session_state.active_conversation = new_name
        st.rerun()

    if st.button("🗑 Clear Current Conversation"):
        st.session_state.conversations[st.session_state.active_conversation] = []
        st.rerun()

    if st.button("♻️ Reset All Conversations"):
        st.session_state.clear()
        st.cache_resource.clear()
        st.rerun()

    if st.button("🔥 Delete All PDFs and Data"):
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        if os.path.exists(PROCESSED_FILES_LOG):
            os.remove(PROCESSED_FILES_LOG)
        st.session_state.clear()
        st.cache_resource.clear()
        st.rerun()


# ---------------- Main Chat Section ---------------- #
qa_chain = load_qa_chain()
messages = st.session_state.get("conversations", {"Conversation 1": []}).get(
    st.session_state.get("active_conversation", "Conversation 1"), []
)

# Display chat history
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if query := st.chat_input("Ask a question about your PDF:"):
    # Guard: No PDFs uploaded yet
    if not os.path.exists(DATA_DIR) or not any(
        f.endswith(".pdf") for f in os.listdir(DATA_DIR)
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

        messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Thinking..."):
            try:
                result = asyncio.run(qa_chain.ainvoke(query))
                response = result["answer"]

                messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
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
