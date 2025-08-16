import streamlit as st
import asyncio
import os
from shared import create_qa_chain
from ingest import main as ingest_main

def get_or_create_eventloop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

get_or_create_eventloop()

@st.cache_resource
def load_qa_chain():
    return create_qa_chain()

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
st.title("📄 PDF RAG Chatbot")

with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        file_path = os.path.join(data_dir, uploaded_file.name)
        
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Ingesting PDF..."):
                ingest_main()
                st.cache_resource.clear()
            st.success("PDF ingested successfully!")
        else:
            st.info("File already exists. Using existing data.")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.cache_resource.clear()
        st.rerun()


qa_chain = load_qa_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question about your PDF:"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Thinking..."):
        result = asyncio.run(qa_chain.ainvoke(query))
        response = result['result']
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

            with st.expander("Show sources"):
                for doc in result["source_documents"]:
                    metadata_str = ", ".join(f"{k}: {v}" for k, v in doc.metadata.items())
                    st.markdown(f'**{metadata_str}**')
                    st.write(doc.page_content[:500] + "...")