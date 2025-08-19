import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from app.config import PERSIST_DIR, EMBEDDING_MODEL, LLM_MODEL

load_dotenv()

def load_chroma():
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    return vectordb

def create_qa_chain():
    db = load_chroma()
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 2})

    template = """
    You are a helpful AI assistant. Answer the question based only on the context below.

    Context:
    {context}

    Question:
    {question}

    Answer:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["chat_history", "context", "question"]
    )

    llm = ChatGroq(model_name=LLM_MODEL, temperature=0)

    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        output_key="answer"
    )
    return qa
