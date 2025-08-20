import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_core.retrievers import BaseRetriever
from langchain.memory import ConversationSummaryMemory
from config import PERSIST_DIR, EMBEDDING_MODEL, LLM_MODEL

load_dotenv()


def load_chroma():
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    if not os.path.exists(os.path.join(PERSIST_DIR, "chroma.sqlite3")):
        class DummyRetriever(BaseRetriever):
            def _get_relevant_documents(self, query):
                return []

            async def _aget_relevant_documents(self, query):
                return []

        return DummyRetriever()

    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    return vectordb


def create_qa_chain():
    db = load_chroma()
    if isinstance(db, Chroma):
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 2})
    else:
        retriever = db
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
