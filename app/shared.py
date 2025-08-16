import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PERSIST_DIR = os.path.join(PROJECT_ROOT, "chroma_db")

def load_chroma():

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    return vectordb

def create_qa_chain():
    db = load_chroma()
    retriever = db.as_retriever(
        search_type="mmr", search_kwargs={"k": 2})

    template = """You are a helpful assistant answering questions strictly using the provided context.
If the context does not contain the answer, say "I don't know based on the document."
Keep answers concise and factual.

Context:
{context}

Question:
{question}

Answer:"""
    prompt = PromptTemplate(template=template, input_variables=[
                            "context", "question"])

    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa