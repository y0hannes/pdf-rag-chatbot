from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_DIR = "data"
PERSIST_DIR = "chroma_db"


def load_pdfs(data_dir):
    docs = []
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            print(f"Loading: {filepath}")
            loader = PyPDFLoader(filepath)
            docs.extend(loader.load())
    return docs


def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def build_chroma_index(chunks, persist_dir):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print(f"Saved ChromaDB to: {persist_dir}")


if __name__ == "__main__":
    documents = load_pdfs(DATA_DIR)
    chunks = chunk_documents(documents)
    build_chroma_index(chunks, PERSIST_DIR)
