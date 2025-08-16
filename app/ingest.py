import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PERSIST_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
PROCESSED_FILES_LOG = os.path.join(PERSIST_DIR, "processed_files.log")


def get_processed_files():
    if not os.path.exists(PROCESSED_FILES_LOG):
        return set()
    with open(PROCESSED_FILES_LOG, "r") as f:
        return set(f.read().splitlines())


def update_processed_files(new_files):
    with open(PROCESSED_FILES_LOG, "a") as f:
        for filename in new_files:
            f.write(f"{filename}\n")


def load_new_pdfs(data_dir, processed_files):
    docs = []
    new_files = []
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".pdf") and filename not in processed_files:
            filepath = os.path.join(data_dir, filename)
            print(f"Loading new file: {filepath}")
            loader = PyPDFLoader(filepath)
            docs.extend(loader.load())
            new_files.append(filename)
    return docs, new_files


def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    if not docs:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def manage_chroma_index(chunks, persist_dir):

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))

    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    if not chunks:
        print("No new documents to add to ChromaDB.")
        return

    if os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
        print("Adding to existing ChromaDB.")
        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        vectordb.add_documents(chunks)
    else:
        print("Creating new ChromaDB.")
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )

    print(f"Saved ChromaDB to: {persist_dir}")


def main():
    load_dotenv()
    processed_files = get_processed_files()
    documents, new_files = load_new_pdfs(DATA_DIR, processed_files)

    if documents:
        chunks = chunk_documents(documents)
        manage_chroma_index(chunks, PERSIST_DIR)
        update_processed_files(new_files)
    else:
        print("No new PDF files to process.")


if __name__ == "__main__":
    main()
