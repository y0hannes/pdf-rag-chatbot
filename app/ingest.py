import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from config import DATA_DIR, PERSIST_DIR, PROCESSED_FILES_LOG, EMBEDDING_MODEL

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
            try:
                logging.info(f"Loading new file: {filepath}")
                loader = PyPDFLoader(filepath)
                docs.extend(loader.load())
                new_files.append(filename)
            except Exception as e:
                logging.error(f"Error loading {filepath}: {e}")
    return docs, new_files


def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    if not docs:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    logging.info(f"Split into {len(chunks)} chunks.")
    return chunks


def manage_chroma_index(chunks, persist_dir):

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=os.getenv("GOOGLE_API_KEY"))

    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    if not chunks:
        logging.info("No new documents to add to ChromaDB.")
        return

    if os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
        logging.info("Adding to existing ChromaDB.")
        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        vectordb.add_documents(chunks)
    else:
        logging.info("Creating new ChromaDB.")
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )

    logging.info(f"Saved ChromaDB to: {persist_dir}")


def main():
    load_dotenv()
    processed_files = get_processed_files()
    documents, new_files = load_new_pdfs(DATA_DIR, processed_files)

    if documents:
        chunks = chunk_documents(documents)
        manage_chroma_index(chunks, PERSIST_DIR)
        update_processed_files(new_files)
    else:
        logging.info("No new PDF files to process.")


if __name__ == "__main__":
    main()