from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

PERSIST_DIR = "chroma_db"
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
ollama_model = "mistral"


def load_chroma():
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    return vectordb


def create_qa_chain():
    db = load_chroma()
    retriever = db.as_retriever(
        search_type="similarity", search_kwargs={"k": 3})

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

    llm = Ollama(model=ollama_model, temperature=0)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa
