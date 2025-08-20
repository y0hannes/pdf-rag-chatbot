# 📄 PDF RAG Chatbot

A conversational AI chatbot that allows users to upload PDFs and query their contents. The app uses **RAG (Retrieval-Augmented Generation)** with:

- 📚 **ChromaDB** for vector storage
- 🔎 **Google Generative AI Embeddings** for text embeddings
- 🧠 **Groq LLaMA3 (via LangChain)**
- 🌐 **Streamlit** frontend for an interactive chat UI

Each conversation is **isolated**, with its own vector DB and file uploads (like ChatGPT).

---

## 🚀 Features

- Upload one or more PDFs per conversation
- Ask natural language questions about the uploaded documents
- Conversational memory (chat history)
- Show document sources for each answer
- Answer 'I don't know' when the PDF doesn't have any information' 
- Multiple conversation threads (each with its own PDF set & DB)
- Reset, clear, or create new conversations at any time

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

### 2. Create virtual environment & install dependencies

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Setup environment variables

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

---

## ▶️ Usage

### Run the Streamlit web app:

```bash
streamlit run app/webapp.py
```

Then open: [**http://localhost:8501**](http://localhost:8501) in your browser.

### Use CLI:
```bash
python3 app/qa.py
```

---

## 📖 Workflow

1. **Upload a PDF** → The file is processed, split into chunks, and embedded into ChromaDB.
2. **Ask a Question** → The retriever fetches relevant chunks, and the LLM answers using context.
3. **View Sources** → Expand to see which PDF pages were used for the answer.
4. **Conversations** → Start new chats with different PDFs (each isolated).
