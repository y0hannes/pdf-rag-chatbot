from shared import create_qa_chain

if __name__ == "__main__":
    qa = create_qa_chain()

    print("💬 PDF Chatbot (Mistral-Ollama) — type 'exit' to quit")
    while True:
        query = input("\nQuestion: ")
        if query.lower() in ["exit", "quit"]:
            break

        result = qa.invoke(query)
        print("\nAnswer:", result["result"])

        print("\n--- Sources ---")
        for doc in result["source_documents"]:
            print(f"{doc.metadata} -> {doc.page_content[:200]}...")