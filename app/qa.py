from shared import create_qa_chain

if __name__ == "__main__":
    qa = create_qa_chain()

    print("💬 PDF Chatbot (with memory) — type 'exit' to quit")
    while True:
        query = input("\nQuestion: ")
        if query.lower() in ["exit", "quit"]:
            break

        result = qa.invoke({"question": query})
        print("\nAnswer:", result["answer"])

        if "source_documents" in result:
            print("\n--- Sources ---")
            for doc in result["source_documents"]:
                metadata_str = ", ".join(f"{k}: {v}" for k, v in doc.metadata.items())
                print(f"{metadata_str} -> {doc.page_content[:200]}...")
