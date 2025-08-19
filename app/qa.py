from shared import create_qa_chain
from rich.console import Console
from rich.panel import Panel

if __name__ == "__main__":
    qa = create_qa_chain()
    console = Console()

    console.print("💬 [bold green]PDF Chatbot (with memory)[/bold green] — type 'exit' to quit")
    try:
        while True:
            query = console.input("\n[bold blue]Question:[/bold blue] ")
            if query.lower() in ["exit", "quit"]:
                break

            with console.status("[bold yellow]Thinking...[/bold yellow]"):
                result = qa.invoke({"question": query})
            
            console.print("\n[bold green]Answer:[/bold green]", result["answer"])

            if result.get("source_documents"):                
                console.print("\n--- [bold]Sources[/bold] ---")                
                for doc in result["source_documents"]:
                    metadata_str = ", ".join(f"[bold]{k}[/bold]: {v}" for k, v in doc.metadata.items())
                    panel_content = f"{metadata_str}\n\n{doc.page_content[:500]}..."
                    console.print(Panel(panel_content, border_style="blue"))

    except KeyboardInterrupt:
        console.print("\n[bold red]Exiting...[/bold red]")

