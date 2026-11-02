from engine.output import show_token_telemetry, show_embedding_telemetry
from engine.tokenizer import analyzer_token
from engine.embeddings import analyze_embeddings
from rich.console import Console
from rich.panel import Panel
import sys

console = Console()

def main() -> None:
    # Ensure at least two prompts are provided via CLI
    if len(sys.argv) < 3:
        print("[ERROR] Two prompts are required for comparison!")
        print('Usage: make run PROMPT1="First text" PROMPT2="Second text"')
        sys.exit(1)

    # Capture prompt from CLI arguments or use default
    prompt_a = sys.argv[1]
    prompt_b = sys.argv[2]

    # Welcome Header
    welcome()

    # Tokenization Execution (analyzing Prompt A using GPT-4)
    token_data = analyzer_token(prompt_a, model_name="gpt-4o")
    show_token_telemetry(token_data)

    # Tokenization Execution (analyzing Prompt A using GPT-4 tokenizer)
    token_data = analyzer_token(prompt_a, model_name="sentence-transformers/all-MiniLM-L6-v2")
    show_token_telemetry(token_data)

    # Embedding & Similarity Comparison
    embedding_data = analyze_embeddings(prompt_a,prompt_b)
    show_embedding_telemetry(embedding_data)

def welcome():
    '''Displays the initial header for the Prompt Telemetry Engine.'''
    console.print(
        Panel(
            f"[bold green]Prompt Telemetry Engine Initialized![/bold green]\n"
            f"",
            title="[bold blue]LLM Inspector[/bold blue]"
        )
    )

if __name__ == "__main__":
    main()