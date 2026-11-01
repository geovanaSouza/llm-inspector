from engine.output import show_token_telemetry
from engine.tokenizer import analyzer_token
from rich.console import Console
from rich.panel import Panel
import sys

console = Console()

def main():
    if len(sys.argv) < 2:
        print("[ERROR] No prompt provided!")
        print("Usage: make run PROMPT=\"Yout text here\"")
        sys.exit(1)

    # Capture prompt from CLI arguments or use default
    prompt = " ".join(sys.argv[1:])

    # Welcome and Initial Telemetry Output
    welcome()

    # Tokenization Execution
    token_data = analyzer_token(prompt)
    show_token_telemetry(token_data)


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