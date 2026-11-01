from rich.console import Console
from rich.panel import Panel

console = Console()

def main():
    welcome()

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