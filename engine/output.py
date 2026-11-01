from rich.console import Console
from rich.table import Table

console = Console()

def show_token_telemetry(data: dict) -> None:
    """
    Renders the tokenization telemetry step visually in the terminal.
    """

    print(data)
