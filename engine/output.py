from typing import Sequence, Any, Optional
from rich.console import Console
from rich.table import Table

console = Console()

def render_table(
        title: str,
        headers: Sequence[str],
        rows: Sequence[Sequence[Any]],
        header_style: str = "bold magenta",
        colum_styles: Optional[Sequence[str]] = None,
) -> None:
    """"
    Renders a generic table in the terminal using Rich.
    """
    table = Table(title=title, show_header=True, header_style=header_style)

    for idx, header in enumerate(headers):
        style = colum_styles[idx] if colum_styles and idx < len(colum_styles) else ""
        table.add_column(header, style=style)

    for row in rows:
        table.add_row(*[str(item) for item in row])

    console.print(table)

def show_token_telemetry(data: dict) -> None:
    """
    Formats tokenization data and uses render_table to display it.
    """
    console.log(f"[bold]Model Name:[/bold] data{'model_name'}\n")

    headers = ["Index", "Token Text Chunck", "Token ID"]
    colum_styles = ["dim", "green", "yellow"]

    rows = [
        [str(idx+1), repr(chunk)[1:-1], str(tid)]
        for idx, (chunk, tid) in enumerate(zip(data["token_bytes"], data["token_ids"]))
    ]

    render_table(
        title=f"[bold cyan]Tokenization Telemetry[/bold cyan]\n[bold]Model Name:[/bold] {data['model_name']}",
        headers=headers,
        rows=rows,
        colum_styles=colum_styles,
    )

    console.print(
        f"[bold]Summary:[/bold] Total Tokens: [cyan]{data['total_tokens']}[/cyan] | "
        f"Characters: [cyan]{data['total_chars']}[/cyan] | "
        f"Char/Token Ratio: [cyan]{data['ratio']}[/cyan]\n"
    )
