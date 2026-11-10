from typing import Sequence, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

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
        [str(idx+1), chunk["text"], str(tid)]
        for idx, (chunk, tid) in enumerate(zip(data["token_bytes"], data["token_ids"]))
    ]

    render_table(
        title=f"[bold cyan]Tokenization Telemetry[/bold cyan]\n[bold]Model Name:[/bold] {data['model_name']}\n[bold]Provider:[/bold] {data['provider']}",
        headers=headers,
        rows=rows,
        colum_styles=colum_styles,
    )

    console.print(
        f"[bold]Summary:[/bold] Total Tokens: [cyan]{data['total_tokens']}[/cyan] | "
        f"Characters: [cyan]{data['total_chars']}[/cyan] | "
        f"Char/Token Ratio: [cyan]{data['ratio']}[/cyan]\n"
    )

def show_embedding_telemetry(data: dict) -> None:
    """
    Formats embedding similarity data and display it using render_table.
    """
    headers = ["Property", "Value"]
    colum_styles = ["bold yellow", "cyan"]

    rows = [
        ["Text A (Prompt)", data["text_a"]],
        ["Text B (Reference)", data["text_b"]],
        ["Vector Dimensions", str(data["vector_dim"])],
        ["Cosine Similarity", f"{data['similarity_score']} ({data['similarity_percentage']}%)"],
        ["Vector Preview (First 5 dims)", str(data["sample_vector_a"])],
    ]

    render_table(
        title=f"[bold cyan]Embedding & Vector Similarity Telemetry[/bold cyan]\n[bold]Transformer Model:[/bold] {data['transformer_model']}",
        headers=headers,
        rows=rows,
        colum_styles=colum_styles,
    )
    console.print()

def show_logit_panel(prompt: str) -> None:
    """
    Formats logits data and display it using render_table.
    """
    console.print(
        Panel(
            f"[bold]Prompt:[/bold] {prompt}",
            title="[bold blue]Logits Telemetry[/bold blue]"
        )
    )

def show_baseline_table(data: dict, model: str) -> None:
    """
    Displays the baseline candidate table for one model's raw logits.
    """
    title = (
        f"[bold cyan]Raw Logits Table[/bold cyan]\n"
        f"[bold]Model:[/bold] {model}"
    )
    headers = ["Rank", "Token Text", "Token ID", "Raw Logit", "Baseline Prob(%)"]
    rows = [
        [
            str(c["rank"]),
            c["token_text"],
            str(c["token_id"]),
            f"{c['raw_logit']:.4f}",
            f"{c['baseline_probability'] * 100:.2f}%"
        ]
        for c in data["candidates"]
    ]

    render_table(headers=headers, rows=rows, title=title)

def show_logits_comparison(results: Sequence[dict]) -> None:
    """
    Side-by-side summary of the same prompt across models.
    """
    headers = ["Metric"] + [r["model_name"] for r in results]

    metrics = [
        ("Top-1 token",                  lambda r: r["candidates"][0]["token_text"]),
        ("Top-1 probability",            lambda r: f"{r['candidates'][0]['baseline_probability'] * 100:.2f}%"),
        ("Top-10 coverage",              lambda r: f"{r['top_k_mass'] * 100:.2f}%"),
        ("Nucleus (90%)",                lambda r: f"{r['nucleus_90_size']:,} tokens"),
        ("Equivalent number of choices", lambda r: f"{2 ** r['entropy_bits']:.0f}"),
        ("Vocabulary",                   lambda r: f"{r['vocab_size']:,}"),
    ]

    rows = [[name] + [fmt(r) for r in results] for name, fmt in metrics]

    render_table(
        title="[bold cyan]Logits Comparison[/bold cyan]",
        headers=headers,
        rows=rows,
        colum_styles=["bold yellow"] + ["cyan"] * len(results),
    )