from engine.embeddings import analyze_embeddings
from engine.logits import get_raw_logits
from engine.output import show_token_telemetry, show_embedding_telemetry, show_logit_panel, show_baseline_table, show_logits_comparison
from engine.sampling import build_sampling_baseline
from engine.tokenizer import analyzer_token
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

    # Logits + Sampling baseline (Prompt A, two models, then a side-by-side comparison)
    show_logit_panel(prompt_a)
    logits_results = []
    for model in ["gpt2", "Qwen/Qwen3-1.7B-Base"]:
        logits = get_raw_logits(prompt_a, model_name=model)
        baseline = build_sampling_baseline(logits['raw_logits'], logits['tokenizer'])
        show_baseline_table(baseline, model=model)

        logits_results.append({
            "model_name": logits["model_name"],
            "vocab_size": logits['vocab_size'],
            **baseline,
        })
    show_logits_comparison(logits_results)

def welcome():
    '''Displays the initial header for the Prompt Telemetry Engine.'''
    console.print(
        Panel(
            "[bold green]Prompt Telemetry Engine Initialized![/bold green]\n",
            title="[bold blue]LLM Inspector[/bold blue]"
        )
    )

if __name__ == "__main__":
    main()