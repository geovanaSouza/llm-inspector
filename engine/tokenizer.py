from typing import Dict, Any
import tiktoken
from transformers import AutoTokenizer

def analyzer_token(text: str, model_name: str) -> Dict[str, Any]:
    """"
    Analyzes and tokenizes input text using either tiktoken (OpenAI)
    or AutoTokenizer (Hugging Face), depending on the provided model_name.

    Returns a dictionary containing metrics, token IDs, and text chuncks. 
    """
    # Check if model_name in an enconding or model supported by tiktoken
    is_tiktoken = False
    try:
        if model_name in {"cl100k_base", "o200k_base", "p50k_base", "r50k_base"}:
            encoding = tiktoken.get_encoding(model_name)
        else:
            encoding = tiktoken.encoding_for_model(model_name)
        is_tiktoken = True
    except KeyError:
        is_tiktoken = False

    # Execute tokenization based on the detected provider
    if is_tiktoken:
        # Ticktoken execution flow (OpenAI BPE)
        token_ids = encoding.encode(text)
        provider = "tiktoken (OpenAI)"
        token_bytes = [
            {"id": tid, "text": repr(encoding.decode([tid]))}
            for tid in token_ids
        ]
    else:
        # Hugging Face excution flow (WorldPiece / Byte-level BPE for open-source models)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        token_ids = tokenizer.encode(text)
        provider = f"Hugging Face ({tokenizer.__class__.__name__})"
        raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)
        token_bytes = [
            {"id": tid, "text": repr(t_str)}
            for tid, t_str in zip(token_ids, raw_tokens)
        ]

    total_tokens = len(token_ids)
    total_chars = len(text)
    ratio = round(total_chars / total_tokens, 2) if total_tokens > 0 else 0.0

    return {
        "text": text,
        "model_name": model_name,
        "provider": provider,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "ratio": ratio,
        "token_ids": token_ids,
        "token_bytes": token_bytes
    }

