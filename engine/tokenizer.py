import tiktoken

def analyzer_token(text: str, model_name: str = "gpt-4o") -> dict:
    """"
    Analyzes input text by converting it into subwords tokens using Byte-Pair Encoding (BPE).

    Returns a dictionary containing metrics, token IDs, and text chuncks. 
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    token_ids = encoding.encode(text)
    token_bytes = [encoding.decode([t]) for t in token_ids]

    total_tokens = len(token_ids)
    total_chars = len(text)
    ratio = round(total_chars / total_tokens, 2) if total_tokens > 0 else 0.0

    return {
        "model_name": model_name,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "ratio": ratio,
        "token_ids": token_ids,
        "token_bytes": token_bytes
    }

