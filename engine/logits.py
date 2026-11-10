from functools import lru_cache
from typing import Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-1.7B-Base"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

@lru_cache(maxsize=2)
def load_model(model_name: str = MODEL_NAME) -> Tuple[Any, Any]:
    """
    Lazily loads the tokenizer and casual LM, caching then after first call.
    It uses the best available device in local machine.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # bfloat16 halves memory usage; MPS os Apple Silicon supports it natively
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(DEVICE)
    model.eval() # inference mode: disables dropout
    return tokenizer, model

def get_raw_logits(prompt: str, model_name: str = MODEL_NAME) -> dict:
    """
    Extracts raw logits from the model's final linear layer for the last token of the input prompt.
    """
    # 1. Text to token IDs
    tokenizer, model = load_model(model_name)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # 2. Forward pass through Transformer layers (inference mode)
    with torch.no_grad():
        outputs = model(**inputs)
        # output.logits shape: (batch_size, sequence_length, vocab_size)
        # We target the LAST token in the sequence to predict the NEXT token candidate
        last_token_logits = outputs.logits[0, -1, :].float().cpu()

    return {
        "model_name": model_name,
        "vocab_size": model.config.vocab_size,
        "raw_logits": last_token_logits,
        "tokenizer": tokenizer
    }
