from functools import lru_cache
from typing import Any, Tuple

import torch
import torch.nn.functional as F
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

def analyze_logits(prompt: str, top_k: int = 10, model_name: str = MODEL_NAME) -> dict:
    """
    Extracts raw logits and baseline softmax probabilities from the model's final linear layer for the last token of the input prompt.
    """
    # 1. Text to token IDs
    tokenizer, model = load_model(model_name)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # 2. Forward pass through Transformer layers (inference mode)
    with torch.no_grad():
        outputs = model(**inputs)
        # output.logits shape: (batch_size, sequence_length, vocab_size)
        # We target the LAST token in the sequence to predict the NEXT token candidate
        last_token_logits = outputs.logits[0, -1, :].float()

    # 3. Raw logits to probabilities (standard Softmax, T = 1.0)
    baseline_probs = F.softmax(last_token_logits, dim=-1)

    # 4. Top-k candidates, with their raw scores and probabilities
    top_logits, top_indices = torch.topk(last_token_logits, k=top_k)
    top_probs = baseline_probs[top_indices]

    # 5. Three ways to say how spread out the probabilities are
    top_k_mass = top_k_coverage(baseline_probs, top_k)
    nucleus_90_size = nucleus_size(baseline_probs, threshold=0.90)
    entropy = entropy_bits(baseline_probs)

    # 6. Assemble the result
    last_token_id = inputs["input_ids"][0, -1].item()
    last_token_text = repr(tokenizer.decode([last_token_id]))

    candidates = []
    for rank, (logit, prob, idx) in enumerate(zip(top_logits, top_probs, top_indices), start=1):
        token_str = repr(tokenizer.decode([idx.item()]))
        candidates.append({
            "rank": rank,
            "token_id": idx.item(),
            "token_text": token_str,
            "raw_token": tokenizer.convert_ids_to_tokens(idx.item()),
            "raw_logit": round(logit.item(), 4),
            "baseline_probability": prob.item()
        })

    return {
        "prompt": prompt,
        "model_name": model_name,
        "vocab_size": model.config.vocab_size,
        "last_token_id": last_token_id,
        "last_token_text": last_token_text,
        "top_k_mass": top_k_mass,
        "nucleus_90_size": nucleus_90_size,
        "entropy_bits": entropy,
        "candidates": candidates
    }

# top_k_coverage(p, k)   -> "for this k, how much is covered?"
def top_k_coverage(probabilities: torch.Tensor, k: int) -> float:
    """How much of the model's total probability sits in the top k tokens."""
    return torch.topk(probabilities, k=k).values.sum().item()

# nucleus_size(p, 0.90)  -> "to cover 90%, which k do I need?"
def nucleus_size(probabilities: torch.Tensor, threshold: float = 0.90) -> int:
    """How many tokens you need to reach `threshold` of the model's total probability."""
    sorted_probs = torch.sort(probabilities, descending=True).values
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    return int((cumulative_probs < threshold).sum().item()) + 1

# entropy_bits(p)      -> "how many tokens share it, without picking a cutoff?"
def entropy_bits(probabilities: torch.Tensor) -> float:
    """
    How many tokens have a similar probability, expressed in bits.
    0.0 means one token holds everything.
    """
    probs_safe = torch.clamp(probabilities, min=1e-12)
    return -(probabilities * torch.log2(probs_safe)).sum().item()