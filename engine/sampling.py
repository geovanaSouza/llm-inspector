import torch
import torch.nn.functional as F
from typing import Dict, Any

def analyze_sampling(
        raw_logits: torch.Tensor,
        tokenizer: Any,
        temperature: float = 1.0,
        num_candidates: int = 10
) -> Dict[str, Any]:
    """
    Computes the baseline (T=1.0) and temperature-scaled probability distribution and spread metrics from raw logits.
    num_candidates also defines the k of top_k_mass — the metric covers exactly the candidates shown.
    """

    # 1. Raw logits to calculated baseline probabilities (standard Softmax, T = 1.0, unfiltered)
    baseline_probs = F.softmax(raw_logits, dim=-1)

    # 2. Temperature scaling transformation
    safe_temp = max(temperature, 1e-5)
    scaled_logits = raw_logits / safe_temp

    # 3. Final probabilities post-temperature scaling
    final_probs = F.softmax(scaled_logits, dim=-1)

    # 4. Candidates, with their raw scores, baseline probabilities, sampled probabilities and ids
    top_logits, top_indices = torch.topk(final_probs, k=num_candidates)

    # 5. Three ways to say how spread out the probabilities are (final probabilities after apply temperature)
    top_k_mass = top_k_coverage(final_probs, num_candidates)
    nucleus_90_size = nucleus_size(final_probs, threshold=0.90)
    entropy = entropy_bits(final_probs)

    candidates = []
    for rank, idx in enumerate(top_indices, start=1):
        token_id = idx.item()
        candidates.append({
            "rank": rank,
            "token_id": token_id,
            "token_text": repr(tokenizer.decode([token_id])),
            "raw_token": tokenizer.convert_ids_to_tokens(token_id),
            "raw_logit": round(raw_logits[token_id].item(), 4),
            "baseline_probability": baseline_probs[token_id].item(),
            "sampled_probability": final_probs[token_id].item()
        })

    return {
        "temperature": temperature,
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