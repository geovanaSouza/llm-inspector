import torch
import torch.nn.functional as F
from typing import Dict, Any

def build_sampling_baseline(
        raw_logits: torch.Tensor,
        tokenizer: Any,
        num_candidates: int = 10
) -> Dict[str, Any]:
    """
    Computes the baseline (T=1.0) probability distribution and spread metrics from raw logits.
    num_candidates also defines the k of top_k_mass — the metric covers exactly the candidates shown.
    """

    # 1. Raw logits to calculated baseline probabilities (standard Softmax, T = 1.0, unfiltered)
    baseline_probs = F.softmax(raw_logits, dim=-1)

    # 2. Candidates, with their raw scores, probabilities and ids
    top_logits, top_indices = torch.topk(raw_logits, k=num_candidates)
    top_probs = baseline_probs[top_indices]

    # 3. Three ways to say how spread out the probabilities are
    top_k_mass = top_k_coverage(baseline_probs, num_candidates)
    nucleus_90_size = nucleus_size(baseline_probs, threshold=0.90)
    entropy = entropy_bits(baseline_probs)

    candidates = []
    for rank, (logit, prob, idx) in enumerate(zip(top_logits, top_probs, top_indices), start=1):
        token_id = idx.item()
        candidates.append({
            "rank": rank,
            "token_id": token_id,
            "token_text": repr(tokenizer.decode([token_id])),
            "raw_token": tokenizer.convert_ids_to_tokens(token_id),
            "raw_logit": round(logit.item(), 4),
            "baseline_probability": prob.item()
        })

    return {
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