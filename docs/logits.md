# Logits & Sampling: what the tool measures

## In one line

A **logit** is the raw score the model gives to **every token in its vocabulary** for the **next position**. Sampling transforms these raw scores into a probability distribution using **Temperature**, **Top-K**, and **Top-P**.

## The pipeline in six steps

| Step | What happens | Shape |
|---|---|---|
| 1. **Tokenize** | text becomes token IDs | `(N,)` |
| 2. **Forward pass** | the model reads the IDs and scores every token | `(1, N, V)` |
| 3. **Cut the last position** | only the final token predicts what comes next | `(V,)` |
| 4. **Temperature scaling** | divide raw logits by `T` (`logits / T`) | `(V,)` |
| 5. **Softmax** | scores become probabilities that add up to 1.0 | `(V,)` |
| 6. **Top-k / Top-p** | top candidates are retained; metrics and probabilities recalculated | `(10,)` |

The model produces a full vector of `V` scores at **every** position — for a 7-token prompt that is 7 × 151,936 numbers. All are discarded except the last, because only the last one predicts the next token.

The pipeline runs across two functions:

```
engine/logits.py    get_raw_logits           -> steps 1-3: model forward pass, raw scores
engine/sampling.py  analyze_sampling  -> steps 4-6: temperature scaling, softmax, top-k/p and metrics
```

## Temperature: calibrating randomness vs determinism

| Temperature | Behavior | What happens mathematically | Use case | 
|---|---|---|---| 
| **T = 1.0 (Baseline)** | **Raw / Uncalibrated** | No scaling (`logits / 1.0`). Shows the original probabilities learned during training. | Baseline analysis | 
| **T < 1.0 (e.g. `0.2`)** | **Deterministic / Focused** | Logits are amplified (`logits / 0.2`). Top choices dominate, approaching 100%. | Code generation, math, factual answers | 
| **T > 1.0 (e.g. `1.5`)** | **Creative / Diverse** | Logits are compressed (`logits / 1.5`). The distribution flattens, boosting long-tail tokens. | Brainstorming, story generation variety | 

💡 **Note:** Lowering temperature sharpens confidence into determinism; raising temperature flattens confidence into variety.

## The three summary numbers

All three come from the **same** probability vector, and each reads it from a different angle:

| Function | Question it answers | Displayed as |
|---|---|---|
| `top_k_coverage` | for this `k`, how much of the 100% is covered? | `Top-10 coverage` |
| `nucleus_size` | to cover 90%, which `k` do I need? | `Nucleus (90%)` |
| `entropy_bits` → `2 ** entropy` | how spread out is it, as a count of equally likely options? | `Equivalent number of choices` |

`top_k_coverage` and `nucleus_size` are the same curve read on opposite axes:

```
coverage(10)       -> "at rank 10, how much has accumulated?"
nucleus(0.90)      -> "at what rank does accumulation reach 90%?"
```

`Equivalent number of choices` is `2 ** entropy`, so `X` reads as "as if there were 458 tokens with the same probability".

## What these numbers do **not** say

- **Probability is not correctness.** The softmax gives the probability the model assigns, not the chance of being right. A flat distribution can still have the right answer somewhere in it.
- **High temperature is not intelligence.** Higher temperature increases randomness, which boosts variety but can lead to hallucinations or incoherent tokens if set too high (T > 2.0).