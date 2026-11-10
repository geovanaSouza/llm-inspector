# Logits: what the tool measures

## In one line

A **logit** is the raw score the model gives to **every token in its vocabulary** for the **next position**. It is not a probability, and it is not a list of "candidate answers".

---

## The pipeline in five steps

| Step | What happens | Shape |
|---|---|---|
| 1. **Tokenize** | text becomes token IDs | `(N,)` |
| 2. **Forward pass** | the model reads the IDs and scores every token | `(1, N, V)` |
| 3. **Cut the last position** | only the final token predicts what comes next | `(V,)` |
| 4. **Softmax** (`T = 1`) | scores become probabilities that add up to 1.0 | `(V,)` |
| 5. **Top-k** | the 10 highest are shown; everything else is hidden but still counted | `(10,)` |

The model produces a full vector of `V` scores at **every** position — for a 7-token prompt that is 7 × 151,936 numbers. All are discarded except the last, because only the last one predicts the next token.

The five steps run in two functions:

```
engine/logits.py    get_raw_logits           -> steps 1-3: model forward pass, raw scores
engine/sampling.py  build_sampling_baseline  -> steps 4-5: softmax (T=1), top-k and the metrics
```

---

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

---

## What these numbers do **not** say

- **Probability is not correctness.** The softmax gives the probability the model assigns, not the chance of being right. A flat distribution can still have the right answer somewhere in it.
