# Phases of the pipeline

The tool runs three phases, in this order. Each one asks a different question and needs a different level of access to the model — see [Models used in each phase](./models.md) for the model choices.

| Phase | Question it answers | Needs from the model | Code |
|---|---|---|---|
| **Tokenization** | "How is this text split?" | Only the vocabulary table | `engine/tokenizer.py` |
| **Embeddings** | "Are these two texts similar?" | A complete *encoder*, but small | `engine/embeddings.py` |
| **Logits** | "What is the next token, and how likely?" | The full weights of a generative model | `engine/logits.py` |

---

## Phase 1 — Tokenization

**Goal:** show that the same text becomes a different number of tokens depending on the model. This affects cost, context limits and performance.

Two tokenizers are used on purpose, for contrast: `o200k_base` (GPT-4o, 200,019 tokens) and the `BertTokenizer` vocab (all-MiniLM-L6-v2, 30,522 tokens). A closed model and an open one, side by side.

---

## Phase 2 — Embeddings

**Goal:** turn text into a vector of numbers and measure how similar two texts are (cosine similarity).

This is the only phase that compares the two prompts. A small encoder is enough — the output is a 384-dimension vector, not text.

---

## Phase 3 — Logits

**Goal:** given a text, list the most likely next tokens, with the raw score (logit) and the probability.

Only the last token of the prompt is read, because that is the position that predicts what comes next. Two models run on the same prompt (GPT-2 and Qwen3-1.7B-Base) so the difference in quality is visible on the same screen.
