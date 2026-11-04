# Models used in each phase — and why these

## Summary in 5 lines

- **Tokenization** only needs the model's *vocabulary* → that is public, even for closed models like GPT-4o.
- **Logits** need the model's *weights* → those are private for closed models. That is why we only inspect open models.
  - We pick **open** models (full access to the internal computation) and **small** ones (they fit on a laptop and answer fast).
  - We use **two logits models** on purpose: GPT-2 (2019, didactic) and Qwen3-1.7B (2025, realistic).

---

## Vocabulary vs. weights

**The rule in one line:** to inspect logits you need the weights. Weights of closed models do not exist on your machine. Therefore, logits only from open models.

### What "vocabulary" is (the tokenizer)

The tokenizer is **a table** that says how to split text into chunks (tokens). It is not a neural network and does no heavy math. GPT-4o's is **a 1.6 MB file**.

Because it is only a table, OpenAI **publishes** the tokenizer (the `tiktoken` package). It lets anyone count tokens, estimate API cost and build integrations — and it reveals nothing about "how the model thinks".

### What "weights" are (the model itself)

Weights are the **numbers learned during training** — billions of them. They are what turn text into predictions.

GPT-2 has **124 million** weights in a ~525 MB file. Qwen3-1.7B has **1.72 billion** in a **3.2 GB** file. GPT-4 has a number OpenAI does not disclose (estimates point to hundreds of billions), and **its weights were never released**.

---

## Models at a glance

| Model | Phase | Type | Year | Parameters | Vocabulary | Max input | Disk | Access |
|---|---|---|---|---|---|---|---|---|
| `gpt-4o` | Tokenization | Tokenizer only · `o200k_base` (closed model) | 2024 | — | 200,019 | — | 1.6 MB | Public file (no weights) |
| `all-MiniLM-L6-v2` | Tokenization + Embeddings | Encoder | 2021 | 22 M | 30,522 | 512 tokens | ~90 MB | Open · Apache 2.0 |
| `gpt2` | Logits | Decoder | 2019 | 124 M | 50,257 | 1,024 tokens | ~525 MB | Open · MIT |
| `Qwen/Qwen3-1.7B-Base` | Logits | Decoder | 2025 | 1.72 B | 151,936 | 32,768 tokens | 3.2 GB | Open · Apache 2.0 |

Rows with `—` are tokenizers: they have no weights and no context window, because they only split text.

**GPT-4o appears only through its tokenizer.** Its weights are closed, so GPT-4o cannot appear in the Logits phase.

---

## Choosing a model to inspect

Two different questions: **is the model good?** (measured with benchmarks) and **is the model good for me to *see* what is happening?** (access, fit, clarity). The two do not always have the same answer — and the same criteria apply to every phase.

The ✅ / ⚠️ / ❌ marks judge **fit for this tool**, not model quality. A closed model can be excellent and still be ❌ here, because there is nothing to open.

| # | Criterion | ✅ Fits | ⚠️ Fits, with a caveat | ❌ Does not fit |
|---|---|---|---|---|
| 1 | **Access** | Open weights, permissive license | *Gated* — needs Hugging Face login and license acceptance | Closed weights — impossible to inspect (a fine choice for other uses) |
| 2 | **Size** ([how it is computed](#model-size-in-memory)) | Model in memory uses ≤ ~80% of the machine's memory | Model in memory uses 80–100% of it — almost nothing left for the OS | Model in memory needs more than 100% of it — it does not fit |
| 3 | **Tuning** | `Base` when the goal is the raw next-token distribution | `Instruct` when the goal is to mimic real chat usage | `Instruct`, when the goal is raw next-token continuation |
| 4 | **Vocabulary and context** | Context longer than your prompt; vocabulary you can read | 150k+ vocabulary — the top-k table looks more spread out | Context shorter than your prompt |
| 5 | **Quality** | Good *for inspection*: predictable, explainable distribution | Bigger model = better answers, but slower and heavier | Picked by benchmark score alone |

### Model size in memory

**The formula:** `params(B)` is the number in the model name (`Qwen3-1.7B` = 1.7 billion) or on the model card. Memory ≈ `params(B) × 2 GB` (bf16) + ~1 GB of runtime + ~5 GB kept free for the OS and your apps → **total ≈ params(B) × 2 + 6 GB**.

**Worked examples on a 24 GB machine:** `Qwen3-1.7B` → 1.7 × 2 + 6 = **9.4 GB (39%) ✅**; a `7B` → **20 GB (83%) ⚠️**; a `14B` → **34 GB (142%) ❌**.

**Which memory to use:** the total of whatever device runs the model — unified memory on Apple Silicon (*About This Mac* → *Memory*), VRAM on an NVIDIA card (`nvidia-smi`), system RAM on CPU. The `+ 6` is a fixed allowance, not a measurement — lower it at your own risk.

**Solved for the model size:** `max params(B) ≈ (memory − 6) / 2`.

---

## Base vs. Instruct Fine Tuning (and tool calling)

The suffix tells you how much the model was "shaped" after pre-training. The further to the right, the better it follows you — and the less raw its next-token distribution is.

| | Base | Instruct / Chat | Tool-trained (agentic) |
|---|---|---|---|
| Trained for | Predicting the next token on raw text | Following instructions in a conversation | Calling tools (read files, run commands) in a loop |
| Input format | Plain text | System + user + assistant messages (chat template) | Same as Instruct, plus a tool schema |
| Typical output | A continuation of the text | An answer in natural language | A structured call the harness can parse (JSON, XML tags) |
| Follows a system prompt | ❌ | ✅ | ✅ |
| Emits tool calls | ❌ | ⚠️ only if it was also tool-trained | ✅ |
| Good for logits inspection | ✅ the best | ⚠️ shaped by the chat format | ❌ too shaped to read |
| Used in this project | `gpt2`, `Qwen3-1.7B-Base` | not used | what powers opencode, Claude Code, Codex… |

A **harness** (opencode, Claude Code, Codex) is a program that gives the model a list of tools and runs a loop: the model returns a structured call (`read this file`, `edit that line`, `run this command`), the harness executes it, sends the result back, and the model decides the next step. That is what people call an "agent" — and the behavior comes from the **model + harness** pair, not from the model alone.

Driving a harness on real code needs four things, and they come from four different places:

| Requirement | Where it comes from |
|---|---|
| Understands code — syntax, APIs, error messages | **Pre-training** (code in the corpus) |
| Follows the request instead of continuing the text | **Instruct tuning** |
| Emits a parseable tool call, and retries when the call fails | **Tool-use training** |
| Copies existing lines exactly, so the patch applies | **Model fidelity** + a harness that rejects mismatches |

For this project, **Base is the right choice**: we want the raw next-token distribution, before any of that shaping.

---

## Common misconceptions

| Belief | Reality |
|---|---|
| "GPT-2 is the model behind ChatGPT" | No. ChatGPT launched in 2022 on GPT-3.5. GPT-2 is from 2019 and ~1000× smaller. |
| "If I have the tokenizer, I have the model" | No. The tokenizer is the split table; the model is the weights. |
| "Any Instruct model can drive a coding harness" | No. It also needs tool-use training and a harness that parses its calls. |

---

## Quick glossary

| Term | What it is, in one sentence |
|---|---|
| **Token** | A chunk of text (word, part of a word, space, punctuation) that the model sees. |
| **Vocabulary** | The list of all tokens a model knows. |
| **Tokenizer** | The table/function that splits text into tokens and returns numeric IDs. |
| **Weights** | The numbers learned during training; the model's "brain". |
| **Logit** | Raw score from the model's last layer: one number for every token in the vocabulary, for the next position only. Not normalized, not a percentage. |
| **Softmax** | Turns logits into probabilities that sum to 1, using `exp`. It exaggerates the gaps, and the result is the model's *preference*, not the chance of being correct. |
| **Embedding** | A vector of numbers representing the meaning of a text. |
| **Encoder / Decoder** | The encoder reads the whole text and produces a vector (embeddings); the decoder generates text token by token (logits). |
| **Base / Instruct** | Base only continues text; Instruct was tuned to chat. |
| **Tool call** | A structured message in which the model asks the harness to run something. |
| **Harness / agent** | The program that offers tools, executes the calls and loops: opencode, Claude Code, Codex. |
| **Gated** | A model that requires login and license acceptance on Hugging Face to download. |
| **MPS** | The PyTorch mode that uses the GPU of Apple chips (Metal). |
| **Context** | The maximum number of tokens a model can read at once. |
