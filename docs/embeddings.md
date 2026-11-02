# Embeddings & Vector Similarity Cheatsheet

## 1. Core Concepts
* **Embedding**: A dense numerical vector (e.g., 384 floats) representing the semantic meaning of text in a high-dimensional space.
* **Semantic Proximity**: Texts with similar meanings or contexts map to nearby coordinates. Unrelated texts map to orthogonal vectors.

---

## 2. BERT vs. `all-MiniLM-L6-v2`

* **`bert-base-uncased` (Raw Encoder)**:
  * Suffers from **anisotropy** (vectors concentrate in a narrow directional cone).
  * Produces artificially high similarity scores ($\approx 0.8+$) even for unrelated sentences.
* **`all-MiniLM-L6-v2` (Sentence-Transformer)**:
  * Fine-tuned with **contrastive learning (SBERT)** for isotropic space distribution.
  * Calibrated cosine similarity: realistic scores for semantic retrieval (~22 MB, 384 dimensions).
