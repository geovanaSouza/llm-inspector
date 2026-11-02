# Transformer Architecture: Concise Reference Guide

## 1. What is a Transformer?
A **Transformer** is a neural network architecture introduced in 2017 (*"Attention Is All You Need"*) that processes an entire text sequence at once rather than word-by-word. By using **Self-Attention**, it dynamically calculates relationships between all words simultaneously, enabling deep contextual understanding and high-speed GPU parallelization.

---

## 2. The 4-Step Vectorization & Processing Pipeline

1. **Tokenization**  
   Raw text is split into subword units (**tokens**) and assigned numerical Token IDs.

2. **Initial Embeddings + Positional Encoding**  
   Each Token ID is mapped to a static base vector (**embedding**). A **positional encoding** is added so the network understands the exact sequence order.

3. **Transformer Processing (Self-Attention)**  
   Vectors pass through stacked Transformer layers. The **multi-head self-attention mechanism** evaluates context across all tokens (e.g., resolving pronouns like *"it"* or *"he"* based on neighboring words).

4. **Contextual Vectorization & Output**  
   The final vectors reflect the complete meaning of the sentence in context, rather than static dictionary definitions of isolated words.

---

## 3. Encoder vs. Decoder Architectures

| Feature | **Encoder** (e.g., BERT, MiniLM) | **Decoder** (e.g., GPT-4o, LLaMA) |
| :--- | :--- | :--- |
| **Core Goal** | Converts text into dense contextual vectors. | Predicts the next token sequentially to generate text. |
| **Attention** | **Bi-directional**: Reads left and right context simultaneously. | **Causal (Masked)**: Reads only previous tokens. |
| **Primary Use** | Semantic Search, RAG, Text Similarity. | Text Generation, Chatbots, Code Writing. |
