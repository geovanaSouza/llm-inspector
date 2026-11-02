import numpy as np
from sentence_transformers import SentenceTransformer

def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Computes cosine similarity between two 1D vectors: (A . B) / (||A|| * ||B||).

    Returns a score between -1.0 and 1.0 (converted to float).
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a, norm_b = np.linalg.norm(vec_a), np.linalg.norm(vec_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))

def analyze_embeddings(text_a: str, text_b: str, transformer: str = "all-MiniLM-L6-v2") -> dict:
    """
    Generates embeddings for two texts and calculates their semantic similarity.
    """
    # Load a lightweight, state-of-the-art embedding model
    model = SentenceTransformer(transformer)

    # Generate vector representations (embeddings).
    # This instruction executes 3 steps:
    #   - Tokenization: Convert text to Token IDs.
    #   - Contextual Encoding: Passes Token IDs through 6 Transformers layers.
    #   - Pooling: Aggregates token vectors into a single 384-dimensional sentence vector.
    embeddings = model.encode([text_a, text_b])

    vec_a, vec_b = embeddings[0], embeddings[1]

    similarity_score = compute_cosine_similarity(vec_a, vec_b)
    percentage = round(similarity_score * 100, 2)

    return{
        "transformer_model": transformer, 
        "text_a": text_a,
        "text_b": text_b,
        "vector_dim": len(vec_a),
        "similarity_score": round(similarity_score, 4),
        "similarity_percentage": percentage,
        "sample_vector_a": vec_a[:5].round(4).tolist(), # First 5 dimensions as preview
    }