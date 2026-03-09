from sentence_transformers import SentenceTransformer
import numpy as np
import time


class Embedder:
    def __init__(self, model_name: str):
        print(f"Loading embedding model '{model_name}'...")
        t = time.time()
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded in {time.time() - t:.1f}s")

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Returns 2D array, shape (len(texts), embedding_dim)"""
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
