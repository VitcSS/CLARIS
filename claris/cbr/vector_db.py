import faiss
import numpy as np
from typing import Tuple, List, Dict, Any

class VectorDatabase:
    def __init__(self, dimension: int):
        """
        Inicializa o banco vetorial utilizando indexação HNSW para o espaço SHAP.
        """
        self.dimension = dimension
        self.index = faiss.IndexHNSWFlat(dimension, 32)
        self.metadata: List[Dict[str, Any]] = [] 

    def add_instances(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        if len(vectors) != len(metadata):
            raise ValueError("O número de vetores e metadados deve ser idêntico.")
        
        vecs_np = np.array(vectors, dtype=np.float32)
        
        # Garante estritamente formato 2D (n, self.dimension)
        if vecs_np.ndim == 1:
            vecs_np = vecs_np.reshape(1, -1)
        elif vecs_np.ndim > 2:
            vecs_np = vecs_np.reshape(vecs_np.shape[0], -1)

        self.index.add(vecs_np)
        self.metadata.extend(metadata)

    def search_neighbors(self, query_vector: np.ndarray, k: int = 10) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Realiza a busca KNN para recuperar vizinhos (âncoras e contrafactuais).
        """
        distances, indices = self.index.search(np.array([query_vector], dtype=np.float32), k)
        
        neighbors_metadata = [self.metadata[idx] for idx in indices[0] if idx != -1]
        return neighbors_metadata, distances[0].tolist()