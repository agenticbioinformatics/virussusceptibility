import pickle
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ── cosine similarity ─────────────────────────────────────────────────────────

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


# ── in-memory backend (default) ───────────────────────────────────────────────

class VectorDatabase:
    """In-memory vector store with cosine-similarity search.

    Backed by a plain dict; no external dependencies required.
    Serialised as a single pickle file.
    """

    def __init__(self):
        self.vectors: dict = defaultdict(np.ndarray)

    def __len__(self) -> int:
        return len(self.vectors)

    def insert(self, key: str, vector: np.ndarray) -> None:
        self.vectors[key] = vector

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        similarities = [
            (key, cosine_similarity(query_vector, vec))
            for key, vec in self.vectors.items()
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def retrieve(self, key: str) -> np.ndarray:
        return self.vectors.get(key)


# ── FAISS backend (optional) ──────────────────────────────────────────────────

class FaissVectorDatabase:
    """FAISS-backed vector store using an inner-product index (cosine after L2 normalisation).

    Requires:  pip install faiss-cpu   (or faiss-gpu)

    Serialised as two files:
      <path>.faiss  — the FAISS index
      <path>.keys.pkl — ordered list of string keys (index position → key)
    """

    def __init__(self):
        self._index = None   # lazily initialised on first insert
        self.keys: List[str] = []

    def __len__(self) -> int:
        return len(self.keys)

    def _require_faiss(self):
        try:
            import faiss
            return faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for the FAISS backend.\n"
                "Install it with:  pip install faiss-cpu"
            )

    def insert(self, key: str, vector: np.ndarray) -> None:
        faiss = self._require_faiss()
        v = vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(v)
        if self._index is None:
            self._index = faiss.IndexFlatIP(v.shape[1])
        self._index.add(v)
        self.keys.append(key)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        faiss = self._require_faiss()
        if self._index is None or len(self.keys) == 0:
            return []
        q = query_vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = self._index.search(q, min(top_k, len(self.keys)))
        return [
            (self.keys[i], float(s))
            for i, s in zip(indices[0], scores[0])
            if i >= 0
        ]

    def retrieve(self, key: str) -> None:
        # FlatIP does not support retrieval by key; use VectorDatabase if needed.
        return None


# ── persistence helpers ───────────────────────────────────────────────────────

def save_database(db, path: Path) -> None:
    """Persist a VectorDatabase or FaissVectorDatabase to disk.

    VectorDatabase  → single file at <path>
    FaissVectorDatabase → <path> (FAISS index) + <path>.keys.pkl
    """
    path = Path(path)
    if isinstance(db, FaissVectorDatabase):
        import faiss
        faiss.write_index(db._index, str(path))
        with open(_keys_path(path), "wb") as f:
            pickle.dump(db.keys, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(path, "wb") as f:
            pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_database(path: Path):
    """Load a database previously saved with save_database().

    Auto-detects the backend from the file extension:
      .faiss  → FaissVectorDatabase
      .pkl    → VectorDatabase
    """
    path = Path(path)
    if path.suffix == ".faiss":
        import faiss
        db = FaissVectorDatabase()
        db._index = faiss.read_index(str(path))
        with open(_keys_path(path), "rb") as f:
            db.keys = pickle.load(f)
        return db
    with open(path, "rb") as f:
        return pickle.load(f)


def _keys_path(faiss_path: Path) -> Path:
    return faiss_path.with_suffix(".faiss.keys.pkl")
