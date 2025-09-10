import json
import os
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer


class VectorStore:
    def __init__(self, path: str):
        self.path = path
        self.vectors: Dict[str, Dict[str, Any]] = {}
        self.texts: List[str] = []   # raw texts
        self.ids: List[str] = []     # keep IDs aligned with texts
        self.vectorizer = None
        self.matrix = None
        if os.path.exists(path):
            self._load()
            self._rebuild()

    def _load(self):
        """Load vectors (metadata + text) from JSONL."""
        self.vectors = {}
        self.texts = []
        self.ids = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.vectors[obj["id"]] = obj
                self.ids.append(obj["id"])
                self.texts.append(obj["metadata"].get("text", ""))

    def _save(self):
        """Save all vectors to JSONL."""
        with open(self.path, "w", encoding="utf-8") as f:
            for obj in self.vectors.values():
                f.write(json.dumps(obj) + "\n")

    def _rebuild(self):
        """Rebuild TF-IDF matrix after inserts/deletes."""
        if not self.texts:
            self.vectorizer = None
            self.matrix = None
            return
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.texts).toarray()  # store as ndarray

    def reset(self):
        self.vectors = {}
        self.texts = []
        self.ids = []
        self.vectorizer = None
        self.matrix = None
        if os.path.exists(self.path):
            os.remove(self.path)

    def count(self) -> int:
        return len(self.vectors)
    
    def upsert(self, id: str, metadata: Dict[str, Any]):
        """Insert or update a vector with metadata."""
        # Normalize date
        date_val = metadata.get("date")
        if date_val is not None and not isinstance(date_val, str):
            try:
                metadata["date"] = str(date_val)
            except Exception:
                metadata["date"] = None

        self.vectors[id] = {"id": id, "metadata": metadata}
        self.ids = list(self.vectors.keys())
        self.texts = [self.vectors[i]["metadata"].get("text", "") for i in self.ids]

        self._rebuild()
        self._save()

    def query(self, query_text: str, k: int = 5):
        """Return top-k matches by cosine similarity on TF-IDF (manual implementation)."""
        if self.vectorizer is None or self.matrix is None or not self.texts:
            return []

        # Query vector
        q_vec = self.vectorizer.transform([query_text]).toarray()[0]

        # Manual cosine similarity
        norm_q = np.linalg.norm(q_vec)
        sims = []
        for doc_vec in self.matrix:
            norm_d = np.linalg.norm(doc_vec)
            if norm_q == 0 or norm_d == 0:
                sim = 0.0
            else:
                sim = float(np.dot(q_vec, doc_vec) / (norm_q * norm_d))
            sims.append(sim)

        scored = []
        for idx, id in enumerate(self.ids):
            scored.append({
                "id": id,
                "score": sims[idx],
                "metadata": self.vectors[id]["metadata"],
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]
