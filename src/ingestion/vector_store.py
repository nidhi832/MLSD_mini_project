"""
Vector Store and Retrieval Engine (L1, D2).

- Index: FAISS (inner-product over L2-normalised vectors = cosine similarity),
  driven through LangChain's FAISS vector store.
- Embeddings: sentence-transformers all-MiniLM-L6-v2, run locally. If the model
  cannot be loaded (no network on first start), a deterministic 384-dim hashing
  embedder is used instead. The backend actually used is recorded in the manifest,
  and stored vectors are re-embedded automatically if the backend changes, so a
  query is never compared against vectors from a different model.
- Idempotent, incremental indexing: chunk ids are content hashes, so on re-run only
  new or changed chunks are embedded; removed chunks are dropped; unchanged
  vectors are reused (re-embedding without rebuilding the whole index).
- Exact clause lookup by identifier, returning the clause together with its
  sub-clauses (looking up CAR-3.7 returns 3.7, 3.7.1 and 3.7.2).
"""

import os
import json
import time
import hashlib
from typing import List, Dict, Any, Optional

import numpy as np

from src.ingestion.chunker import RegulatoryChunk

BUILTIN_BACKEND = "builtin-hashing-384"


try:
    from langchain_core.embeddings import Embeddings as _LCEmbeddings
except Exception:  # pragma: no cover
    _LCEmbeddings = object


class _PrecomputedEmbeddings(_LCEmbeddings):
    """LangChain Embeddings adapter around this store's embedding function."""

    def __init__(self, store: "RegulatoryVectorStore"):
        self._store = store

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._store._compute_embeddings(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._store._compute_embeddings([text])[0].tolist()


class RegulatoryVectorStore:
    def __init__(
        self,
        storage_dir: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
    ):
        self.storage_dir = storage_dir or os.getenv("VECTOR_INDEX_DIR", "data/vector_store")
        self.embedding_model_name = embedding_model_name or os.getenv(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
        os.makedirs(self.storage_dir, exist_ok=True)

        self.chunks_file = os.path.join(self.storage_dir, "chunks.json")
        self.embeddings_file = os.path.join(self.storage_dir, "embeddings.npy")
        self.manifest_file = os.path.join(self.storage_dir, "manifest.json")
        self.faiss_dir = os.path.join(self.storage_dir, "faiss")

        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.manifest: Dict[str, Any] = {}

        self._encoder = None
        self._faiss = None
        self._load_existing_store()

    # ------------------------------------------------------------------ embeddings
    def _get_encoder(self):
        if self._encoder is None:
            if os.getenv("EMBEDDING_BACKEND", "").lower() == "builtin":
                self._encoder = "builtin_deterministic"
                return self._encoder
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.embedding_model_name)
            except Exception:
                self._encoder = "builtin_deterministic"
        return self._encoder

    @property
    def embedding_backend(self) -> str:
        enc = self._get_encoder()
        return BUILTIN_BACKEND if enc == "builtin_deterministic" else self.embedding_model_name

    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        encoder = self._get_encoder()
        if encoder != "builtin_deterministic":
            return np.asarray(
                encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                dtype=np.float32,
            )

        dim = 384
        vectors = []
        for text in texts:
            vec = np.zeros(dim, dtype=np.float32)
            words = text.lower().replace("\n", " ").split()
            for i, w in enumerate(words):
                w = w.strip(".,:;()'\"")
                if not w:
                    continue
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
                weight = 1.0 + (1.0 / (1.0 + i * 0.05))
                vec[h % dim] += weight * (1.0 if (h & 1) else -1.0)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

    @staticmethod
    def _embed_text(c: Dict[str, Any]) -> str:
        return f"{c['section_title']}\n{c['text']}"

    # ------------------------------------------------------------------ persistence
    def _load_existing_store(self):
        if os.path.exists(self.chunks_file) and os.path.exists(self.embeddings_file):
            try:
                with open(self.chunks_file, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                self.embeddings = np.load(self.embeddings_file)
                if os.path.exists(self.manifest_file):
                    with open(self.manifest_file, "r", encoding="utf-8") as f:
                        self.manifest = json.load(f)
            except Exception:
                self.chunks = []
                self.embeddings = None

    def _ensure_backend_consistency(self):
        """Re-embed stored chunks if they were built with a different embedding backend."""
        if not self.chunks:
            return
        stored = self.manifest.get("embedding_backend")
        if stored == self.embedding_backend and self.embeddings is not None \
                and len(self.embeddings) == len(self.chunks):
            return
        self.embeddings = self._compute_embeddings([self._embed_text(c) for c in self.chunks])
        self.manifest["embedding_backend"] = self.embedding_backend
        self.manifest["reembedded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._faiss = None
        # Kept in memory only; `python -m src.ingestion.pipeline` persists a rebuilt store.

    def _get_faiss(self):
        """Build (or rebuild) the LangChain FAISS index from the stored vectors."""
        if self._faiss is not None:
            return self._faiss
        self._ensure_backend_consistency()
        if not self.chunks:
            return None
        from langchain_community.vectorstores import FAISS
        from langchain_community.vectorstores.utils import DistanceStrategy

        pairs = [(self._embed_text(c), vec.tolist()) for c, vec in zip(self.chunks, self.embeddings)]
        metadatas = [{"idx": i} for i in range(len(self.chunks))]
        self._faiss = FAISS.from_embeddings(
            text_embeddings=pairs,
            embedding=_PrecomputedEmbeddings(self),
            metadatas=metadatas,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )
        return self._faiss

    # ------------------------------------------------------------------ indexing
    def index_chunks(self, new_chunks: List[RegulatoryChunk], source_versions: Dict[str, str]) -> Dict[str, Any]:
        """
        Idempotent, incremental indexing (D2, L1): embeds only new or modified chunks,
        reuses stored vectors for unchanged ones, and drops chunks no longer in the corpus.
        """
        start_time = time.time()
        backend = self.embedding_backend
        same_backend = self.manifest.get("embedding_backend") == backend
        existing = {}
        if same_backend and self.embeddings is not None and len(self.embeddings) == len(self.chunks):
            existing = {c["chunk_id"]: self.embeddings[i] for i, c in enumerate(self.chunks)}

        ordered, vectors, to_embed_idx, to_embed_txt = [], [], [], []
        for chunk in new_chunks:
            c = chunk.to_dict()
            ordered.append(c)
            if c["chunk_id"] in existing:
                vectors.append(existing[c["chunk_id"]])
            else:
                vectors.append(None)
                to_embed_idx.append(len(ordered) - 1)
                to_embed_txt.append(self._embed_text(c))

        if to_embed_txt:
            new_vecs = self._compute_embeddings(to_embed_txt)
            for i, vec in zip(to_embed_idx, new_vecs):
                vectors[i] = vec

        removed = len(set(existing) - {c["chunk_id"] for c in ordered})
        self.chunks = ordered
        self.embeddings = np.array(vectors, dtype=np.float32) if vectors else np.zeros((0, 384), dtype=np.float32)
        self._faiss = None

        duration = max(time.time() - start_time, 0.001)
        self.manifest = {
            "build_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_versions": source_versions,
            "total_chunks": len(self.chunks),
            "newly_embedded": len(to_embed_txt),
            "reused_chunks": len(self.chunks) - len(to_embed_txt),
            "removed_chunks": removed,
            "duration_seconds": round(duration, 4),
            "throughput_chunks_per_second": round(len(new_chunks) / duration, 2),
            "embedding_model": self.embedding_model_name,
            "embedding_backend": backend,
            "index_type": "FAISS (LangChain, inner product on normalised vectors)",
        }

        with open(self.chunks_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)
        np.save(self.embeddings_file, self.embeddings)
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

        index = self._get_faiss()
        if index is not None:
            index.save_local(self.faiss_dir)

        return self.manifest

    # ------------------------------------------------------------------ retrieval
    def _result(self, i: int, score: float) -> Dict[str, Any]:
        chunk = self.chunks[i]
        return {
            "chunk_id": chunk["chunk_id"],
            "clause_id": chunk["clause_id"],
            "document_title": chunk["document_title"],
            "section_title": chunk["section_title"],
            "text": chunk["text"],
            "score": float(round(score, 4)),
            "metadata": chunk.get("metadata", {}),
        }

    def search(
        self,
        query: str,
        top_k: int = 4,
        category: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Dense similarity search through LangChain FAISS, with category filtering."""
        index = self._get_faiss()
        if index is None:
            return []
        if min_score is None:
            min_score = float(os.getenv("RETRIEVAL_SIMILARITY_THRESHOLD", "0.25"))
            if self.embedding_backend == BUILTIN_BACKEND:
                min_score = min(min_score, 0.05)

        qvec = self._compute_embeddings([query])[0].tolist()
        hits = index.similarity_search_with_score_by_vector(qvec, k=len(self.chunks))
        results = []
        for doc, score in hits:
            i = doc.metadata["idx"]
            chunk = self.chunks[i]
            if len(chunk["text"]) < 60 and chunk["clause_id"].endswith("GEN"):
                continue  # bare headings carry no provision
            cat = chunk.get("metadata", {}).get("category")
            if category and cat not in (category, "facilities"):
                continue
            if float(score) >= min_score:
                results.append(self._result(i, float(score)))
            if len(results) >= top_k:
                break
        return results

    def keyword_search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Lexical fallback strategy used when dense retrieval fails verification."""
        terms = [t.strip(".,:;()").lower() for t in query.split() if len(t) > 3]
        scored = []
        for i, c in enumerate(self.chunks):
            text = self._embed_text(c).lower()
            hits = sum(1 for t in terms if t in text)
            if hits:
                scored.append((hits / max(len(terms), 1), i))
        scored.sort(reverse=True)
        return [self._result(i, s) for s, i in scored[:top_k]]

    @staticmethod
    def normalise_clause_id(clause_id: str) -> str:
        cid = clause_id.strip().upper().replace(" ", "")
        for prefix in ("CLAUSE", "PARAGRAPH", "PARA"):
            if cid.startswith(prefix):
                cid = cid[len(prefix):]
        cid = cid.replace("CAR.", "CAR-").replace("CAR", "CAR-").replace("--", "-")
        if cid and cid[0].isdigit():
            cid = "CAR-" + cid
        if cid.startswith("CAR-") and cid.count(".") >= 3:
            cid = ".".join(cid.split(".")[:3])  # 3.3.2.a -> 3.3.2 (lettered tiers sit inside 3.3.2)
        return cid.rstrip(".")

    def lookup_clause(self, clause_id: str) -> Optional[Dict[str, Any]]:
        """
        Exact clause-id lookup. Returns the clause with all of its sub-clauses merged,
        e.g. CAR-3.7 -> 3.7 heading + 3.7.1 + 3.7.2. Never returns a different clause.
        """
        norm = self.normalise_clause_id(clause_id)
        matches = [c for c in self.chunks
                   if c["clause_id"].upper() == norm or c["clause_id"].upper().startswith(norm + ".")]
        if not matches:
            return None
        head = next((c for c in matches if c["clause_id"].upper() == norm), matches[0])
        text = "\n".join(c["text"] for c in matches)
        return {
            "chunk_id": head["chunk_id"],
            "clause_id": head["clause_id"] if head["clause_id"].upper() == norm else norm,
            "document_title": head["document_title"],
            "section_title": head["section_title"],
            "text": text,
            "metadata": head.get("metadata", {}),
            "score": 1.0,
        }

    def get_freshness(self) -> Dict[str, Any]:
        """Expose data freshness metrics for observability (D2, F8)."""
        return self.manifest
