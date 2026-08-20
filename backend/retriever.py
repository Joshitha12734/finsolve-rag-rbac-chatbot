"""
Retrieval layer of the RAG pipeline.

Two "signal" backends, combined into one hybrid score:

1. LexicalRetriever - BM25 (rank_bm25), pure Python, no external
   downloads. Strong on exact keyword/entity matches (names, numbers,
   specific terms) — the classic weakness of embeddings.
2. EmbeddingRetriever - sentence-transformers embeddings persisted in a
   persistent Chroma vector store. Strong on semantic/paraphrase matches.
   Needs internet on first run to download the embedding model.

HybridRetriever blends both (normalized score fusion) when the embedding
backend is available, and falls back to BM25-only when it isn't (e.g. no
internet to Hugging Face) — same graceful-degradation pattern as before,
now with a genuinely better lexical baseline than TF-IDF.

An optional LLM reranking pass (`rerank_with_llm`) re-scores the fused
top-N candidates for relevance right before generation — this is the
"cross-encoder reranker" idea, implemented via an LLM call (see
llm_client.py — Groq by default) instead of a downloaded cross-encoder
model, so it works without extra model downloads.

RBAC is enforced at TWO levels here, both applied inside `query()`:
  - department: from the caller's role (folder-level, as before)
  - classification: from config.role_can_access_classification (e.g. a
    'confidential' doc is c-level-only even within a department the
    caller otherwise has access to)
Both checks happen before ranking/reranking ever sees a chunk's text.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any

from .config import INDEX_DIR, role_can_access_classification
from .ingest import load_all_chunks, chunks_to_dicts
from . import llm_client

TOP_K_DEFAULT = 5
RERANK_CANDIDATE_MULTIPLIER = 3  # fetch this many x top_k before reranking

# Minimal, hand-rolled stopword list — no extra dependency needed. Without
# this, common words like "what"/"is"/"the" (present in nearly every chunk)
# can dominate a BM25 score for text-dense chunks (e.g. FAQ sections) that
# don't actually contain any of the query's meaningful terms. This was a
# real bug: "what is the quarterly revenue?" was matching an HR FAQ section
# that contained neither "quarterly" nor "revenue" — purely on the strength
# of repeated "the"/"is" tokens, since punctuation wasn't stripped either
# (so "revenue?" never matched "revenue" in the corpus in the first place).
_STOPWORDS = frozenset("""
a an the is are was were be been being to of in on at for with by from
as it its this that these those and or but if then so than too very
what who whom which when where why how do does did doing have has had
having i you he she we they them his her our your their not no nor can
could will would shall should may might must about into over under
""".split())

_PUNCT_RE = re.compile(r"[^\w\s]")


def _tokenize(text: str) -> List[str]:
    text = _PUNCT_RE.sub(" ", text.lower())
    return [tok for tok in text.split() if tok and tok not in _STOPWORDS]


def _passes_rbac(chunk: dict, allowed_departments: set, role: str) -> bool:
    if chunk["department"] not in allowed_departments:
        return False
    if not role_can_access_classification(role, chunk.get("classification", "internal")):
        return False
    return True


def _normalize(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


class BaseRetriever:
    def build(self) -> None:
        raise NotImplementedError

    def query(self, query_text: str, allowed_departments: set, role: str, top_k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def best_possible_score(self, query_text: str) -> float:
        raise NotImplementedError

    def best_match_department(self, query_text: str) -> tuple[str, float] | None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Lexical signal: BM25 (offline, no external downloads)
# ---------------------------------------------------------------------------
class LexicalRetriever(BaseRetriever):
    NAME = "bm25"

    def __init__(self):
        self.bm25 = None
        self.chunks: List[Dict[str, Any]] = []
        self.last_rerank_method = "none"
        self._paths = {
            "bm25": INDEX_DIR / "bm25_index.pkl",
            "chunks": INDEX_DIR / "bm25_chunks.json",
        }

    def build(self) -> None:
        from rank_bm25 import BM25Okapi

        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        chunks = chunks_to_dicts(load_all_chunks())
        tokenized = [_tokenize(c["text"]) for c in chunks]
        bm25 = BM25Okapi(tokenized)

        with open(self._paths["bm25"], "wb") as f:
            pickle.dump(bm25, f)
        with open(self._paths["chunks"], "w", encoding="utf-8") as f:
            json.dump(chunks, f)

        self.bm25, self.chunks = bm25, chunks

    def _ensure_loaded(self) -> None:
        if self.bm25 is not None:
            return
        if not all(p.exists() for p in self._paths.values()):
            self.build()
            return
        with open(self._paths["bm25"], "rb") as f:
            self.bm25 = pickle.load(f)
        with open(self._paths["chunks"], "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def _scored(self, query_text: str) -> List[tuple]:
        self._ensure_loaded()
        scores = self.bm25.get_scores(_tokenize(query_text))
        return list(zip(range(len(scores)), scores))

    def query(self, query_text: str, allowed_departments: set, role: str, top_k: int = TOP_K_DEFAULT, rerank: bool | None = None) -> List[Dict[str, Any]]:
        scored = sorted(self._scored(query_text), key=lambda p: p[1], reverse=True)
        candidates = []
        fetch_n = top_k * RERANK_CANDIDATE_MULTIPLIER
        for idx, score in scored:
            if score <= 0:
                continue
            chunk = self.chunks[idx]
            if not _passes_rbac(chunk, allowed_departments, role):
                continue
            candidates.append({**chunk, "score": float(score)})
            if len(candidates) >= fetch_n:
                break

        should_attempt_rerank = rerank if rerank is not None else True
        self.last_rerank_method = "none"
        if should_attempt_rerank and len(candidates) > top_k:
            reranked = rerank_with_cross_encoder(query_text, candidates, top_k)
            if reranked is not None:
                candidates = reranked
                self.last_rerank_method = "cross-encoder"
            elif llm_client.api_key_configured():
                candidates = rerank_with_llm(query_text, candidates, top_k)
                self.last_rerank_method = "llm"
        return candidates[:top_k]

    def best_possible_score(self, query_text: str) -> float:
        scored = self._scored(query_text)
        return max((s for _, s in scored), default=0.0)

    def best_match_department(self, query_text: str) -> tuple[str, float] | None:
        """The single best-scoring chunk's department, ignoring RBAC
        entirely — used to detect 'the real answer is in a department you
        can't see' rather than confidently answering from a weaker match
        in an allowed department."""
        scored = self._scored(query_text)
        if not scored:
            return None
        best_idx, best_score = max(scored, key=lambda p: p[1])
        if best_score <= 0:
            return None
        return self.chunks[best_idx]["department"], float(best_score)

    @staticmethod
    def confidence(raw_score: float) -> float:
        """BM25 scores are unbounded, so map to 0-1 with a saturating
        curve (a score of ~5 is already a strong match for this corpus
        size) rather than a hard clamp, which would flatten everything
        above 1.0 to 100% confidence."""
        import math
        return 1 - math.exp(-max(raw_score, 0) / 5.0)


# ---------------------------------------------------------------------------
# Semantic signal: sentence-transformers + Chroma
# ---------------------------------------------------------------------------
class EmbeddingRetriever(BaseRetriever):
    NAME = "embeddings"
    COLLECTION_NAME = "finsolve_docs"
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        self._collection = None

    def _get_client(self):
        import chromadb
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(INDEX_DIR / "chroma"))

    def _get_embedding_fn(self):
        from chromadb.utils import embedding_functions
        # device="cpu" is explicit on purpose: with certain torch/accelerate
        # version combinations, letting the library auto-select a device can
        # trigger a "Cannot copy out of meta tensor" error — the model gets
        # lazily initialized on the placeholder "meta" device and never
        # properly materialized. Forcing cpu here is the standard fix. If
        # you hit this error anyway, try: pip install -U torch
        # sentence-transformers accelerate (version mismatches between
        # these three are the usual cause).
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.MODEL_NAME, device="cpu"
        )

    def build(self) -> None:
        client = self._get_client()
        ef = self._get_embedding_fn()
        try:
            client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        collection = client.create_collection(self.COLLECTION_NAME, embedding_function=ef)

        chunks = chunks_to_dicts(load_all_chunks())
        collection.add(
            ids=[c["chunk_id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{"department": c["department"], "source": c["source"], "classification": c.get("classification", "internal")} for c in chunks],
        )
        self._collection = collection

    def _ensure_loaded(self) -> None:
        if self._collection is not None:
            return
        client = self._get_client()
        ef = self._get_embedding_fn()
        try:
            self._collection = client.get_collection(self.COLLECTION_NAME, embedding_function=ef)
        except Exception:
            self.build()

    def query_unfiltered(self, query_text: str, n_results: int) -> List[Dict[str, Any]]:
        """Raw top-n by embedding similarity, no RBAC applied — used
        internally by HybridRetriever, which applies RBAC itself after
        fusing scores. Never call this directly from outside this module."""
        self._ensure_loaded()
        res = self._collection.query(query_texts=[query_text], n_results=n_results)
        results = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        ids = res.get("ids", [[]])[0]
        for doc, meta, dist, cid in zip(docs, metas, dists, ids):
            results.append({
                "chunk_id": cid,
                "department": meta["department"],
                "source": meta["source"],
                "classification": meta.get("classification", "internal"),
                "text": doc,
                "score": 1 - dist,
            })
        return results

    def query(self, query_text: str, allowed_departments: set, role: str, top_k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        candidates = self.query_unfiltered(query_text, n_results=top_k * RERANK_CANDIDATE_MULTIPLIER)
        results = [c for c in candidates if _passes_rbac(c, allowed_departments, role)]
        return results[:top_k]

    def best_possible_score(self, query_text: str) -> float:
        self._ensure_loaded()
        res = self._collection.query(query_texts=[query_text], n_results=1)
        dists = res.get("distances", [[]])[0]
        return float(1 - dists[0]) if dists else 0.0

    def best_match_department(self, query_text: str) -> tuple[str, float] | None:
        candidates = self.query_unfiltered(query_text, n_results=1)
        if not candidates:
            return None
        top = candidates[0]
        return top["department"], top["score"]

    @staticmethod
    def confidence(raw_score: float) -> float:
        """Cosine-similarity-derived scores are already roughly 0-1."""
        return max(0.0, min(1.0, raw_score))


# ---------------------------------------------------------------------------
# Hybrid: fuse BM25 + embedding scores, with optional LLM reranking
# ---------------------------------------------------------------------------
class HybridRetriever(BaseRetriever):
    NAME = "hybrid-bm25-embeddings"
    LEXICAL_WEIGHT = 0.4
    SEMANTIC_WEIGHT = 0.6

    def __init__(self, lexical: LexicalRetriever, semantic: EmbeddingRetriever):
        self.lexical = lexical
        self.semantic = semantic
        self.last_rerank_method = "none"

    def build(self) -> None:
        self.lexical.build()
        self.semantic.build()

    def _fused_candidates(self, query_text: str, fetch_n: int) -> List[Dict[str, Any]]:
        lexical_scored = sorted(self.lexical._scored(query_text), key=lambda p: p[1], reverse=True)[:fetch_n]
        lex_by_id = {}
        lex_chunks = self.lexical.chunks
        lex_scores = _normalize([s for _, s in lexical_scored])
        for (idx, _), norm_score in zip(lexical_scored, lex_scores):
            chunk = lex_chunks[idx]
            lex_by_id[chunk["chunk_id"]] = (chunk, norm_score)

        semantic_candidates = self.semantic.query_unfiltered(query_text, n_results=fetch_n)
        sem_scores = _normalize([c["score"] for c in semantic_candidates])
        sem_by_id = {}
        for c, norm_score in zip(semantic_candidates, sem_scores):
            sem_by_id[c["chunk_id"]] = (c, norm_score)

        all_ids = set(lex_by_id) | set(sem_by_id)
        fused = []
        for cid in all_ids:
            chunk = (lex_by_id.get(cid) or sem_by_id.get(cid))[0]
            lex_score = lex_by_id.get(cid, (None, 0.0))[1]
            sem_score = sem_by_id.get(cid, (None, 0.0))[1]
            fused_score = self.LEXICAL_WEIGHT * lex_score + self.SEMANTIC_WEIGHT * sem_score
            fused.append({**chunk, "score": fused_score, "_lexical": lex_score, "_semantic": sem_score})

        fused.sort(key=lambda c: c["score"], reverse=True)
        return fused

    def query(self, query_text: str, allowed_departments: set, role: str, top_k: int = TOP_K_DEFAULT, rerank: bool | None = None) -> List[Dict[str, Any]]:
        fetch_n = top_k * RERANK_CANDIDATE_MULTIPLIER
        fused = self._fused_candidates(query_text, fetch_n)
        allowed = [c for c in fused if _passes_rbac(c, allowed_departments, role)]

        should_attempt_rerank = rerank if rerank is not None else True
        self.last_rerank_method = "none"
        if should_attempt_rerank and len(allowed) > top_k:
            # Priority: real HF cross-encoder (no LLM key needed) > LLM-based
            # rerank (needs a key) > unranked fused order. Each step returns
            # None/unchanged on failure so this degrades cleanly either way.
            reranked = rerank_with_cross_encoder(query_text, allowed, top_k)
            if reranked is not None:
                allowed = reranked
                self.last_rerank_method = "cross-encoder"
            elif llm_client.api_key_configured():
                allowed = rerank_with_llm(query_text, allowed, top_k)
                self.last_rerank_method = "llm"
        return allowed[:top_k]

    def best_possible_score(self, query_text: str) -> float:
        lex_best = self.lexical.best_possible_score(query_text)
        sem_best = self.semantic.best_possible_score(query_text)
        # These are on different scales (BM25 unbounded, cosine ~0-1); just
        # normalize BM25 loosely so a "did anything match at all" signal is
        # still meaningful for the vague-query heuristic in llm.py.
        return max(min(lex_best / 10.0, 1.0), sem_best)

    def best_match_department(self, query_text: str) -> tuple[str, float] | None:
        fused = self._fused_candidates(query_text, fetch_n=10)
        if not fused:
            return None
        top = fused[0]
        return top["department"], top["score"]

    @staticmethod
    def confidence(raw_score: float) -> float:
        """The fused score is already a weighted sum of two 0-1-normalized
        signals, so it's already in range."""
        return max(0.0, min(1.0, raw_score))


_cross_encoder_model = None  # lazy-loaded, shared across calls once loaded


def rerank_with_cross_encoder(query_text: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]] | None:
    """Real neural reranking via a Hugging Face cross-encoder
    (cross-encoder/ms-marco-MiniLM-L-6-v2) — scores each (query, passage)
    pair jointly through a small transformer, which is meaningfully more
    accurate than the fused BM25+embedding score alone (those score query
    and passage independently, then compare vectors; a cross-encoder
    reads them together). This is the standard "retrieve wide, rerank
    narrow" pattern used in production retrieval systems.

    Returns None (rather than the unranked candidates) on any failure —
    e.g. no internet to download the model — so the caller knows to fall
    back to the LLM-based reranker instead, rather than silently getting
    a no-op that looks identical to "reranking happened but found nothing
    worth reordering"."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return None

    global _cross_encoder_model
    try:
        if _cross_encoder_model is None:
            _cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
        pairs = [(query_text, c["text"][:512]) for c in candidates]
        scores = _cross_encoder_model.predict(pairs)
        reranked = sorted(zip(candidates, scores), key=lambda p: p[1], reverse=True)
        return [c for c, _ in reranked]
    except Exception as e:
        print(f"[retriever] cross-encoder reranking unavailable ({e!r}); falling back to LLM reranker")
        return None


def rerank_with_llm(query_text: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Cross-encoder-style reranking, implemented with an LLM call instead
    of a downloaded cross-encoder model (no extra model download
    required). Asks the model to score each candidate's relevance to the
    query directly, then re-sorts by that score. Falls back to the
    original (fused-score) order on any failure — reranking is a quality
    improvement, never a hard dependency."""
    if not llm_client.api_key_configured():
        return candidates

    numbered = "\n\n".join(f"[{i}] {c['text'][:400]}" for i, c in enumerate(candidates))
    prompt = (
        f"Query: \"{query_text}\"\n\n"
        f"Candidate passages:\n{numbered}\n\n"
        "Rate each passage's relevance to the query from 0-10. "
        'Respond as JSON only: {"scores": [n0, n1, n2, ...]} in the same order as the passages.'
    )
    try:
        text = llm_client.chat([{"role": "user", "content": prompt}], max_tokens=200).strip()
        text = text.strip("`").removeprefix("json").strip()
        parsed = json.loads(text)
        scores = parsed["scores"]
        if len(scores) != len(candidates):
            return candidates
        reranked = sorted(zip(candidates, scores), key=lambda p: p[1], reverse=True)
        return [c for c, _ in reranked]
    except Exception:
        return candidates


# ---------------------------------------------------------------------------
# Factory: pick the best backend available, with graceful fallback
# ---------------------------------------------------------------------------
_retriever_instance: BaseRetriever | None = None


def get_retriever(force_backend: str | None = None) -> BaseRetriever:
    global _retriever_instance
    if _retriever_instance is not None and force_backend is None:
        return _retriever_instance

    backend_pref = force_backend or "auto"

    lexical = LexicalRetriever()
    lexical._ensure_loaded()

    if backend_pref in ("auto", "hybrid", "embeddings"):
        try:
            semantic = EmbeddingRetriever()
            semantic._ensure_loaded()  # will attempt to build/download if needed
            hybrid = HybridRetriever(lexical, semantic)
            _retriever_instance = hybrid
            return hybrid
        except Exception as e:
            if backend_pref in ("hybrid", "embeddings"):
                raise
            print(f"[retriever] embedding backend unavailable ({e!r}); using BM25-only")

    _retriever_instance = lexical
    return lexical
