"""
reranker.py
-----------
Cross-encoder re-ranking of candidate chunks retrieved by dense + BM25 search.

A cross-encoder takes (query, document) as a SINGLE input and produces
one relevance score. This is fundamentally different from bi-encoders
(used in embedder.py) which encode query and document separately:

    Bi-encoder:    encode(query) · encode(document)  → approximate similarity
    Cross-encoder: encode(query + document)           → precise relevance score

Cross-encoders are ~10-50x more accurate than bi-encoders for relevance
scoring but too slow to run on the full corpus. The correct pattern is:

    1. Retrieve top-20 candidates quickly  (bi-encoder + BM25)
    2. Re-rank those 20 precisely          (cross-encoder)  ← this file
    3. Return top-5 to the LLM

Model: ms-marco-MiniLM-L6-v2
    - Trained on MS MARCO passage ranking dataset (650k query-passage pairs)
    - Distilled from a larger model — fast enough for CPU inference
    - ~22MB download, cached locally by sentence-transformers after first run
    - Latency: ~15-40ms per (query, document) pair on CPU
    - At 20 candidates: ~300-800ms total — acceptable for interactive use

Why not a larger model?
    ms-marco-MiniLM-L12-v2 is more accurate but ~2x slower on CPU.
    For our use case (financial text, structured sections) the L6 model
    is accurate enough and keeps total latency under 1 second.

Pipeline position:
    fusion.py → reranker.py → query/multi_hop_chain.py

Usage (Python):
    from deal_intelligence_rag.retrieval.reranker import Reranker
    from deal_intelligence_rag.retrieval.vector_store import VectorStore

    store = VectorStore()
    reranker = Reranker()

    candidates = store.search("Apple revenue breakdown", n_results=20)
    reranked = reranker.rerank(
        query="Apple revenue breakdown by segment",
        candidates=candidates,
        top_k=5,
    )

    for r in reranked:
        print(r.rerank_score, r.original_score, r.text[:100])
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import structlog
from sentence_transformers import CrossEncoder

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ms-marco-MiniLM-L6-v2: fast, CPU-friendly, trained on passage ranking
# First run downloads ~22MB to ~/.cache/huggingface/
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

DEFAULT_TOP_K = 5
DEFAULT_BATCH_SIZE = 32   # process pairs in batches for efficiency


# ---------------------------------------------------------------------------
# Protocol — allows reranker to accept both VectorStore and BM25 results
# ---------------------------------------------------------------------------


@runtime_checkable
class HasTextAndMetadata(Protocol):
    """
    Structural protocol for any search result type.
    Both SearchResult (vector_store.py) and BM25Result (bm25_retriever.py)
    satisfy this protocol — the reranker works with either.
    """
    chunk_id: str
    text: str
    score: float
    metadata: dict


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RankedResult:
    """
    A re-ranked search result with both original and cross-encoder scores.

    Preserving both scores is important for:
      - Debugging: see how much re-ranking changed the order
      - Fusion: original_score used in reciprocal rank fusion (fusion.py)
      - Evaluation: compare reranker vs baseline in ablation study
    """

    chunk_id: str
    text: str
    rerank_score: float      # cross-encoder relevance score (higher = better)
    original_score: float    # score from the upstream retriever
    metadata: dict
    rank: int = field(default=0)   # final rank position (1-indexed)

    @property
    def ticker(self) -> str:
        return self.metadata.get("ticker", "")

    @property
    def filed_date(self) -> str:
        return self.metadata.get("filed_date", "")

    @property
    def section_heading(self) -> str:
        return self.metadata.get("section_heading", "")

    def __repr__(self) -> str:
        return (
            f"RankedResult(rank={self.rank}, ticker={self.ticker}, "
            f"date={self.filed_date}, rerank={self.rerank_score:.3f}, "
            f"orig={self.original_score:.3f}, "
            f"section={self.section_heading[:40]!r})"
        )


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class Reranker:
    """
    Cross-encoder re-ranker using ms-marco-MiniLM-L6-v2.

    Architecture note:
        The model is loaded once at construction time and reused across
        all rerank() calls. Loading takes ~1-2 seconds on first call
        (downloads model if not cached), then is near-instant on reuse.

        The CrossEncoder.predict() method handles batching internally.
        We pass all (query, document) pairs at once and let sentence-
        transformers handle the batching efficiently.

        Score normalisation: CrossEncoder returns raw logits (unbounded).
        We apply sigmoid to map to [0, 1] range for interpretability,
        though ranking order is preserved either way.
    """

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.batch_size = batch_size
        self.model_name = model_name

        log.info("loading_reranker_model", model=model_name)
        t0 = time.time()

        self._model = CrossEncoder(
            model_name,
            max_length=512,    # max tokens per (query + document) pair
        )

        elapsed = time.time() - t0
        log.info("reranker_model_loaded", model=model_name, seconds=round(elapsed, 2))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[RankedResult]:
        """
        Re-rank a list of candidate chunks using cross-encoder scoring.

        Args:
            query:      The original user query.
            candidates: List of search results (SearchResult or BM25Result).
                        Must have .chunk_id, .text, .score, .metadata attrs.
            top_k:      Number of top results to return after re-ranking.

        Returns:
            List of RankedResult sorted by rerank_score descending,
            truncated to top_k. Each result has both rerank_score and
            original_score for comparison.

        Performance note:
            At 20 candidates with 800-token chunks on CPU:
              - L6 model: ~400-600ms
              - L12 model: ~800-1200ms
            This is the bottleneck in the retrieval pipeline — acceptable
            for a RAG system where the LLM call takes 2-5 seconds anyway.
        """
        if not candidates:
            return []

        top_k = min(top_k, len(candidates))

        log.info(
            "reranking",
            query=query[:80],
            candidates=len(candidates),
            top_k=top_k,
        )

        t0 = time.time()

        # Build (query, document) pairs for the cross-encoder
        # We truncate document text to ~2000 chars to stay within the
        # 512-token limit of the cross-encoder (query takes ~20-30 tokens)
        pairs = [
            (query, self._truncate_for_crossencoder(c.text))
            for c in candidates
        ]

        # Score all pairs — CrossEncoder.predict handles batching
        raw_scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Apply sigmoid to convert logits → [0, 1] probability scores
        rerank_scores = self._sigmoid(raw_scores)

        elapsed = time.time() - t0

        log.info(
            "reranking_complete",
            candidates=len(candidates),
            elapsed_ms=round(elapsed * 1000),
        )

        # Build RankedResult objects
        results = [
            RankedResult(
                chunk_id=c.chunk_id,
                text=c.text,
                rerank_score=round(float(score), 4),
                original_score=round(float(c.score), 4),
                metadata=c.metadata,
            )
            for c, score in zip(candidates, rerank_scores)
        ]

        # Sort by rerank_score descending
        results.sort(key=lambda r: r.rerank_score, reverse=True)

        # Assign rank positions (1-indexed)
        for i, result in enumerate(results):
            result.rank = i + 1

        top_results = results[:top_k]

        # Log rank changes for debugging (original rank vs new rank)
        self._log_rank_changes(candidates, top_results)

        return top_results

    def rerank_with_scores(
        self,
        query: str,
        candidates: list,
    ) -> list[RankedResult]:
        """
        Re-rank all candidates and return ALL of them (no top_k truncation).
        Useful for ablation studies and evaluation.
        """
        return self.rerank(query, candidates, top_k=len(candidates))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _truncate_for_crossencoder(self, text: str, max_chars: int = 1800) -> str:
        """
        Truncate document text to fit within the cross-encoder's token limit.

        The cross-encoder has a 512-token limit for the combined
        (query + document) input. A query is typically 10-30 tokens,
        leaving ~480 tokens for the document (~1,800-2,000 characters).

        We truncate at a word boundary rather than mid-character to avoid
        feeding partial words to the tokeniser.
        """
        if len(text) <= max_chars:
            return text

        truncated = text[:max_chars]
        # Find last whitespace to avoid cutting mid-word
        last_space = truncated.rfind(" ")
        if last_space > max_chars * 0.8:   # only snap to word if close enough
            truncated = truncated[:last_space]

        return truncated + "..."

    @staticmethod
    def _sigmoid(scores) -> list[float]:
        """
        Apply sigmoid function to convert raw logits to [0, 1] scores.

        sigmoid(x) = 1 / (1 + e^(-x))

        CrossEncoder returns raw logits from the linear classification head.
        Sigmoid maps these to probabilities — a score of 0.9 means the model
        is 90% confident this chunk is relevant to the query.
        """
        import math
        return [1.0 / (1.0 + math.exp(-float(s))) for s in scores]

    def _log_rank_changes(
        self,
        original_candidates: list,
        reranked: list[RankedResult],
    ) -> None:
        """
        Log how much re-ranking changed the order — useful for debugging
        and for the ablation study in eval/ablation.py.
        """
        original_order = {c.chunk_id: i + 1 for i, c in enumerate(original_candidates)}
        for result in reranked:
            orig_rank = original_order.get(result.chunk_id, -1)
            if orig_rank != result.rank:
                log.info(
                    "rank_change",
                    chunk_id=result.chunk_id[:30],
                    from_rank=orig_rank,
                    to_rank=result.rank,
                    ticker=result.ticker,
                    section=result.section_heading[:40],
                )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    """
    Run with: python -m deal_intelligence_rag.retrieval.reranker
    Tests the reranker against live vector store results.
    """
    import logging
    logging.basicConfig(level=logging.WARNING)

    from deal_intelligence_rag.retrieval.vector_store import VectorStore

    print("Loading vector store...")
    store = VectorStore()

    if store.collection.count() == 0:
        print("No chunks in vector store — run the embedder first.")
        return

    print(f"Loading reranker model ({RERANKER_MODEL})...")
    reranker = Reranker()

    test_cases = [
        {
            "query": "What are Apple's main sources of revenue?",
            "filters": {"ticker": "AAPL"},
        },
        {
            "query": "Microsoft Azure cloud services revenue growth",
            "filters": {"ticker": "MSFT", "year": 2024},
        },
        {
            "query": "Supply chain concentration risks in Asia",
            "filters": None,
        },
    ]

    for tc in test_cases:
        query = tc["query"]
        filters = tc["filters"]

        print(f"\n{'='*60}")
        print(f"Query: {query!r}")
        if filters:
            print(f"Filters: {filters}")

        # Step 1: dense retrieval — get top 15 candidates
        candidates = store.search(query, n_results=15, filters=filters)
        print(f"\nBefore reranking (top 5 of {len(candidates)} candidates):")
        for i, c in enumerate(candidates[:5], 1):
            print(f"  {i}. [{c.score:.3f}] {c.ticker} {c.filed_date} | {c.section_heading[:45]}")

        # Step 2: re-rank
        reranked = reranker.rerank(query, candidates, top_k=5)
        print(f"\nAfter reranking (top 5):")
        for r in reranked:
            rank_arrow = "↑" if r.rank < (list(candidates).index(
                next(c for c in candidates if c.chunk_id == r.chunk_id)
            ) + 1) else "↓" if r.rank > (list(candidates).index(
                next(c for c in candidates if c.chunk_id == r.chunk_id)
            ) + 1) else "="
            print(
                f"  {r.rank}. [{r.rerank_score:.3f}] "
                f"{r.ticker} {r.filed_date} | {r.section_heading[:45]}"
            )
            print(f"     {r.text[:120].strip()}...")


if __name__ == "__main__":
    _smoke_test()