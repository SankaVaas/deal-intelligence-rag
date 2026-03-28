"""
fusion.py
---------
Reciprocal Rank Fusion (RRF) combining dense vector search and BM25 results.

RRF is the standard algorithm for combining multiple ranked retrieval lists.
It uses only rank positions (not scores), making it robust to the different
score scales of dense retrieval (cosine similarity: 0-1) and BM25 (raw
term-frequency scores: 0-20+).

Algorithm:
    For each unique chunk across all retrieval lists:
        rrf_score = Σ  1 / (k + rank_i)
    where rank_i is the chunk's rank in list i, and k=60 is a constant
    that dampens the effect of very high rankings.

    k=60 is the standard value from the original RRF paper (Cormack et al.
    2009). It means rank #1 contributes 1/61 ≈ 0.0164, rank #10 contributes
    1/70 ≈ 0.0143 — a gentle curve that rewards consistency across lists
    more than dominance in a single list.

Why RRF over score normalisation?
    Dense scores (0.62) and BM25 scores (14.07) can't be directly compared
    or averaged — they live in completely different ranges and distributions.
    Normalising to [0,1] helps but doesn't account for the different
    statistical properties of each scoring function.

    RRF sidesteps this entirely by only using rank positions. A chunk that
    ranks #3 in both lists is clearly better than one that ranks #1 in one
    list and doesn't appear at all in the other.

Pipeline position:
    vector_store.py ──┐
                      ├── fusion.py → reranker.py → multi_hop_chain.py
    bm25_retriever.py ┘

Usage (Python):
    from deal_intelligence_rag.retrieval.fusion import HybridRetriever

    retriever = HybridRetriever()

    results = retriever.search(
        query="Apple iPhone revenue by segment",
        n_results=5,
        filters={"ticker": "AAPL"},
    )

    for r in results:
        print(r.rrf_score, r.ticker, r.section_heading[:50])
        print(r.text[:200])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from deal_intelligence_rag.retrieval.bm25_retriever import BM25Retriever
from deal_intelligence_rag.retrieval.reranker import RankedResult, Reranker
from deal_intelligence_rag.retrieval.vector_store import VectorStore

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# RRF constant k — standard value from Cormack et al. 2009
# Higher k = less weight on top ranks, more uniform distribution
RRF_K = 60

# How many candidates to fetch from each retriever before fusion
# More candidates = better recall but slower reranking
DENSE_FETCH_N = 20
BM25_FETCH_N = 20

# Default number of results to return after reranking
DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class FusedResult:
    """
    A result from hybrid retrieval with provenance information.

    Tracks which retrieval systems contributed to this result and
    their original ranks — useful for debugging and ablation studies.
    """

    chunk_id: str
    text: str
    rrf_score: float
    metadata: dict

    # Provenance — which systems found this chunk and at what rank
    dense_rank: int | None = field(default=None)    # None if not in dense results
    bm25_rank: int | None = field(default=None)     # None if not in BM25 results
    rerank_score: float | None = field(default=None)

    @property
    def ticker(self) -> str:
        return self.metadata.get("ticker", "")

    @property
    def filed_date(self) -> str:
        return self.metadata.get("filed_date", "")

    @property
    def section_heading(self) -> str:
        return self.metadata.get("section_heading", "")

    @property
    def found_by(self) -> str:
        """Human-readable string showing which retrievers found this chunk."""
        sources = []
        if self.dense_rank is not None:
            sources.append(f"dense(#{self.dense_rank})")
        if self.bm25_rank is not None:
            sources.append(f"bm25(#{self.bm25_rank})")
        return " + ".join(sources) if sources else "unknown"

    @property
    def score(self) -> float:
        """Alias for rrf_score — normalized to [0,1] for compatibility with reranker."""
        # Map RRF score to approximately [0,1] range for the reranker
        # RRF scores typically range from 0 to ~0.03, so we scale appropriately
        return min(self.rrf_score * 50, 1.0)

    def __repr__(self) -> str:
        return (
            f"FusedResult(rrf={self.rrf_score:.4f}, "
            f"ticker={self.ticker}, date={self.filed_date}, "
            f"found_by={self.found_by}, "
            f"section={self.section_heading[:40]!r})"
        )


# ---------------------------------------------------------------------------
# Fusion functions
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    ranked_lists: list[list],
    k: int = RRF_K,
) -> list[tuple[str, float, dict]]:
    """
    Apply Reciprocal Rank Fusion to multiple ranked retrieval lists.

    Args:
        ranked_lists: List of retrieval result lists. Each item in a list
                      must have .chunk_id, .text, .metadata attributes.
        k:            RRF constant (default 60).

    Returns:
        List of (chunk_id, rrf_score, metadata) tuples sorted by
        rrf_score descending. Includes all unique chunks across all lists.
    """
    # Accumulate RRF scores: chunk_id → cumulative score
    rrf_scores: dict[str, float] = {}

    # Store text and metadata for later reconstruction
    chunk_data: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list, start=1):
            cid = result.chunk_id
            contribution = 1.0 / (k + rank)

            if cid not in rrf_scores:
                rrf_scores[cid] = 0.0
                chunk_data[cid] = {
                    "text": result.text,
                    "metadata": result.metadata,
                }

            rrf_scores[cid] += contribution

    # Sort by RRF score descending
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [
        (cid, score, chunk_data[cid])
        for cid, score in sorted_chunks
    ]


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """
    Full hybrid retrieval pipeline: dense + BM25 → RRF fusion → reranking.

    This is the primary retrieval interface used by the agent layer
    (multi_hop_chain.py, tools.py). It encapsulates the full pipeline
    so agent code doesn't need to know about individual retrievers.

    Architecture:
        1. Dense retrieval  → top DENSE_FETCH_N candidates
        2. BM25 retrieval   → top BM25_FETCH_N candidates
        3. RRF fusion       → merged, deduplicated, ranked list
        4. Cross-encoder    → re-scores top candidates precisely
        5. Return top_k     → final results with full provenance

    The reranker is optional — set use_reranker=False for faster
    retrieval during development or when latency is critical.
    """

    def __init__(
        self,
        use_reranker: bool = True,
        dense_fetch_n: int = DENSE_FETCH_N,
        bm25_fetch_n: int = BM25_FETCH_N,
        rrf_k: int = RRF_K,
    ) -> None:
        self.use_reranker = use_reranker
        self.dense_fetch_n = dense_fetch_n
        self.bm25_fetch_n = bm25_fetch_n
        self.rrf_k = rrf_k

        log.info("initialising_hybrid_retriever")

        # Dense retriever
        self.vector_store = VectorStore()

        # Sparse retriever — build index on init
        self.bm25 = BM25Retriever()
        self.bm25.build_index()

        # Cross-encoder reranker (loaded lazily on first use if enabled)
        self._reranker: Reranker | None = None
        if use_reranker:
            self._reranker = Reranker()

        log.info(
            "hybrid_retriever_ready",
            total_chunks=self.vector_store.collection.count(),
            bm25_chunks=self.bm25.index_size(),
            reranker=use_reranker,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        n_results: int = DEFAULT_TOP_K,
        filters: dict | None = None,
        section_filter: str | None = None,
        use_reranker: bool | None = None,
    ) -> list[FusedResult]:
        """
        Full hybrid search: dense + BM25 → RRF → optional reranking.

        Args:
            query:         Natural language query.
            n_results:     Number of results to return (default 5).
            filters:       Metadata filters: {"ticker": "AAPL", "year": 2024}
            section_filter: Restrict to sections containing this string.
            use_reranker:  Override instance-level reranker setting.

        Returns:
            List of FusedResult sorted by relevance (rerank_score if
            reranker enabled, otherwise rrf_score).
        """
        should_rerank = use_reranker if use_reranker is not None else self.use_reranker

        log.info(
            "hybrid_search",
            query=query[:80],
            n_results=n_results,
            filters=filters,
            reranker=should_rerank,
        )

        # Step 1: dense retrieval
        dense_results = self.vector_store.search(
            query=query,
            n_results=self.dense_fetch_n,
            filters=filters,
            section_filter=section_filter,
        )

        # Step 2: BM25 sparse retrieval
        bm25_results = self.bm25.search(
            query=query,
            n_results=self.bm25_fetch_n,
            filters=filters,
        )

        log.info(
            "retrieval_counts",
            dense=len(dense_results),
            bm25=len(bm25_results),
        )

        # Step 3: RRF fusion
        fused = self._fuse(dense_results, bm25_results)

        log.info("after_fusion", unique_chunks=len(fused))

        # Step 4: optional reranking
        if should_rerank and self._reranker and len(fused) > 0:
            # Take top (n_results * 3) fused candidates into the reranker
            # to balance quality vs latency
            rerank_candidates = fused[: n_results * 3]
            reranked = self._reranker.rerank(
                query=query,
                candidates=rerank_candidates,
                top_k=n_results,
            )
            final = self._apply_rerank_scores(fused, reranked)
        else:
            final = fused[:n_results]

        log.info("hybrid_search_complete", returned=len(final))
        return final

    def search_no_rerank(
        self,
        query: str,
        n_results: int = DEFAULT_TOP_K,
        filters: dict | None = None,
    ) -> list[FusedResult]:
        """Convenience method — hybrid search without reranking. Faster."""
        return self.search(
            query=query,
            n_results=n_results,
            filters=filters,
            use_reranker=False,
        )

    def get_stats(self) -> dict:
        """Return stats about the retrieval system."""
        return {
            "total_chunks": self.vector_store.collection.count(),
            "bm25_chunks": self.bm25.index_size(),
            "tickers": self.vector_store.get_available_tickers(),
            "years": self.vector_store.get_available_years(),
            "reranker_enabled": self.use_reranker,
            "rrf_k": self.rrf_k,
        }

    # ------------------------------------------------------------------
    # Internal: fusion
    # ------------------------------------------------------------------

    def _fuse(
        self,
        dense_results: list,
        bm25_results: list,
    ) -> list[FusedResult]:
        """
        Apply RRF to dense and BM25 result lists.

        Builds rank lookup dicts so each FusedResult knows which
        systems found it and at what rank (for provenance tracking).
        """
        # Build rank lookups for provenance
        dense_rank_map = {r.chunk_id: i + 1 for i, r in enumerate(dense_results)}
        bm25_rank_map = {r.chunk_id: i + 1 for i, r in enumerate(bm25_results)}

        # Run RRF
        fused_tuples = reciprocal_rank_fusion(
            [dense_results, bm25_results],
            k=self.rrf_k,
        )

        results = []
        for chunk_id, rrf_score, data in fused_tuples:
            results.append(
                FusedResult(
                    chunk_id=chunk_id,
                    text=data["text"],
                    rrf_score=round(rrf_score, 6),
                    metadata=data["metadata"],
                    dense_rank=dense_rank_map.get(chunk_id),
                    bm25_rank=bm25_rank_map.get(chunk_id),
                )
            )

        return results

    def _apply_rerank_scores(
        self,
        fused: list[FusedResult],
        reranked: list[RankedResult],
    ) -> list[FusedResult]:
        """
        Merge reranker scores back into FusedResult objects.

        The reranker returns RankedResult objects — we map the rerank
        scores back to our FusedResult objects and re-sort by rerank_score.
        """
        # Map chunk_id → rerank_score
        rerank_map = {r.chunk_id: r.rerank_score for r in reranked}
        reranked_ids = [r.chunk_id for r in reranked]

        # Update fused results that were reranked
        for result in fused:
            if result.chunk_id in rerank_map:
                result.rerank_score = rerank_map[result.chunk_id]

        # Return only the reranked chunks, in reranked order
        reranked_results = [
            next(r for r in fused if r.chunk_id == cid)
            for cid in reranked_ids
            if any(r.chunk_id == cid for r in fused)
        ]

        return reranked_results


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    """
    Run with: python -m deal_intelligence_rag.retrieval.fusion
    """
    logging.basicConfig(level=logging.WARNING)

    print("Initialising hybrid retriever (this loads all components)...")
    retriever = HybridRetriever(use_reranker=True)

    stats = retriever.get_stats()
    print(f"\n  Chunks in store : {stats['total_chunks']:,}")
    print(f"  BM25 index size : {stats['bm25_chunks']:,}")
    print(f"  Tickers         : {stats['tickers']}")
    print(f"  Years           : {stats['years']}")

    test_cases = [
        {
            "query": "What are Apple's main revenue segments and their performance?",
            "filters": {"ticker": "AAPL", "year": 2024},
        },
        {
            "query": "Azure cloud services revenue growth percentage",
            "filters": {"ticker": "MSFT"},
        },
        {
            "query": "Supply chain manufacturing risks outsourcing partners",
            "filters": None,
        },
    ]

    for tc in test_cases:
        query = tc["query"]
        filters = tc["filters"]

        print(f"\n{'='*65}")
        print(f"Query  : {query!r}")
        print(f"Filters: {filters}")

        results = retriever.search(query, n_results=5, filters=filters)

        print(f"\nTop {len(results)} results:")
        for i, r in enumerate(results, 1):
            rerank_str = f"rerank={r.rerank_score:.3f} " if r.rerank_score else ""
            print(
                f"  {i}. [{rerank_str}rrf={r.rrf_score:.4f}] "
                f"{r.ticker} {r.filed_date} "
                f"| {r.section_heading[:45]}"
            )
            print(f"     found_by: {r.found_by}")
            print(f"     {r.text[:130].strip()}...")


if __name__ == "__main__":
    _smoke_test()