"""
vector_store.py
---------------
Query-time interface to the ChromaDB vector store.

Wraps ChromaDB with a clean search API, handles query embedding via Cohere
(using input_type="search_query" — the asymmetric pair to "search_document"),
and supports metadata filtering for company, date range, and section type.

This is the dense retrieval component. It is combined with BM25 sparse
retrieval in fusion.py to produce the final hybrid retrieval pipeline.

Pipeline position:
    ChromaDB  ←→  vector_store.py  ←  query/multi_hop_chain.py
                  bm25_retriever.py ↗
                  (fusion.py combines both)

Key design decisions:
    1. input_type="search_query" at query time (vs "search_document" at index
       time) — Cohere v3 is an asymmetric model: documents and queries are
       embedded into related but distinct subspaces optimised for comparison.
       Mixing them degrades recall by ~10-15%.

    2. Metadata filtering before vector search — ChromaDB applies metadata
       filters as a pre-filter (reduces the candidate set before ANN search).
       This is more efficient than post-filtering and allows precise company/
       date scoping that M&A analysts need ("only AAPL, last 2 years").

    3. SearchResult dataclass — returning a typed object rather than raw dicts
       makes downstream code (multi_hop_chain.py, judge.py) cleaner and
       allows mypy to catch attribute errors at type-check time.

Usage (Python):
    from deal_intelligence_rag.retrieval.vector_store import VectorStore

    store = VectorStore()

    # Basic search
    results = store.search("What are Apple's main revenue streams?", n_results=5)

    # Filtered search — only AAPL 2024 filings
    results = store.search(
        "revenue breakdown by segment",
        n_results=5,
        filters={"ticker": "AAPL", "year": 2024},
    )

    # Section-scoped search — only risk factors
    results = store.search(
        "supply chain risks",
        n_results=5,
        filters={"ticker": "AAPL"},
        section_filter="Risk Factors",
    )

    for r in results:
        print(r.score, r.metadata["section_heading"], r.text[:200])
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import chromadb
import cohere
import structlog
from dotenv import load_dotenv

load_dotenv()

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma"))
COLLECTION_NAME = "sec_filings"
COHERE_MODEL = "embed-english-v3.0"
DEFAULT_N_RESULTS = 10
MAX_N_RESULTS = 50   # hard cap — ChromaDB gets slow beyond this for our DB size


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single result returned from a vector search."""

    chunk_id: str
    text: str
    score: float          # cosine similarity (0-1, higher = more similar)
    metadata: dict

    # Convenience properties for common metadata fields
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
    def is_table(self) -> bool:
        return self.metadata.get("is_table", False)

    def __repr__(self) -> str:
        return (
            f"SearchResult(ticker={self.ticker}, date={self.filed_date}, "
            f"section={self.section_heading[:40]!r}, score={self.score:.3f})"
        )


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------


class VectorStore:
    """
    Query interface over the ChromaDB collection of embedded SEC filing chunks.

    Responsibilities:
      - Embed incoming queries with Cohere (search_query input type)
      - Execute ANN search in ChromaDB with optional metadata filters
      - Convert raw ChromaDB results to typed SearchResult objects
      - Provide utility methods for collection inspection and filtering

    Not responsible for:
      - BM25 sparse retrieval (see bm25_retriever.py)
      - Result fusion (see fusion.py)
      - Re-ranking (see reranker.py)
    """

    def __init__(
        self,
        cohere_api_key: str | None = None,
        chroma_persist_dir: Path = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "COHERE_API_KEY not found. Set it in your .env file."
            )

        self.co = cohere.Client(api_key)

        self.chroma = chromadb.PersistentClient(path=str(chroma_persist_dir))
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        log.info(
            "vector_store_ready",
            collection=collection_name,
            total_chunks=self.collection.count(),
        )

    # ------------------------------------------------------------------
    # Primary search interface
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        n_results: int = DEFAULT_N_RESULTS,
        filters: dict | None = None,
        section_filter: str | None = None,
        exclude_tables: bool = False,
    ) -> list[SearchResult]:
        """
        Embed the query and retrieve the top-n most similar chunks.

        Args:
            query:          Natural language question or search string.
            n_results:      Number of results to return (default 10, max 50).
            filters:        Exact-match metadata filters, e.g.:
                              {"ticker": "AAPL"}
                              {"ticker": "AAPL", "year": 2024}
                              {"form_type": "10-K"}
            section_filter: Partial match on section_heading, e.g. "Risk Factors".
                            Applied as a post-filter after vector search since
                            ChromaDB doesn't support substring matching natively.
            exclude_tables: If True, removes chunks flagged as primarily tabular.

        Returns:
            List of SearchResult objects sorted by descending cosine similarity.
        """
        n_results = min(n_results, MAX_N_RESULTS)

        # Embed the query — "search_query" is the asymmetric counterpart
        # to "search_document" used at index time
        query_embedding = self._embed_query(query)

        # Build ChromaDB where clause from filters
        where = self._build_where_clause(filters)

        # Fetch more results than needed if we're going to post-filter
        fetch_n = n_results * 3 if (section_filter or exclude_tables) else n_results

        log.info(
            "vector_search",
            query=query[:80],
            n_results=n_results,
            filters=filters,
            section_filter=section_filter,
        )

        try:
            raw = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(fetch_n, self.collection.count()),
                where=where if where else None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            log.error("vector_search_failed", error=str(e))
            return []

        results = self._parse_results(raw)

        # Post-filters
        if section_filter:
            results = [
                r for r in results
                if section_filter.lower() in r.section_heading.lower()
            ]
        if exclude_tables:
            results = [r for r in results if not r.is_table]

        results = results[:n_results]

        log.info("vector_search_complete", returned=len(results))
        return results

    def search_multi_query(
        self,
        queries: list[str],
        n_results: int = DEFAULT_N_RESULTS,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """
        Search with multiple queries and deduplicate results by chunk_id.

        Used by the multi-hop chain to combine sub-query results into a
        single deduplicated candidate set before re-ranking.

        Results are ordered by best score per unique chunk_id.
        """
        seen_ids: dict[str, SearchResult] = {}

        for query in queries:
            results = self.search(query, n_results=n_results, filters=filters)
            for result in results:
                # Keep the highest-scoring occurrence of each chunk
                if (
                    result.chunk_id not in seen_ids
                    or result.score > seen_ids[result.chunk_id].score
                ):
                    seen_ids[result.chunk_id] = result

        # Sort by score descending and return top n
        merged = sorted(seen_ids.values(), key=lambda r: r.score, reverse=True)
        return merged[:n_results]

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_available_tickers(self) -> list[str]:
        """Return sorted list of tickers currently indexed in the collection."""
        try:
            # Sample up to 10k documents to find unique tickers
            sample = self.collection.get(
                limit=min(10_000, self.collection.count()),
                include=["metadatas"],
            )
            tickers = sorted({
                m["ticker"]
                for m in sample["metadatas"]
                if "ticker" in m
            })
            return tickers
        except Exception as e:
            log.error("get_tickers_failed", error=str(e))
            return []

    def get_available_years(self, ticker: str | None = None) -> list[int]:
        """Return sorted list of filing years in the collection."""
        try:
            where = {"ticker": ticker} if ticker else None
            sample = self.collection.get(
                limit=min(10_000, self.collection.count()),
                where=where,
                include=["metadatas"],
            )
            years = sorted({
                int(m["year"])
                for m in sample["metadatas"]
                if "year" in m
            })
            return years
        except Exception as e:
            log.error("get_years_failed", error=str(e))
            return []

    def get_chunk_count(self, filters: dict | None = None) -> int:
        """Return number of chunks matching the given filters."""
        try:
            where = self._build_where_clause(filters)
            if where:
                result = self.collection.get(where=where, include=[])
                return len(result["ids"])
            return self.collection.count()
        except Exception as e:
            log.error("count_failed", error=str(e))
            return 0

    def stats(self) -> dict:
        """Return a summary of the collection contents."""
        tickers = self.get_available_tickers()
        return {
            "total_chunks": self.collection.count(),
            "tickers": tickers,
            "ticker_count": len(tickers),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string using Cohere.

        input_type="search_query" is the critical detail here — this is the
        asymmetric counterpart to "search_document" used at ingestion time.
        Using the correct input_type ensures query and document vectors are
        in comparable subspaces and improves retrieval recall.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.co.embed(
                    texts=[query],
                    model=COHERE_MODEL,
                    input_type="search_query",
                    embedding_types=["float"],
                )
                return response.embeddings.float[0]
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    wait = 65 if attempt == 0 else 30
                    log.warning("query_rate_limit", attempt=attempt + 1, sleeping=wait)
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"Failed to embed query after {max_retries} attempts")

    def _build_where_clause(self, filters: dict | None) -> dict | None:
        """
        Convert a flat filters dict into a ChromaDB where clause.

        ChromaDB supports exact-match filters via:
            {"ticker": "AAPL"}                    → single condition
            {"$and": [{"ticker": ...}, {...}]}     → multiple conditions

        We convert year to int since metadata stores it as int.
        """
        if not filters:
            return None

        conditions = []
        for key, value in filters.items():
            if key == "year":
                conditions.append({key: {"$eq": int(value)}})
            elif isinstance(value, list):
                # e.g. {"ticker": ["AAPL", "MSFT"]} → $in operator
                conditions.append({key: {"$in": value}})
            else:
                conditions.append({key: {"$eq": value}})

        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _parse_results(self, raw: dict) -> list[SearchResult]:
        """
        Convert raw ChromaDB query output to SearchResult objects.

        ChromaDB returns distances (lower = more similar for cosine space),
        but we convert to similarity scores (higher = more similar) by
        computing: similarity = 1 - distance.

        Note: ChromaDB's cosine "distance" is actually 1 - cosine_similarity,
        so this conversion is exact (not an approximation).
        """
        results = []

        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            # Convert distance → similarity score
            score = round(1.0 - distance, 4)

            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=text,
                    score=score,
                    metadata=metadata,
                )
            )

        return results


# ---------------------------------------------------------------------------
# Quick smoke test — run directly to verify the store is working
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    """
    Quick sanity check — not a replacement for unit tests.
    Run with: python -m deal_intelligence_rag.retrieval.vector_store
    """
    import logging
    logging.basicConfig(level=logging.WARNING)  # quiet for smoke test

    print("Initialising VectorStore...")
    store = VectorStore()
    s = store.stats()
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Total chunks: {s['total_chunks']:,}")
    print(f"  Tickers: {s['tickers']}")
    print(f"  Years: {store.get_available_years()}")

    if s["total_chunks"] == 0:
        print("\n  No chunks found — run the embedder first.")
        return

    queries = [
        ("Revenue breakdown by segment", None),
        ("Supply chain and geopolitical risks", {"ticker": "AAPL"}),
        ("Cloud revenue growth", {"ticker": "MSFT", "year": 2024}),
        ("Risk factors for the business", {"ticker": "AAPL"}, "Risk Factors"),
    ]

    for item in queries:
        query = item[0]
        filters = item[1] if len(item) > 1 else None
        section = item[2] if len(item) > 2 else None

        print(f"\nQuery: {query!r}")
        if filters:
            print(f"Filters: {filters}")
        if section:
            print(f"Section filter: {section!r}")

        results = store.search(
            query,
            n_results=3,
            filters=filters,
            section_filter=section,
        )

        for i, r in enumerate(results, 1):
            print(
                f"  {i}. [{r.score:.3f}] {r.ticker} {r.filed_date} "
                f"| {r.section_heading[:50]}"
            )
            print(f"     {r.text[:120].strip()}...")


if __name__ == "__main__":
    _smoke_test()