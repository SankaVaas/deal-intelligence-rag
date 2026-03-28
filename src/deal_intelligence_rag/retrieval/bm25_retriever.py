"""
bm25_retriever.py
-----------------
Sparse keyword retrieval using BM25 (Best Match 25) algorithm.

BM25 is the gold standard for sparse retrieval — it scores documents
based on term frequency (TF) and inverse document frequency (IDF),
with saturation parameters that prevent single terms from dominating.

Why BM25 alongside dense retrieval?
    Dense (vector) retrieval excels at semantic similarity — it finds
    chunks that *mean* the same thing as the query even if they use
    different words. But it fails on:
      - Exact terms: "Azure OpenAI Service", "iPhone 15 Pro", "ASC 606"
      - Numbers: "$94.3 billion", "Q3 2024", "18% growth"
      - Rare proper nouns: "Satya Nadella", "Tim Cook", specific product names
      - Legal clause references: "Section 4.2(b)", "Exhibit 21.1"

    BM25 catches all of these reliably because it's term-frequency based.
    Combining both signals via Reciprocal Rank Fusion (fusion.py) produces
    retrieval quality that neither system achieves alone.

Architecture:
    BM25 index is built from the .chunks.jsonl files on disk (same source
    as the vector store). The index is held in memory — at 750 chunks it's
    ~5MB, trivially small. For a much larger corpus (100k+ chunks) we would
    persist the index to disk using pickle or a dedicated BM25 server.

    Tokenisation: we use a simple whitespace + punctuation tokeniser rather
    than a full NLP pipeline. This is intentional — for financial text,
    preserving tokens like "10-K", "ASC-606", "$94.3B" is more important
    than linguistic normalisation. A spaCy pipeline would split these.

Usage (Python):
    from deal_intelligence_rag.retrieval.bm25_retriever import BM25Retriever

    retriever = BM25Retriever()
    retriever.build_index()   # or load from cache

    results = retriever.search("Azure cloud revenue growth", n_results=10)
    results = retriever.search(
        "supply chain risks",
        n_results=10,
        filters={"ticker": "AAPL"},
    )
"""

from __future__ import annotations

import json
import logging
import pickle
import re
import string
from dataclasses import dataclass
from pathlib import Path

import structlog
from rank_bm25 import BM25Okapi

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNKS_DIR = Path("data/chunks")
INDEX_CACHE_PATH = Path("data/bm25_index.pkl")

# BM25Okapi parameters (defaults are well-tuned for most corpora)
# k1: term frequency saturation (1.5 = moderate saturation)
# b:  length normalisation (0.75 = standard)
BM25_K1 = 1.5
BM25_B = 0.75

# Stopwords — common English words that carry no retrieval signal
# Kept minimal: financial text uses words like "may", "will", "year"
# as meaningful terms, so we don't strip them
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "this", "that", "these", "those", "it", "its", "we", "our", "us",
    "as", "not", "no", "so", "if", "than", "then", "such", "also",
}

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class BM25Result:
    """A single result from BM25 keyword search."""

    chunk_id: str
    text: str
    score: float          # raw BM25 score (not normalised — use for ranking only)
    metadata: dict

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
            f"BM25Result(ticker={self.ticker}, date={self.filed_date}, "
            f"section={self.section_heading[:40]!r}, score={self.score:.3f})"
        )


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------


class BM25Retriever:
    """
    Sparse keyword retrieval over the SEC filing chunk corpus.

    Two-phase usage:
        1. build_index()  — tokenise all chunks and build BM25 index
                            (takes ~2-3 seconds for 750 chunks)
        2. search()       — query the index, optionally filter by metadata

    The index can be cached to disk (save_index / load_index) so
    rebuild is only needed when new filings are added.
    """

    def __init__(
        self,
        chunks_dir: Path = CHUNKS_DIR,
        index_cache_path: Path = INDEX_CACHE_PATH,
    ) -> None:
        self.chunks_dir = chunks_dir
        self.index_cache_path = index_cache_path

        # Set after build_index() or load_index()
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict] = []          # parallel list to BM25 corpus
        self._tokenised_corpus: list[list[str]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_index(
        self,
        form_type: str = "10-K",
        force_rebuild: bool = False,
    ) -> int:
        """
        Load all chunks from disk and build the BM25 index.

        Tries to load from cache first unless force_rebuild=True.
        Returns the number of chunks indexed.
        """
        # Try cache first
        if not force_rebuild and self.index_cache_path.exists():
            loaded = self._load_index()
            if loaded:
                log.info(
                    "bm25_loaded_from_cache",
                    chunks=len(self._chunks),
                    path=str(self.index_cache_path),
                )
                return len(self._chunks)

        log.info("building_bm25_index", chunks_dir=str(self.chunks_dir))

        # Load all chunks from all .chunks.jsonl files
        self._chunks = self._load_all_chunks(form_type)

        if not self._chunks:
            log.warning("no_chunks_found", chunks_dir=str(self.chunks_dir))
            return 0

        # Tokenise corpus
        log.info("tokenising_corpus", chunk_count=len(self._chunks))
        self._tokenised_corpus = [
            self._tokenise(chunk["text"])
            for chunk in self._chunks
        ]

        # Build BM25 index
        self._bm25 = BM25Okapi(
            self._tokenised_corpus,
            k1=BM25_K1,
            b=BM25_B,
        )

        log.info("bm25_index_built", chunks=len(self._chunks))

        # Save to cache
        self._save_index()

        return len(self._chunks)

    def search(
        self,
        query: str,
        n_results: int = 10,
        filters: dict | None = None,
    ) -> list[BM25Result]:
        """
        Search the BM25 index for chunks matching the query.

        Args:
            query:     Natural language query or keyword string.
            n_results: Number of results to return.
            filters:   Metadata filters — same format as VectorStore.search():
                         {"ticker": "AAPL"}
                         {"ticker": "AAPL", "year": 2024}
                         {"ticker": ["AAPL", "MSFT"]}

        Returns:
            List of BM25Result sorted by descending BM25 score.
            Chunks with score 0 (no term overlap) are excluded.
        """
        if self._bm25 is None:
            log.warning("index_not_built_building_now")
            self.build_index()

        if self._bm25 is None:
            log.error("bm25_index_unavailable")
            return []

        query_tokens = self._tokenise(query)
        if not query_tokens:
            return []

        log.info(
            "bm25_search",
            query=query[:80],
            query_tokens=query_tokens[:10],
            filters=filters,
        )

        # Get BM25 scores for all chunks
        scores = self._bm25.get_scores(query_tokens)

        # Build results with metadata filtering
        results: list[BM25Result] = []
        for idx, (chunk, score) in enumerate(zip(self._chunks, scores)):
            if score <= 0:
                continue  # no term overlap — skip

            if filters and not self._matches_filters(chunk["metadata"], filters):
                continue

            results.append(
                BM25Result(
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"],
                    score=float(score),
                    metadata=chunk["metadata"],
                )
            )

        # Sort by score descending and return top n
        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:n_results]

        log.info("bm25_search_complete", returned=len(results))
        return results

    def get_indexed_tickers(self) -> list[str]:
        """Return list of tickers currently in the BM25 index."""
        if not self._chunks:
            return []
        return sorted({c["metadata"].get("ticker", "") for c in self._chunks})

    def index_size(self) -> int:
        """Return number of chunks in the index."""
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Internal: tokenisation
    # ------------------------------------------------------------------

    def _tokenise(self, text: str) -> list[str]:
        """
        Tokenise text for BM25 indexing and querying.

        Strategy:
          1. Lowercase
          2. Split on whitespace and punctuation (but keep hyphens in
             compound terms like "10-K", "non-GAAP", "ASC-606")
          3. Remove pure punctuation tokens
          4. Remove stopwords
          5. Remove very short tokens (single chars, except "$")

        Design decision: we preserve hyphenated terms and numeric tokens
        because in financial text these carry high retrieval signal:
          - "10-K" is a specific document type
          - "$94.3" is a specific figure
          - "non-GAAP" is a specific accounting concept
        Splitting these would lose the signal.
        """
        # Lowercase
        text = text.lower()

        # Split on whitespace and most punctuation, but keep hyphens
        # and periods within numbers (e.g. $94.3, 2024.0)
        tokens = re.split(r"[\s,;:!?()\[\]{}\"\'/\\|<>@#%^&*+=~`]+", text)

        clean_tokens = []
        for token in tokens:
            # Strip trailing punctuation (periods, colons at end of words)
            token = token.strip(".")

            # Skip empty tokens
            if not token:
                continue

            # Skip pure punctuation
            if all(c in string.punctuation for c in token):
                continue

            # Skip stopwords
            if token in STOPWORDS:
                continue

            # Skip single characters except "$" (dollar sign = financial signal)
            if len(token) == 1 and token != "$":
                continue

            clean_tokens.append(token)

        return clean_tokens

    # ------------------------------------------------------------------
    # Internal: filtering
    # ------------------------------------------------------------------

    def _matches_filters(self, metadata: dict, filters: dict) -> bool:
        """
        Check if a chunk's metadata matches all filter conditions.

        Supports:
          - Exact match: {"ticker": "AAPL"}
          - List (any of): {"ticker": ["AAPL", "MSFT"]}
          - Numeric: {"year": 2024}
        """
        for key, value in filters.items():
            chunk_val = metadata.get(key)
            if chunk_val is None:
                return False
            if isinstance(value, list):
                if chunk_val not in value:
                    return False
            else:
                # Normalise types for comparison (year stored as int)
                if key == "year":
                    if int(chunk_val) != int(value):
                        return False
                else:
                    if chunk_val != value:
                        return False
        return True

    # ------------------------------------------------------------------
    # Internal: loading chunks
    # ------------------------------------------------------------------

    def _load_all_chunks(self, form_type: str) -> list[dict]:
        """Load all chunks from all .chunks.jsonl files under chunks_dir."""
        all_chunks: list[dict] = []

        # Walk: data/chunks/{TICKER}/{FORM_TYPE}/*.chunks.jsonl
        pattern = f"*/{form_type}/*.chunks.jsonl"
        chunk_files = sorted(self.chunks_dir.glob(pattern))

        if not chunk_files:
            # Also try without form_type subdirectory
            chunk_files = sorted(self.chunks_dir.glob("*/*.chunks.jsonl"))

        for chunk_file in chunk_files:
            file_chunks = self._load_jsonl(chunk_file)
            all_chunks.extend(file_chunks)
            log.info(
                "loaded_chunk_file",
                file=chunk_file.name,
                count=len(file_chunks),
            )

        return all_chunks

    def _load_jsonl(self, path: Path) -> list[dict]:
        """Load a .chunks.jsonl file into a list of dicts."""
        chunks = []
        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunks.append(json.loads(line))
        except Exception as e:
            log.error("failed_to_load_jsonl", path=str(path), error=str(e))
        return chunks

    # ------------------------------------------------------------------
    # Internal: index persistence
    # ------------------------------------------------------------------

    def _save_index(self) -> None:
        """Pickle the BM25 index and chunk list to disk for fast reload."""
        try:
            self.index_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.index_cache_path.open("wb") as f:
                pickle.dump(
                    {
                        "bm25": self._bm25,
                        "chunks": self._chunks,
                        "tokenised_corpus": self._tokenised_corpus,
                    },
                    f,
                )
            log.info("bm25_index_saved", path=str(self.index_cache_path))
        except Exception as e:
            log.warning("bm25_save_failed", error=str(e))

    def _load_index(self) -> bool:
        """Load the BM25 index from the pickle cache. Returns True on success."""
        try:
            with self.index_cache_path.open("rb") as f:
                data = pickle.load(f)
            self._bm25 = data["bm25"]
            self._chunks = data["chunks"]
            self._tokenised_corpus = data.get("tokenised_corpus", [])
            return True
        except Exception as e:
            log.warning("bm25_load_failed", error=str(e))
            return False


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    """
    Quick sanity check.
    Run with: python -m deal_intelligence_rag.retrieval.bm25_retriever
    """
    logging.basicConfig(level=logging.WARNING)

    print("Building BM25 index...")
    retriever = BM25Retriever()
    count = retriever.build_index()
    print(f"  Indexed {count} chunks")
    print(f"  Tickers: {retriever.get_indexed_tickers()}")

    if count == 0:
        print("  No chunks found — run the chunker first.")
        return

    queries = [
        ("Azure cloud revenue growth", None),
        ("iPhone supply chain risks", {"ticker": "AAPL"}),
        ("capital expenditure investments", {"ticker": "MSFT", "year": 2024}),
        ("10-K annual report fiscal year", None),
        ("non-GAAP operating income", {"ticker": ["AAPL", "MSFT"]}),
    ]

    print()
    for query, filters in queries:
        print(f"Query: {query!r}")
        if filters:
            print(f"Filters: {filters}")

        results = retriever.search(query, n_results=3, filters=filters)

        if not results:
            print("  No results.")
        for i, r in enumerate(results, 1):
            print(
                f"  {i}. [{r.score:.2f}] {r.ticker} {r.filed_date} "
                f"| {r.section_heading[:50]}"
            )
            print(f"     {r.text[:120].strip()}...")
        print()


if __name__ == "__main__":
    _smoke_test()