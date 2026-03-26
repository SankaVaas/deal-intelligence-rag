"""
embedder.py
-----------
Embeds chunked SEC filings using Cohere's embed-english-v3.0 model
and stores vectors + metadata in a local ChromaDB collection.

Pipeline position:
    chunker.py  →  embedder.py  →  vector_store.py (query time)

Why Cohere embed-english-v3.0?
    - Free tier: 1,000 calls/min, no credit card required
    - 1024-dimensional vectors — richer than many paid alternatives
    - input_type parameter lets us distinguish "search_document" (indexing)
      from "search_query" (query time) — this is important: the same model
      produces different vector spaces depending on input_type, and using
      the correct type improves retrieval recall by ~10-15%

ChromaDB persistence:
    - Vectors are stored locally at data/chroma/ (configurable via .env)
    - No server needed — ChromaDB runs embedded in-process
    - Collection name: "sec_filings"
    - On re-run, existing chunks are upserted (not duplicated) because
      chunk_ids are deterministic (set by chunker.py)

Batching strategy:
    Cohere's API accepts up to 96 texts per request. We batch at 90
    (slightly under the limit) to give headroom for edge cases.
    Each batch is retried up to 3 times with exponential backoff
    on transient errors (429 rate limit, 503 service unavailable).

Usage (CLI):
    python -m deal_intelligence_rag.ingestion.embedder --ticker AAPL
    python -m deal_intelligence_rag.ingestion.embedder --ticker AAPL --ticker MSFT

Usage (Python):
    from deal_intelligence_rag.ingestion.embedder import Embedder
    embedder = Embedder()
    embedder.embed_ticker("AAPL")
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import chromadb
import cohere
import structlog
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

load_dotenv()

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNKS_DIR = Path("data/chunks")
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma"))

COHERE_MODEL = "embed-english-v3.0"
COHERE_EMBED_DIMS = 1024
COHERE_BATCH_SIZE = 45          # ~45 chunks × ~700 tokens = ~31,500 tokens/call
                                # well under the 100k tokens/min trial limit
COLLECTION_NAME = "sec_filings"

# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class Embedder:
    """
    Reads .chunks.jsonl files, embeds with Cohere, stores in ChromaDB.

    Architecture note:
        The Embedder is intentionally separate from the Chunker.
        Embedding is an I/O-bound, API-dependent operation — keeping it
        separate means you can:
          - Re-embed with a different model without re-chunking
          - Run chunking offline and embedding online
          - Unit-test chunking without needing an API key

        ChromaDB is initialised once at construction time and reused
        across all embed_ticker() calls in the same process.
    """

    def __init__(
        self,
        cohere_api_key: str | None = None,
        chroma_persist_dir: Path = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        batch_size: int = COHERE_BATCH_SIZE,
    ) -> None:
        api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "COHERE_API_KEY not found. "
                "Set it in your .env file or pass cohere_api_key= explicitly."
            )

        self.co = cohere.Client(api_key)
        self.batch_size = batch_size

        # ChromaDB persistent client — data survives between runs
        chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.chroma = chromadb.PersistentClient(path=str(chroma_persist_dir))

        # Get or create the collection
        # embedding_function=None because we provide our own vectors directly
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",      # cosine similarity for text
                "description": "SEC 10-K/10-Q/8-K filings — deal intelligence RAG",
            },
        )

        log.info(
            "embedder_ready",
            collection=collection_name,
            chroma_dir=str(chroma_persist_dir),
            existing_docs=self.collection.count(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_ticker(
        self,
        ticker: str,
        form_type: str = "10-K",
        chunks_dir: Path = CHUNKS_DIR,
    ) -> int:
        """
        Embed all chunks for a ticker and upsert into ChromaDB.
        Returns the total number of chunks embedded.
        """
        ticker_dir = chunks_dir / ticker / form_type
        if not ticker_dir.exists():
            log.warning("no_chunks_dir", ticker=ticker, path=str(ticker_dir))
            return 0

        chunk_files = sorted(ticker_dir.glob("*.chunks.jsonl"))
        if not chunk_files:
            log.warning("no_chunk_files", ticker=ticker)
            return 0

        total_embedded = 0
        for chunk_file in chunk_files:
            count = self._embed_file(chunk_file, ticker)
            total_embedded += count

        log.info("ticker_embed_complete", ticker=ticker, total=total_embedded)
        return total_embedded

    def embed_multiple_tickers(
        self,
        tickers: list[str],
        form_type: str = "10-K",
        chunks_dir: Path = CHUNKS_DIR,
    ) -> dict[str, int]:
        """Embed chunks for multiple tickers. Returns {ticker: count}."""
        results: dict[str, int] = {}
        for ticker in tickers:
            count = self.embed_ticker(ticker, form_type=form_type, chunks_dir=chunks_dir)
            results[ticker] = count
        return results

    def collection_stats(self) -> dict:
        """Return stats about the current ChromaDB collection."""
        count = self.collection.count()
        return {
            "collection": self.collection.name,
            "total_chunks": count,
            "chroma_dir": str(CHROMA_PERSIST_DIR),
        }

    # ------------------------------------------------------------------
    # Internal: file-level processing
    # ------------------------------------------------------------------

    def _embed_file(self, chunk_file: Path, ticker: str) -> int:
        """
        Load a .chunks.jsonl file, embed all chunks, upsert to ChromaDB.
        Skips chunks that are already in the collection (by chunk_id).
        Returns the number of newly embedded chunks.
        """
        chunks = self._load_chunks(chunk_file)
        if not chunks:
            return 0

        # Check which chunk_ids are already in ChromaDB
        existing_ids = self._get_existing_ids([c["chunk_id"] for c in chunks])
        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

        if not new_chunks:
            log.info(
                "all_chunks_already_indexed",
                file=chunk_file.name,
                count=len(chunks),
            )
            return 0

        log.info(
            "embedding_file",
            file=chunk_file.name,
            total=len(chunks),
            new=len(new_chunks),
            ticker=ticker,
        )

        # Embed in batches
        embedded = 0
        batches = self._make_batches(new_chunks)

        for batch in tqdm(
            batches,
            desc=f"Embedding {chunk_file.stem[:40]}",
            unit="batch",
        ):
            self._embed_and_upsert_batch(batch)
            embedded += len(batch)
            # Small pause between batches — keeps token/min well under
            # the 100k trial limit even for large MSFT-style filings
            if len(batches) > 1:
                time.sleep(2)

        log.info("file_embed_complete", file=chunk_file.name, embedded=embedded)
        return embedded

    # ------------------------------------------------------------------
    # Internal: batching and embedding
    # ------------------------------------------------------------------

    def _make_batches(self, chunks: list[dict]) -> list[list[dict]]:
        """Split chunk list into batches of self.batch_size."""
        return [
            chunks[i: i + self.batch_size]
            for i in range(0, len(chunks), self.batch_size)
        ]

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
    )
    def _embed_and_upsert_batch(self, batch: list[dict]) -> None:
        """
        Embed a batch of chunks with Cohere and upsert to ChromaDB.

        input_type="search_document" is critical — Cohere's v3 models use
        different representation spaces for documents vs queries. Using
        "search_document" here and "search_query" at retrieval time
        ensures vectors are comparable and improves recall.

        Rate limit handling: Cohere trial = 100k tokens/min.
        At ~700 tokens/chunk × 45 chunks = ~31,500 tokens/call we stay
        well under. If a 429 still occurs we sleep 65s and retry once.
        """
        try:
            response = self.co.embed(
                texts=[c["text"] for c in batch],
                model=COHERE_MODEL,
                input_type="search_document",
                embedding_types=["float"],
            )
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower() or "TooManyRequests" in type(e).__name__:
                log.warning("rate_limit_hit_sleeping", seconds=65)
                time.sleep(65)
                # retry the embed call once after sleeping
                response = self.co.embed(
                    texts=[c["text"] for c in batch],
                    model=COHERE_MODEL,
                    input_type="search_document",
                    embedding_types=["float"],
                )
            else:
                raise

        embeddings = response.embeddings.float

        self.collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )

    # ------------------------------------------------------------------
    # Internal: utilities
    # ------------------------------------------------------------------

    def _load_chunks(self, path: Path) -> list[dict]:
        """Load all chunks from a .chunks.jsonl file."""
        chunks = []
        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunks.append(json.loads(line))
        except Exception as e:
            log.error("failed_to_load_chunks", path=str(path), error=str(e))
        return chunks

    def _get_existing_ids(self, chunk_ids: list[str]) -> set[str]:
        """
        Check which chunk_ids already exist in ChromaDB.
        Uses get() with include=[] to fetch only IDs (no vectors/docs).
        """
        try:
            result = self.collection.get(ids=chunk_ids, include=[])
            return set(result["ids"])
        except Exception:
            return set()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed chunked SEC filings with Cohere and store in ChromaDB."
    )
    parser.add_argument(
        "--ticker",
        action="append",
        dest="tickers",
        required=True,
        metavar="TICKER",
    )
    parser.add_argument(
        "--form",
        default="10-K",
        choices=["10-K", "10-Q", "8-K"],
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=CHUNKS_DIR,
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=CHROMA_PERSIST_DIR,
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    embedder = Embedder(chroma_persist_dir=args.chroma_dir)

    results = embedder.embed_multiple_tickers(
        tickers=args.tickers,
        form_type=args.form,
        chunks_dir=args.chunks_dir,
    )

    stats = embedder.collection_stats()

    print("\n--- Embedding Summary ---")
    for ticker, count in results.items():
        print(f"  {ticker}: {count} chunks embedded")
    print(f"\n  ChromaDB collection : {stats['collection']}")
    print(f"  Total chunks in DB  : {stats['total_chunks']:,}")
    print(f"  Stored at           : {stats['chroma_dir']}")


if __name__ == "__main__":
    main()