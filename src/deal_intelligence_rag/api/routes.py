"""
routes.py
---------
FastAPI route handlers for the deal intelligence API.

Endpoints:
  GET  /health          → system health check
  GET  /stats           → collection statistics
  POST /query           → main query endpoint
  POST /ingest          → trigger ingestion for new tickers
"""

from __future__ import annotations

from typing import Optional

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    query: str = Field(..., min_length=3, max_length=1000)
    ticker: Optional[str] = Field(None, description="Filter by ticker e.g. AAPL")
    year: Optional[int] = Field(None, ge=2000, le=2030)
    tool: str = Field(
        default="search_filings",
        description="Tool to use: search_filings | compare_metric | extract_clause | summarise_risks",
    )
    use_reranker: bool = Field(default=True)

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "query": "What were Apple's iPhone revenue figures in fiscal 2024?",
                "ticker": "AAPL",
                "year": 2024,
                "tool": "search_filings",
            }
        ]
    }}


class IngestRequest(BaseModel):
    """Request body for the /ingest endpoint."""
    tickers: list[str] = Field(..., min_length=1)
    form_type: str = Field(default="10-K")
    limit: int = Field(default=3, ge=1, le=10)


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@router.get("/health")
async def health_check():
    """Health check endpoint — returns 200 if the service is running."""
    return {"status": "ok", "service": "deal-intelligence-rag"}


@router.get("/stats")
async def get_stats():
    """Return statistics about the indexed collection."""
    from deal_intelligence_rag.api.state import get_agent
    agent = get_agent()
    stats = agent.chain.retriever.get_stats()
    return {"status": "ok", "collection_stats": stats}


@router.post("/query")
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint. Runs the full RAG pipeline and returns a structured answer.
    """
    from deal_intelligence_rag.api.state import get_agent

    log.info(
        "api_query",
        query=request.query[:80],
        ticker=request.ticker,
        year=request.year,
        tool=request.tool,
    )

    try:
        agent = get_agent()
        answer = agent.run(
            query=request.query,
            ticker=request.ticker,
            year=request.year,
            tool=request.tool,
        )
        return answer.to_api_response()

    except Exception as e:
        log.error("query_endpoint_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest")
async def ingest_endpoint(request: IngestRequest):
    """
    Trigger ingestion pipeline for new tickers.
    Downloads, parses, chunks, and embeds filings.
    Note: this is a long-running operation — may take several minutes.
    """
    import asyncio

    from deal_intelligence_rag.ingestion.edgar_downloader import EdgarDownloader
    from deal_intelligence_rag.ingestion.pdf_parser import FilingParser
    from deal_intelligence_rag.ingestion.chunker import Chunker
    from deal_intelligence_rag.retrieval.embedder import Embedder

    log.info("ingest_start", tickers=request.tickers, form_type=request.form_type)

    try:
        # Download
        downloader = EdgarDownloader()
        results = await downloader.download_multiple_tickers(
            tickers=request.tickers,
            form_type=request.form_type,
            limit=request.limit,
        )

        # Parse
        parser = FilingParser()
        chunker = Chunker()
        embedder = Embedder()

        summary = {}
        for ticker in request.tickers:
            parsed = parser.parse_ticker(ticker, form_type=request.form_type)
            chunk_results = chunker.chunk_ticker(ticker, form_type=request.form_type)
            embedded = embedder.embed_ticker(ticker, form_type=request.form_type)
            summary[ticker] = {
                "filings_downloaded": sum(1 for r in results.get(ticker, []) if r.success),
                "filings_parsed": len(parsed),
                "chunks_created": sum(len(r.chunks) for r in chunk_results),
                "chunks_embedded": embedded,
            }

        log.info("ingest_complete", summary=summary)
        return {"status": "ok", "summary": summary}

    except Exception as e:
        log.error("ingest_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))