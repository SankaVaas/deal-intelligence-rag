"""
main.py
-------
FastAPI application entry point for the Deal Intelligence RAG API.

The agent is initialised once at startup and reused across all requests
(singleton pattern). This avoids reloading the 90MB reranker model on
every request — critical for latency.

Run with:
    uvicorn src.deal_intelligence_rag.api.main:app --reload --port 8000

Or via hatch:
    hatch run serve
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from deal_intelligence_rag.api.middleware import LoggingMiddleware
from deal_intelligence_rag.api.routes import router
from deal_intelligence_rag.api.state import set_agent

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise agent at startup, clean up at shutdown."""
    log.info("startup_initialising_agent")

    use_reranker = os.getenv("USE_RERANKER", "true").lower() == "true"
    use_judge = os.getenv("USE_JUDGE", "true").lower() == "true"

    try:
        from deal_intelligence_rag.agent.agent_loop import AgentLoop
        agent = AgentLoop(use_reranker=use_reranker, use_judge=use_judge)
        set_agent(agent)
        log.info("startup_complete", use_reranker=use_reranker, use_judge=use_judge)
    except Exception as e:
        log.error("startup_failed", error=str(e))
        raise

    yield

    # Shutdown
    log.info("shutdown")
    _agent = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Deal Intelligence RAG",
    description=(
        "LLM-powered M&A deal intelligence platform. "
        "Query SEC 10-K/10-Q/8-K filings with multi-hop RAG, "
        "hybrid retrieval, and hallucination detection."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
app.add_middleware(LoggingMiddleware)

# Routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "service": "Deal Intelligence RAG",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "query": "/api/v1/query",
    }