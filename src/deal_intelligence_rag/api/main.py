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

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Agent singleton — initialised once at startup
# ---------------------------------------------------------------------------

_agent = None


def get_agent():
    """
    Return the global agent instance.
    
    Raises:
        RuntimeError: if agent not initialised (check startup logs and environment variables,
                     especially COHERE_API_KEY)
    """
    if _agent is None:
        error_msg = (
            "Agent not initialised. Check server startup logs for errors. "
            "Common causes: COHERE_API_KEY not set, vector store unavailable, or model download failed."
        )
        log.error("get_agent_failed", error=error_msg)
        raise RuntimeError(error_msg)
    return _agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Initialises the agent at startup, cleans up at shutdown.
    """
    global _agent

    log.info("startup_initialising_agent")
    use_reranker = os.getenv("USE_RERANKER", "true").lower() == "true"
    use_judge = os.getenv("USE_JUDGE", "true").lower() == "true"

    try:
        from deal_intelligence_rag.agent.agent_loop import AgentLoop
        _agent = AgentLoop(use_reranker=use_reranker, use_judge=use_judge)
        log.info("startup_complete", use_reranker=use_reranker, use_judge=use_judge)
    except Exception as e:
        log.error(
            "startup_failed",
            error=str(e),
            use_reranker=use_reranker,
            use_judge=use_judge,
        )
        _agent = None
        # Re-raise to prevent app from starting in broken state
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