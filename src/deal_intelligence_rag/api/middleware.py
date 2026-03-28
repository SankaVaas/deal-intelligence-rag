"""
middleware.py
-------------
FastAPI middleware for request logging and timing.
"""

from __future__ import annotations

import time
import uuid

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

log = structlog.get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request with method, path, status code, and latency.
    Assigns a unique request_id to each request for tracing.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        t_start = time.time()

        # Attach request_id to request state for use in route handlers
        request.state.request_id = request_id

        log.info(
            "request_start",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        response = await call_next(request)

        elapsed_ms = round((time.time() - t_start) * 1000)
        log.info(
            "request_complete",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            elapsed_ms=elapsed_ms,
        )

        # Add timing header to response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(elapsed_ms)

        return response