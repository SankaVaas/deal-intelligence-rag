"""
state.py
--------
Global application state — agent singleton shared across all modules.

This module is the single source of truth for the agent instance.
Both main.py (which sets it) and routes.py (which reads it) import
from here — guaranteeing they share the same object reference.

Why a separate module?
    Python module imports are cached. When main.py and routes.py both
    do `from deal_intelligence_rag.api.state import get_agent`, they
    get the exact same module object from sys.modules — so mutations
    to _agent in main.py are visible to routes.py.

    If routes.py imported directly `from main import get_agent`, circular
    import issues and uvicorn's reload process isolation could cause them
    to see different module instances.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Agent singleton
# ---------------------------------------------------------------------------

_agent = None


def set_agent(agent) -> None:
    """Called once during FastAPI lifespan startup."""
    global _agent
    _agent = agent
    log.info("agent_state_set", agent_is_none=(_agent is None))


def get_agent():
    """
    Return the global agent instance.
    Raises RuntimeError if not yet initialised.
    """
    if _agent is None:
        raise RuntimeError(
            "Agent not initialised. Check server startup logs. "
            "Common causes: COHERE_API_KEY not set, or startup failed."
        )
    return _agent


def is_agent_ready() -> bool:
    """Non-raising check — used by health endpoint."""
    return _agent is not None