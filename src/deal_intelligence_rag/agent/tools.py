"""
tools.py
--------
Agent tools wrapping the multi-hop chain for specific M&A use cases.

Each tool is a focused, named function that the agent can call with
specific parameters. Tools are simpler than the raw chain — they have
opinionated defaults and domain-specific prompting.

Tools available:
  - search_filings:              General semantic search across filings
  - compare_metric:              Compare a specific metric across companies/years
  - extract_clause:              Extract specific legal/financial clauses
  - summarise_risk_factors:      Summarise risk factors for a company
"""

from __future__ import annotations

import structlog

from deal_intelligence_rag.query.multi_hop_chain import MultiHopChain
from deal_intelligence_rag.query.output_schema import Answer

log = structlog.get_logger(__name__)


class DealIntelligenceTools:
    """
    Domain-specific tools wrapping the MultiHopChain.
    Instantiated once and reused by the agent loop.
    """

    def __init__(self, chain: MultiHopChain) -> None:
        self.chain = chain

    def search_filings(
        self,
        query: str,
        ticker: str | None = None,
        year: int | None = None,
        form_type: str = "10-K",
    ) -> Answer:
        """
        General semantic search across indexed SEC filings.

        Args:
            query:     Natural language search query.
            ticker:    Optional ticker filter (e.g. "AAPL").
            year:      Optional filing year filter (e.g. 2024).
            form_type: Form type filter (default "10-K").
        """
        filters: dict = {}
        if ticker:
            filters["ticker"] = ticker.upper()
        if year:
            filters["year"] = int(year)
        if form_type:
            filters["form_type"] = form_type

        log.info("tool_search_filings", query=query[:60], filters=filters)
        return self.chain.query(query, filters=filters or None)

    def compare_metric(
        self,
        metric: str,
        tickers: list[str],
        year: int | None = None,
    ) -> Answer:
        """
        Compare a specific financial metric across multiple companies.

        Args:
            metric:  The metric to compare (e.g. "gross margin", "R&D spend").
            tickers: List of tickers to compare (e.g. ["AAPL", "MSFT"]).
            year:    Optional year filter.
        """
        ticker_names = " and ".join(tickers)
        year_str = f"in {year}" if year else "most recently"

        question = (
            f"Compare {metric} for {ticker_names} {year_str}. "
            f"Include specific figures and percentage changes where available."
        )

        filters: dict = {"ticker": [t.upper() for t in tickers]}
        if year:
            filters["year"] = year

        log.info("tool_compare_metric", metric=metric, tickers=tickers)
        return self.chain.query(question, filters=filters)

    def extract_clause(
        self,
        clause_type: str,
        ticker: str,
        year: int | None = None,
    ) -> Answer:
        """
        Extract a specific legal or financial clause from a filing.

        Args:
            clause_type: Type of clause (e.g. "material adverse change",
                         "debt covenant", "indemnification").
            ticker:      Company ticker.
            year:        Optional year filter.
        """
        question = (
            f"What does {ticker}'s filing say about {clause_type}? "
            f"Quote the relevant language directly."
        )

        filters: dict = {"ticker": ticker.upper()}
        if year:
            filters["year"] = year

        log.info("tool_extract_clause", clause_type=clause_type, ticker=ticker)
        return self.chain.query(question, filters=filters, n_evidence=8)

    def summarise_risk_factors(
        self,
        ticker: str,
        year: int | None = None,
        focus: str | None = None,
    ) -> Answer:
        """
        Summarise risk factors for a company, optionally focused on a theme.

        Args:
            ticker: Company ticker.
            year:   Optional year filter.
            focus:  Optional risk theme (e.g. "regulatory", "supply chain").
        """
        focus_str = f" related to {focus}" if focus else ""
        question = (
            f"What are the key risk factors{focus_str} disclosed by {ticker}? "
            f"Summarise the most significant risks with specific details."
        )

        filters: dict = {
            "ticker": ticker.upper(),
        }
        if year:
            filters["year"] = year

        log.info("tool_summarise_risks", ticker=ticker, focus=focus)
        return self.chain.query(
            question,
            filters=filters,
            n_evidence=6,
        )