"""
decomposer.py
-------------
Decomposes complex analytical queries into simpler sub-queries.

For simple factual questions ("What was Apple's revenue in 2024?"),
a single retrieval pass is sufficient. But for complex analytical
questions ("Compare Apple and Microsoft's cloud strategy and margins
over the last 2 years"), we need to:
  1. Break the question into focused sub-queries
  2. Retrieve evidence for each sub-query independently
  3. Synthesise a final answer from all evidence

This decomposition step is what separates a production RAG from a demo.

Decomposition strategy:
  - Uses Cohere command-r to generate sub-queries via a structured prompt
  - Falls back to the original query if decomposition fails
  - Detects query type (factual vs comparative vs analytical)
  - Limits to 4 sub-queries max to control latency and API costs

Usage (Python):
    from deal_intelligence_rag.query.decomposer import QueryDecomposer

    decomposer = QueryDecomposer()
    result = decomposer.decompose(
        "Compare Apple and Microsoft revenue growth and profit margins in 2024"
    )
    print(result.query_type)    # QueryType.COMPARATIVE
    print(result.sub_queries)   # ["Apple revenue growth 2024", ...]
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import cohere
import structlog
from dotenv import load_dotenv

from deal_intelligence_rag.query.output_schema import QueryType

load_dotenv()

log = structlog.get_logger(__name__)

LLM_MODEL = os.getenv("LLM_MODEL", "command-r")
MAX_SUB_QUERIES = 4


@dataclass
class DecompositionResult:
    """Result of query decomposition."""
    original_query: str
    query_type: QueryType
    sub_queries: list[str]
    needs_multi_hop: bool
    detected_tickers: list[str] = field(default_factory=list)
    detected_years: list[int] = field(default_factory=list)


# Known tickers to detect in queries
KNOWN_TICKERS = {
    "AAPL": ["apple", "aapl"],
    "MSFT": ["microsoft", "msft"],
    "GOOGL": ["google", "alphabet", "googl"],
    "AMZN": ["amazon", "amzn"],
    "META": ["meta", "facebook"],
    "NVDA": ["nvidia", "nvda"],
}


class QueryDecomposer:
    """
    Decomposes complex queries into focused sub-queries using an LLM.

    Architecture note:
        We use a structured JSON prompt rather than free-form generation.
        The LLM is instructed to return a JSON object with specific fields.
        This makes parsing reliable and avoids post-processing heuristics.

        If the LLM call fails or returns malformed JSON, we fall back to
        returning the original query as a single sub-query — ensuring the
        pipeline never fails due to decomposition errors.
    """

    def __init__(self, cohere_api_key: str | None = None) -> None:
        api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY not found in environment.")
        self.co = cohere.Client(api_key)

    def decompose(self, query: str) -> DecompositionResult:
        """
        Decompose a query into sub-queries and detect metadata.

        For simple factual queries, returns the original query as the
        single sub-query (no LLM call needed).
        For complex queries, calls the LLM to generate sub-queries.
        """
        # Detect tickers and years directly (no LLM needed)
        detected_tickers = self._detect_tickers(query)
        detected_years = self._detect_years(query)

        # Detect query type heuristically first
        query_type = self._detect_query_type(query)

        # Simple factual queries don't need decomposition
        if query_type == QueryType.FACTUAL:
            log.info("query_is_factual_no_decomposition", query=query[:60])
            return DecompositionResult(
                original_query=query,
                query_type=query_type,
                sub_queries=[query],
                needs_multi_hop=False,
                detected_tickers=detected_tickers,
                detected_years=detected_years,
            )

        # Complex queries — use LLM to decompose
        sub_queries = self._llm_decompose(query, query_type)
        needs_multi_hop = len(sub_queries) > 1

        log.info(
            "query_decomposed",
            query_type=query_type.value,
            sub_queries=len(sub_queries),
            needs_multi_hop=needs_multi_hop,
        )

        return DecompositionResult(
            original_query=query,
            query_type=query_type,
            sub_queries=sub_queries,
            needs_multi_hop=needs_multi_hop,
            detected_tickers=detected_tickers,
            detected_years=detected_years,
        )

    # ------------------------------------------------------------------
    # Internal: query type detection
    # ------------------------------------------------------------------

    def _detect_query_type(self, query: str) -> QueryType:
        """
        Heuristically detect query type from keywords.
        Falls back to FACTUAL for unrecognised patterns.
        """
        q = query.lower()

        comparative_signals = [
            "compare", "vs", "versus", "difference between",
            "how does", "relative to", "better than", "worse than",
            "more than", "less than", "both", "each",
        ]
        analytical_signals = [
            "why", "explain", "analyse", "analyze", "what caused",
            "trend", "over time", "year over year", "historically",
            "impact of", "effect of", "strategy", "outlook",
        ]
        extraction_signals = [
            "clause", "section", "provision", "covenant",
            "exact wording", "verbatim", "quote", "states that",
            "according to", "as defined",
        ]

        if any(s in q for s in extraction_signals):
            return QueryType.EXTRACTION
        if any(s in q for s in comparative_signals):
            return QueryType.COMPARATIVE
        if any(s in q for s in analytical_signals):
            return QueryType.ANALYTICAL
        return QueryType.FACTUAL

    # ------------------------------------------------------------------
    # Internal: LLM decomposition
    # ------------------------------------------------------------------

    def _llm_decompose(self, query: str, query_type: QueryType) -> list[str]:
        """
        Use Cohere command-r to decompose a complex query.
        Returns list of sub-queries. Falls back to [original_query] on error.
        """
        prompt = f"""You are a financial analyst assistant. Break down the following complex question about SEC filings into {MAX_SUB_QUERIES} or fewer focused sub-questions that can each be answered by searching a single document section.

Question: {query}
Query type: {query_type.value}

Rules:
- Each sub-question should be self-contained and searchable
- Focus on specific facts, metrics, or statements
- Do not exceed {MAX_SUB_QUERIES} sub-questions
- If the question is already simple, return just 1 sub-question

Respond with ONLY a JSON object in this exact format:
{{"sub_queries": ["sub-question 1", "sub-question 2", ...]}}"""

        try:
            response = self.co.chat(
                model=LLM_MODEL,
                message=prompt,
                temperature=0.1,
                max_tokens=300,
            )
            text = response.text.strip()

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if not json_match:
                log.warning("decomposer_no_json", response=text[:100])
                return [query]

            data = json.loads(json_match.group())
            sub_queries = data.get("sub_queries", [query])

            # Validate and clean
            sub_queries = [q.strip() for q in sub_queries if q.strip()]
            sub_queries = sub_queries[:MAX_SUB_QUERIES]

            if not sub_queries:
                return [query]

            return sub_queries

        except Exception as e:
            log.warning("decomposer_llm_failed", error=str(e))
            return [query]

    # ------------------------------------------------------------------
    # Internal: entity detection
    # ------------------------------------------------------------------

    def _detect_tickers(self, query: str) -> list[str]:
        """Detect company tickers mentioned in the query."""
        q = query.lower()
        found = []
        for ticker, aliases in KNOWN_TICKERS.items():
            if any(alias in q for alias in aliases):
                found.append(ticker)
        return found

    def _detect_years(self, query: str) -> list[int]:
        """Detect years mentioned in the query."""
        years = re.findall(r'\b(20\d{2})\b', query)
        return sorted(set(int(y) for y in years))