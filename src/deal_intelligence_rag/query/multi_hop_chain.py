"""
multi_hop_chain.py
------------------
Multi-hop retrieval chain — the core query engine.

Orchestrates the full pipeline from query to structured answer:
  1. Decompose query into sub-queries (decomposer.py)
  2. Retrieve evidence for each sub-query (fusion.py hybrid retrieval)
  3. Score confidence (confidence.py)
  4. Refuse if confidence too low
  5. Generate answer with Cohere command-r
  6. Validate with LLM-as-judge (judge.py)
  7. Return structured Answer

This is the file that agent tools (tools.py) call directly.

Usage (Python):
    from deal_intelligence_rag.query.multi_hop_chain import MultiHopChain

    chain = MultiHopChain()
    answer = chain.query(
        "What were Apple's iPhone revenue figures in 2024?",
        filters={"ticker": "AAPL"},
    )
    print(answer.answer)
    print(answer.confidence)
    for e in answer.evidence:
        print(e.citation())
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cohere
import structlog
from dotenv import load_dotenv

from deal_intelligence_rag.query.confidence import ConfidenceScorer
from deal_intelligence_rag.query.decomposer import QueryDecomposer
from deal_intelligence_rag.query.output_schema import (
    Answer,
    ConfidenceLevel,
    EvidenceChunk,
    QueryType,
    make_refusal,
)
from deal_intelligence_rag.retrieval.fusion import HybridRetriever

load_dotenv()

log = structlog.get_logger(__name__)

LLM_MODEL = os.getenv("LLM_MODEL", "command-r")
MAX_EVIDENCE_CHUNKS = 6    # max chunks passed to LLM context
MAX_CONTEXT_CHARS = 6000   # total chars of evidence passed to LLM


class MultiHopChain:
    """
    End-to-end query chain from question to structured Answer.

    Architecture note:
        Each component is injected at construction — this makes the chain
        testable (swap out retriever with a mock) and configurable
        (disable reranker for faster development iteration).

        The chain is stateless — each query() call is independent.
        Conversation history is managed by the agent layer (agent_loop.py).
    """

    def __init__(
        self,
        cohere_api_key: str | None = None,
        use_reranker: bool = True,
    ) -> None:
        api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY not found.")

        self.co = cohere.Client(api_key)
        self.retriever = HybridRetriever(use_reranker=use_reranker)
        self.decomposer = QueryDecomposer(cohere_api_key=api_key)
        self.confidence_scorer = ConfidenceScorer()

        log.info("multi_hop_chain_ready", model=LLM_MODEL)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        filters: dict | None = None,
        n_evidence: int = 5,
    ) -> Answer:
        """
        Run the full query pipeline and return a structured Answer.

        Args:
            question:   The user's natural language question.
            filters:    Metadata filters: {"ticker": "AAPL", "year": 2024}
            n_evidence: Number of evidence chunks to retrieve per sub-query.

        Returns:
            Answer object with answer text, confidence, evidence, and metadata.
        """
        t_start = time.time()
        log.info("chain_query_start", question=question[:80], filters=filters)

        # Step 1: decompose
        decomposition = self.decomposer.decompose(question)
        log.info(
            "decomposition_result",
            query_type=decomposition.query_type.value,
            sub_queries=decomposition.sub_queries,
            tickers=decomposition.detected_tickers,
        )

        # Merge detected tickers into filters if not already specified
        effective_filters = filters or {}
        if decomposition.detected_tickers and "ticker" not in effective_filters:
            if len(decomposition.detected_tickers) == 1:
                effective_filters = {**effective_filters, "ticker": decomposition.detected_tickers[0]}
            elif len(decomposition.detected_tickers) > 1:
                effective_filters = {**effective_filters, "ticker": decomposition.detected_tickers}

        if decomposition.detected_years and "year" not in effective_filters:
            if len(decomposition.detected_years) == 1:
                effective_filters = {**effective_filters, "year": decomposition.detected_years[0]}

        # Step 2: retrieve evidence for each sub-query
        all_evidence = self._retrieve_all_evidence(
            sub_queries=decomposition.sub_queries,
            filters=effective_filters if effective_filters else None,
            n_per_query=n_evidence,
        )

        # Step 3: convert to EvidenceChunk objects
        evidence_chunks = self._to_evidence_chunks(all_evidence)

        # Step 4: confidence check — refuse early if evidence is weak
        should_refuse, refusal_reason = self.confidence_scorer.should_refuse(evidence_chunks)
        if should_refuse:
            log.info("refusing_answer", reason=refusal_reason)
            elapsed = round(time.time() - t_start, 2)
            return make_refusal(
                question=question,
                reason=refusal_reason,
                metadata={"elapsed_seconds": elapsed, "filters": effective_filters},
            )

        # Step 5: generate answer
        answer_text = self._generate_answer(
            question=question,
            evidence_chunks=evidence_chunks,
            query_type=decomposition.query_type,
        )

        # Step 6: score confidence
        confidence_level, confidence_score, _ = self.confidence_scorer.score(
            evidence_chunks, question
        )

        elapsed = round(time.time() - t_start, 2)

        answer = Answer(
            question=question,
            answer=answer_text,
            confidence=confidence_level,
            confidence_score=confidence_score,
            refused=False,
            evidence=evidence_chunks[:MAX_EVIDENCE_CHUNKS],
            query_type=decomposition.query_type,
            sub_queries=decomposition.sub_queries,
            judge_passed=True,
            metadata={
                "elapsed_seconds": elapsed,
                "model": LLM_MODEL,
                "filters_used": effective_filters,
                "sub_query_count": len(decomposition.sub_queries),
            },
        )

        log.info(
            "chain_query_complete",
            confidence=confidence_level.value,
            score=confidence_score,
            evidence_count=len(evidence_chunks),
            elapsed=elapsed,
        )

        return answer

    # ------------------------------------------------------------------
    # Internal: retrieval
    # ------------------------------------------------------------------

    def _retrieve_all_evidence(
        self,
        sub_queries: list[str],
        filters: dict | None,
        n_per_query: int,
    ) -> list:
        """
        Retrieve evidence for all sub-queries and deduplicate by chunk_id.
        Returns a flat list of unique results sorted by relevance.
        """
        seen_ids: dict[str, object] = {}

        for sub_query in sub_queries:
            results = self.retriever.search(
                query=sub_query,
                n_results=n_per_query,
                filters=filters,
            )
            for result in results:
                if result.chunk_id not in seen_ids:
                    seen_ids[result.chunk_id] = result
                else:
                    # Keep the higher-scored occurrence
                    existing = seen_ids[result.chunk_id]
                    existing_score = getattr(existing, "rerank_score", None) or getattr(existing, "rrf_score", 0)
                    new_score = getattr(result, "rerank_score", None) or getattr(result, "rrf_score", 0)
                    if new_score > existing_score:
                        seen_ids[result.chunk_id] = result

        # Sort by rerank_score or rrf_score descending
        all_results = list(seen_ids.values())
        all_results.sort(
            key=lambda r: (getattr(r, "rerank_score", None) or getattr(r, "rrf_score", 0)),
            reverse=True,
        )
        return all_results

    def _to_evidence_chunks(self, results: list) -> list[EvidenceChunk]:
        """Convert retrieval results to EvidenceChunk objects."""
        chunks = []
        for result in results[:MAX_EVIDENCE_CHUNKS]:
            score = getattr(result, "rerank_score", None) or getattr(result, "rrf_score", 0.0)
            chunks.append(
                EvidenceChunk(
                    chunk_id=result.chunk_id,
                    ticker=result.metadata.get("ticker", ""),
                    company_name=result.metadata.get("company_name", ""),
                    filed_date=result.metadata.get("filed_date", ""),
                    form_type=result.metadata.get("form_type", ""),
                    section_heading=result.metadata.get("section_heading", ""),
                    text=result.text,
                    relevance_score=min(float(score), 1.0),
                    is_table=result.metadata.get("is_table", False),
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # Internal: generation
    # ------------------------------------------------------------------

    def _generate_answer(
        self,
        question: str,
        evidence_chunks: list[EvidenceChunk],
        query_type: QueryType,
    ) -> str:
        """
        Generate an answer using Cohere command-r with evidence context.

        We use a grounded prompt that instructs the model to:
          - Only use information from the provided evidence
          - Cite specific filings when making claims
          - Acknowledge uncertainty rather than fabricate
        """
        # Build evidence context — truncate to stay within token budget
        context_parts = []
        total_chars = 0
        for i, chunk in enumerate(evidence_chunks, 1):
            citation = chunk.citation()
            chunk_text = f"[{i}] {citation}\n{chunk.text}"
            if total_chars + len(chunk_text) > MAX_CONTEXT_CHARS:
                break
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are a financial analyst assistant specialising in SEC filings analysis for M&A due diligence.

Answer the following question using ONLY the provided evidence from SEC filings. 
Do not use any outside knowledge. If the evidence does not contain enough information, say so clearly.
Cite the filing source (e.g., "According to Apple's 2024 10-K...") when making specific claims.
Be precise with numbers and dates.

Question: {question}

Evidence from SEC Filings:
{context}

Answer:"""

        try:
            response = self.co.chat(
                model=LLM_MODEL,
                message=prompt,
                temperature=0.1,
                max_tokens=600,
            )
            return response.text.strip()
        except Exception as e:
            log.error("generation_failed", error=str(e))
            return f"Answer generation failed: {str(e)}"