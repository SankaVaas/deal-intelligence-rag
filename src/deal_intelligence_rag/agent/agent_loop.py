"""
agent_loop.py
-------------
Main agent orchestration — routes queries to the right tool and applies
the LLM-as-judge post-generation check.

For the current scope, this is a single-turn agent (no memory between
calls). The agent:
  1. Receives a query + optional parameters
  2. Selects the appropriate tool
  3. Calls the multi-hop chain via the tool
  4. Runs the judge on the answer
  5. Returns the final Answer with judge results attached

Usage (Python):
    from deal_intelligence_rag.agent.agent_loop import AgentLoop

    agent = AgentLoop()
    answer = agent.run("What are Apple's main revenue segments?")
    print(answer.answer)
    print(answer.confidence)
"""

from __future__ import annotations

import os
import time

import structlog
from dotenv import load_dotenv

from deal_intelligence_rag.agent.judge import AnswerJudge
from deal_intelligence_rag.agent.tools import DealIntelligenceTools
from deal_intelligence_rag.query.multi_hop_chain import MultiHopChain
from deal_intelligence_rag.query.output_schema import Answer

load_dotenv()
log = structlog.get_logger(__name__)


class AgentLoop:
    """
    Top-level agent that orchestrates tools and judge.

    This is the entry point called by the API layer (routes.py).
    It initialises all components once and reuses them across requests.
    """

    def __init__(
        self,
        use_reranker: bool = True,
        use_judge: bool = True,
    ) -> None:
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY not found.")

        self.use_judge = use_judge

        # Initialise chain and tools
        self.chain = MultiHopChain(
            cohere_api_key=cohere_api_key,
            use_reranker=use_reranker,
        )
        self.tools = DealIntelligenceTools(chain=self.chain)
        self.judge = AnswerJudge(cohere_api_key=cohere_api_key) if use_judge else None

        log.info(
            "agent_loop_ready",
            use_reranker=use_reranker,
            use_judge=use_judge,
        )

    def run(
        self,
        query: str,
        ticker: str | None = None,
        year: int | None = None,
        tool: str = "search_filings",
    ) -> Answer:
        """
        Run the agent on a query and return a structured Answer.

        Args:
            query:  Natural language question.
            ticker: Optional ticker filter.
            year:   Optional year filter.
            tool:   Which tool to use:
                      "search_filings"       — general search (default)
                      "compare_metric"       — compare across companies
                      "extract_clause"       — extract specific clause
                      "summarise_risks"      — risk factor summary
        """
        t_start = time.time()
        log.info("agent_run", query=query[:80], tool=tool, ticker=ticker, year=year)

        # Route to appropriate tool
        if tool == "compare_metric" and ticker:
            tickers = [t.strip() for t in ticker.split(",")]
            answer = self.tools.compare_metric(
                metric=query,
                tickers=tickers,
                year=year,
            )
        elif tool == "extract_clause" and ticker:
            answer = self.tools.extract_clause(
                clause_type=query,
                ticker=ticker,
                year=year,
            )
        elif tool == "summarise_risks" and ticker:
            answer = self.tools.summarise_risk_factors(
                ticker=ticker,
                year=year,
                focus=query if query else None,
            )
        else:
            # Default: general search
            answer = self.tools.search_filings(
                query=query,
                ticker=ticker,
                year=year,
            )

        # Run judge on non-refused answers
        if not answer.refused and self.judge and answer.evidence:
            passed, feedback = self.judge.evaluate(
                question=answer.question,
                answer=answer.answer,
                evidence_chunks=answer.evidence,
            )
            answer.judge_passed = passed
            answer.judge_feedback = feedback

            if not passed:
                log.warning(
                    "judge_flagged_answer",
                    feedback=feedback[:100],
                )

        elapsed = round(time.time() - t_start, 2)
        answer.metadata["total_elapsed_seconds"] = elapsed

        log.info(
            "agent_run_complete",
            refused=answer.refused,
            confidence=answer.confidence.value,
            judge_passed=answer.judge_passed,
            elapsed=elapsed,
        )

        return answer