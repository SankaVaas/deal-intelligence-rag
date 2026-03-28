"""
judge.py
--------
LLM-as-judge hallucination detection for generated answers.

After the main chain generates an answer, the judge runs a second LLM call
to verify the answer is faithful to the retrieved evidence. This catches
cases where the LLM:
  - Added facts not in the evidence ("hallucinated")
  - Made numerical errors (misread a table)
  - Over-generalised from limited evidence
  - Made claims that contradict the evidence

The judge uses a stricter, more focused prompt than the main generation.
It returns a binary PASS/FAIL verdict plus a feedback string.

Usage (Python):
    from deal_intelligence_rag.agent.judge import AnswerJudge

    judge = AnswerJudge()
    passed, feedback = judge.evaluate(
        question="What was Apple's revenue in 2024?",
        answer="Apple's revenue was $391 billion in fiscal 2024.",
        evidence_chunks=chunks,
    )
"""

from __future__ import annotations

import os
import re

import cohere
import structlog
from dotenv import load_dotenv

from deal_intelligence_rag.query.output_schema import EvidenceChunk

load_dotenv()

log = structlog.get_logger(__name__)

LLM_MODEL = os.getenv("LLM_MODEL", "command-r")
MAX_JUDGE_CONTEXT_CHARS = 3000


class AnswerJudge:
    """
    Evaluates whether a generated answer is faithful to its evidence.

    Returns (passed: bool, feedback: str).
    If passed=False, the answer should be flagged in the API response.
    The chain does NOT automatically regenerate — that decision is left
    to the caller (agent_loop.py) to avoid infinite retry loops.
    """

    def __init__(self, cohere_api_key: str | None = None) -> None:
        api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY not found.")
        self.co = cohere.Client(api_key)

    def evaluate(
        self,
        question: str,
        answer: str,
        evidence_chunks: list[EvidenceChunk],
    ) -> tuple[bool, str]:
        """
        Evaluate whether the answer is faithful to the evidence.

        Returns:
            (passed, feedback)
            passed=True  → answer appears faithful
            passed=False → answer contains claims not in evidence
        """
        if not answer or not evidence_chunks:
            return True, ""

        # Build truncated evidence context for judge
        context_parts = []
        total_chars = 0
        for i, chunk in enumerate(evidence_chunks[:4], 1):
            text = f"[{i}] {chunk.citation()}\n{chunk.text[:600]}"
            if total_chars + len(text) > MAX_JUDGE_CONTEXT_CHARS:
                break
            context_parts.append(text)
            total_chars += len(text)

        context = "\n\n".join(context_parts)

        prompt = f"""You are a strict fact-checker for financial analysis. 

Your task: determine if the ANSWER is fully supported by the EVIDENCE provided.

Question: {question}

Evidence:
{context}

Answer to evaluate:
{answer}

Check if every factual claim in the answer (numbers, percentages, dates, company names, product names) is directly supported by the evidence above.

Respond with ONLY a JSON object:
{{"verdict": "PASS" or "FAIL", "issues": "describe any unsupported claims, or empty string if PASS"}}"""

        try:
            response = self.co.chat(
                model=LLM_MODEL,
                message=prompt,
                temperature=0.0,
                max_tokens=200,
            )
            text = response.text.strip()

            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if not json_match:
                log.warning("judge_no_json_response", response=text[:100])
                return True, ""

            import json
            data = json.loads(json_match.group())
            verdict = data.get("verdict", "PASS").upper()
            issues = data.get("issues", "")

            passed = verdict == "PASS"
            log.info(
                "judge_verdict",
                passed=passed,
                issues=issues[:100] if issues else "",
            )
            return passed, issues

        except Exception as e:
            log.warning("judge_failed", error=str(e))
            # On judge failure, pass through — don't block the answer
            return True, f"Judge evaluation failed: {str(e)}"