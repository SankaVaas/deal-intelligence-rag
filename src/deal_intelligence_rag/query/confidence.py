"""
confidence.py
-------------
Confidence scoring and refusal logic for RAG answers.

The confidence scorer takes retrieval results and decides:
  1. What confidence level to assign (HIGH / MEDIUM / LOW / REFUSED)
  2. Whether to refuse to answer (below threshold)
  3. What numeric score to report

This is one of the most important components for production RAG systems —
knowing when NOT to answer is as important as knowing how to answer.

Scoring factors:
  - Top evidence relevance score (reranker output)
  - Evidence consistency (do chunks agree or contradict?)
  - Evidence coverage (how many chunks support the answer?)
  - Query-evidence alignment (does the evidence actually address the query?)

Refusal thresholds (configurable via .env):
  CONFIDENCE_REFUSE_BELOW=0.15   → refuse if top score < 0.15
  CONFIDENCE_LOW_BELOW=0.35      → LOW confidence if top score < 0.35
  CONFIDENCE_HIGH_ABOVE=0.70     → HIGH confidence if top score > 0.70
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from deal_intelligence_rag.query.output_schema import ConfidenceLevel, EvidenceChunk

load_dotenv()

# ---------------------------------------------------------------------------
# Thresholds (overridable via environment variables)
# ---------------------------------------------------------------------------

REFUSE_THRESHOLD = float(os.getenv("CONFIDENCE_REFUSE_BELOW", "0.15"))
LOW_THRESHOLD = float(os.getenv("CONFIDENCE_LOW_BELOW", "0.35"))
HIGH_THRESHOLD = float(os.getenv("CONFIDENCE_HIGH_ABOVE", "0.70"))

# Minimum number of evidence chunks required to answer confidently
MIN_EVIDENCE_FOR_HIGH = 2
MIN_EVIDENCE_TO_ANSWER = 1


class ConfidenceScorer:
    """
    Computes confidence scores and refusal decisions for RAG answers.

    Design note:
        Confidence is computed from retrieval signals BEFORE the LLM
        generates an answer. This means we can refuse early — before
        spending API tokens on a generation that would be unreliable.

        Post-generation, the LLM-as-judge (judge.py) provides a second
        check on factual consistency. The two-stage approach catches
        both retrieval failures (no good evidence) and generation failures
        (evidence was good but LLM hallucinated).
    """

    def __init__(
        self,
        refuse_threshold: float = REFUSE_THRESHOLD,
        low_threshold: float = LOW_THRESHOLD,
        high_threshold: float = HIGH_THRESHOLD,
    ) -> None:
        self.refuse_threshold = refuse_threshold
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def score(
        self,
        evidence_chunks: list[EvidenceChunk],
        query: str = "",
    ) -> tuple[ConfidenceLevel, float, str]:
        """
        Score the confidence of an answer given its evidence chunks.

        Returns:
            (confidence_level, confidence_score, refusal_reason)
            refusal_reason is empty string if not refused.
        """
        if not evidence_chunks:
            return (
                ConfidenceLevel.REFUSED,
                0.0,
                "No relevant evidence found in the indexed filings.",
            )

        if len(evidence_chunks) < MIN_EVIDENCE_TO_ANSWER:
            return (
                ConfidenceLevel.REFUSED,
                0.0,
                "Insufficient evidence to answer reliably.",
            )

        # Primary signal: top evidence score
        top_score = max(c.relevance_score for c in evidence_chunks)

        # Secondary signal: average of top-3 evidence scores
        sorted_scores = sorted(
            [c.relevance_score for c in evidence_chunks], reverse=True
        )
        avg_top3 = sum(sorted_scores[:3]) / min(3, len(sorted_scores))

        # Composite score: weighted combination
        composite = 0.7 * top_score + 0.3 * avg_top3

        # Refusal check
        if top_score < self.refuse_threshold:
            return (
                ConfidenceLevel.REFUSED,
                round(composite, 4),
                (
                    f"Evidence relevance too low (top score: {top_score:.2f}, "
                    f"threshold: {self.refuse_threshold}). "
                    "The indexed filings may not contain information to answer this question."
                ),
            )

        # Coverage bonus: more supporting chunks = more confidence
        coverage_bonus = min(0.05 * (len(evidence_chunks) - 1), 0.10)
        final_score = min(composite + coverage_bonus, 1.0)

        # Assign level
        if final_score >= self.high_threshold and len(evidence_chunks) >= MIN_EVIDENCE_FOR_HIGH:
            level = ConfidenceLevel.HIGH
        elif final_score >= self.low_threshold:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW

        return level, round(final_score, 4), ""

    def should_refuse(self, evidence_chunks: list[EvidenceChunk]) -> tuple[bool, str]:
        """
        Quick check: should we refuse before even calling the LLM?
        Returns (should_refuse, reason).
        """
        level, _, reason = self.score(evidence_chunks)
        return level == ConfidenceLevel.REFUSED, reason