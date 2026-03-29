"""
output_schema.py
----------------
Pydantic response models for every answer produced by the RAG pipeline.

Every answer in the system — whether from a single retrieval or a multi-hop
chain — must conform to one of these schemas. This enforces:
  - Structured output that downstream code can rely on
  - Explicit confidence scoring
  - Evidence traceability (which chunks supported the answer)
  - A refuse flag for low-confidence answers

Design principle: the schema is the contract between the retrieval/agent
layer and the API layer. If a field is in the schema, it is always present.
No optional fields that might or might not exist depending on the query type.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConfidenceLevel(str, Enum):
    HIGH = "high"        # rerank score > 0.7, strong evidence
    MEDIUM = "medium"    # rerank score 0.3-0.7, partial evidence
    LOW = "low"          # rerank score < 0.3, weak evidence
    REFUSED = "refused"  # below refusal threshold — answer withheld


class QueryType(str, Enum):
    FACTUAL = "factual"           # single fact lookup
    COMPARATIVE = "comparative"   # compare across companies/years
    ANALYTICAL = "analytical"     # multi-step reasoning
    EXTRACTION = "extraction"     # extract specific clause/term


# ---------------------------------------------------------------------------
# Evidence chunk model
# ---------------------------------------------------------------------------


class EvidenceChunk(BaseModel):
    """A single chunk of evidence supporting an answer."""

    chunk_id: str
    ticker: str
    company_name: str
    filed_date: str
    form_type: str
    section_heading: str
    text: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    is_table: bool = False

    @field_validator("relevance_score")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(v, 4)

    def citation(self) -> str:
        """Human-readable citation string."""
        return (
            f"{self.company_name} {self.form_type} "
            f"({self.filed_date}) — {self.section_heading}"
        )


# ---------------------------------------------------------------------------
# Core answer model
# ---------------------------------------------------------------------------


class Answer(BaseModel):
    """
    The structured answer returned by every query in the pipeline.

    Fields:
        question:        The original user question (echoed back).
        answer:          The generated answer text. Empty string if refused.
        confidence:      HIGH / MEDIUM / LOW / REFUSED.
        confidence_score: Numeric score 0-1 (avg of top evidence scores).
        refused:         True if the system declined to answer.
        refusal_reason:  Why the answer was refused (if refused=True).
        evidence:        Chunks that support the answer (empty if refused).
        query_type:      Detected type of the query.
        sub_queries:     Sub-queries used in multi-hop chain (if any).
        judge_passed:    Whether LLM-as-judge approved the answer.
        judge_feedback:  Judge's feedback (if it flagged issues).
        metadata:        Arbitrary extra metadata (latency, model used, etc.)
    """

    question: str
    answer: str
    confidence: ConfidenceLevel
    confidence_score: float = Field(ge=0.0, le=1.0)
    refused: bool = False
    refusal_reason: str = ""
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    query_type: QueryType = QueryType.FACTUAL
    sub_queries: list[str] = Field(default_factory=list)
    judge_passed: bool = True
    judge_feedback: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence_score")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 4)

    def to_api_response(self) -> dict:
        """Serialise for API response — excludes internal fields."""
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence.value,
            "confidence_score": self.confidence_score,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "evidence": [
                {
                    "citation": e.citation(),
                    "text": e.text[:500],
                    "relevance_score": e.relevance_score,
                    "ticker": e.ticker,
                    "filed_date": e.filed_date,
                    "section": e.section_heading,
                }
                for e in self.evidence
            ],
            "sub_queries": self.sub_queries,
            "judge_passed": self.judge_passed,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Refusal factory
# ---------------------------------------------------------------------------


def make_refusal(
    question: str,
    reason: str,
    confidence_score: float = 0.0,
    metadata: dict | None = None,
) -> Answer:
    """
    Factory function for creating a refused answer.
    Centralises the refusal logic so it's consistent across all callers.
    """
    return Answer(
        question=question,
        answer="",
        confidence=ConfidenceLevel.REFUSED,
        confidence_score=confidence_score,
        refused=True,
        refusal_reason=reason,
        evidence=[],
        judge_passed=False,
        metadata=metadata or {},
    )