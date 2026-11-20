"""Typed messages passed between agents."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Passage:
    source: str
    text: str


@dataclass
class KnowledgeResult:
    passages: List[Passage]
    top_score: float
    expanded_query: str = ""

    @property
    def found(self) -> bool:
        return bool(self.passages)


@dataclass
class SafetyVerdict:
    risk: str = "none"          # none | low | moderate | high | emergency
    category: str = "none"      # none | emergency | self_harm | dosing | symptom_concern | unclear
    reasons: List[str] = field(default_factory=list)
    source: str = "rules"       # rules | llm | fail-safe


@dataclass
class ReviewVerdict:
    approved: bool
    issues: List[str] = field(default_factory=list)


@dataclass
class EscalationTicket:
    id: str
    created_at: str
    session_id: str
    urgency: str                # emergency | urgent | routine
    reasons: List[str]
    summary: str
    patient_message: str
    status: str = "open"


@dataclass
class TraceEvent:
    agent: str
    action: str
    detail: str


@dataclass
class FinalResponse:
    answer: str
    sources: List[str]
    escalated: bool
    risk: str
    ticket: Optional[EscalationTicket] = None
    trace: List[TraceEvent] = field(default_factory=list)
