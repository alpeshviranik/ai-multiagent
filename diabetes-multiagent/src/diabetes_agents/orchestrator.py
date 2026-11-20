from collections import defaultdict
from typing import Dict, List, Optional

from .agents.escalation import EscalationAgent, FileNotifier
from .agents.knowledge import KnowledgeAgent
from .agents.patient import PatientAgent, format_context
from .agents.safety import SafetyAgent
from .config import Settings
from .llm import LLM, get_llm
from .models import FinalResponse, TraceEvent
from .retriever import Retriever

SAFE_FALLBACK = (
    "I'm not able to answer that safely. I've flagged your question for your care team, "
    "and you can also contact your doctor, diabetes educator, or pharmacist."
)


class Orchestrator:
    """Coordinates the four agents:  Safety -> Knowledge -> Patient -> Safety(review) -> [Escalation]."""

    def __init__(self, settings: Optional[Settings] = None, llm: Optional[LLM] = None):
        self.s = settings or Settings.from_env()
        self.llm = llm or get_llm(self.s)
        self.safety = SafetyAgent(self.llm)
        self.knowledge = KnowledgeAgent(
            self.llm, Retriever(self.s.knowledge_dir), self.s.top_k, self.s.min_retrieval_score
        )
        self.patient = PatientAgent(self.llm)
        self.notifier = FileNotifier(self.s.escalation_dir)
        self.escalation = EscalationAgent(self.llm, self.notifier)
        self.history: Dict[str, List[dict]] = defaultdict(list)

    # ------------------------------------------------------------------
    def handle(self, session_id: str, message: str) -> FinalResponse:
        trace: List[TraceEvent] = []
        log = lambda agent, action, detail="": trace.append(TraceEvent(agent, action, detail))
        history = self.history[session_id][-self.s.max_history_turns * 2:]

        # 1) Safety agent: triage the incoming message
        verdict = self.safety.assess_input(message, history)
        log("safety", "assess_input", f"risk={verdict.risk} category={verdict.category} via={verdict.source}")

        if verdict.risk == "emergency" or verdict.risk == "high" or verdict.category == "dosing":
            urgency = {"emergency": "emergency", "high": "urgent"}.get(verdict.risk, "routine")
            return self._escalate(session_id, message, urgency, verdict.reasons, verdict.category, trace, log)

        # 2) Knowledge agent: retrieve vetted passages
        knowledge = self.knowledge.retrieve(message)
        log("knowledge", "retrieve",
            f"{len(knowledge.passages)} passages, top_score={knowledge.top_score:.2f}, expanded='{knowledge.expanded_query}'")

        if not knowledge.found:
            if self.s.escalate_on_no_knowledge:
                return self._escalate(session_id, message, "routine",
                                      ["No reliable knowledge found for this question"], "routine", trace, log)
            return self._finish(session_id, message, SAFE_FALLBACK, [], False, verdict.risk, None, trace)

        # 3) Patient agent drafts, 4) Safety agent reviews (with bounded revisions)
        context = format_context(knowledge)
        draft = self.patient.draft(message, knowledge, history)
        log("patient", "draft", f"{len(draft)} chars")

        approved, issues = False, []
        for attempt in range(self.s.max_revisions + 1):
            review = self.safety.review_draft(draft, context)
            log("safety", "review_draft", f"approved={review.approved} issues={review.issues}")
            if review.approved:
                approved = True
                break
            issues = review.issues
            if attempt < self.s.max_revisions:
                draft = self.patient.revise(message, draft, review.issues, knowledge)
                log("patient", "revise", f"attempt {attempt + 1}")

        if not approved:
            return self._escalate(session_id, message, "routine",
                                  ["Draft answer failed safety review"] + issues, "routine", trace, log,
                                  override_message=SAFE_FALLBACK)

        # 5) Moderate-risk but answerable: answer AND queue a follow-up for a human
        ticket, escalated, answer = None, False, draft
        if verdict.risk == "moderate":
            ticket = self.escalation.escalate(session_id, "routine", verdict.reasons,
                                              self.history[session_id] + [{"role": "user", "content": message}],
                                              verdict.category)
            escalated = True
            answer += "\n\nI've also flagged this for your care team to follow up."
            log("escalation", "ticket", f"{ticket.id} urgency={ticket.urgency}")

        return self._finish(session_id, message, answer, [p.source for p in knowledge.passages],
                            escalated, verdict.risk, ticket, trace)

    # ------------------------------------------------------------------
    def _escalate(self, session_id, message, urgency, reasons, category, trace, log, override_message=None):
        transcript = self.history[session_id] + [{"role": "user", "content": message}]
        ticket = self.escalation.escalate(session_id, urgency, reasons, transcript, category)
        log("escalation", "ticket", f"{ticket.id} urgency={urgency}")
        return self._finish(session_id, message, override_message or ticket.patient_message,
                            [], True, "emergency" if urgency == "emergency" else urgency, ticket, trace)

    def _finish(self, session_id, message, answer, sources, escalated, risk, ticket, trace):
        self.history[session_id] += [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        return FinalResponse(answer, sources, escalated, risk, ticket, trace)
