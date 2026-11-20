from typing import List, Optional

from ..llm import LLM
from ..models import KnowledgeResult

DRAFT_SYSTEM = """[PATIENT_DRAFT] You are a supportive assistant that answers patient questions about diabetes.
Rules:
- Use ONLY the provided context passages. If they don't answer the question, say you don't have reliable information.
- Never diagnose. Never recommend, change, or quantify medication or insulin doses.
- Plain, kind language at about an 8th-grade reading level. Under 150 words. Short lists are fine.
- If symptoms sound urgent, advise seeking emergency care.
- Mention confirming personal decisions with their care team when relevant."""

REVISE_SYSTEM = """[PATIENT_REVISE] You are revising a draft answer to a diabetes patient that failed a safety review.
Fix every listed issue. Keep using ONLY the provided context. No diagnoses, no dosing advice. Under 150 words."""


def format_context(knowledge: KnowledgeResult) -> str:
    return "\n\n".join(f"[{p.source}]\n{p.text}" for p in knowledge.passages)


class PatientAgent:
    """Talks to the patient. Drafts answers grounded in retrieved knowledge."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def draft(self, question: str, knowledge: KnowledgeResult, history: Optional[List[dict]] = None) -> str:
        user = f"Context passages:\n{format_context(knowledge)}\n\nPatient question: {question}"
        return self.llm.complete(DRAFT_SYSTEM, user, history=history, max_tokens=500).strip()

    def revise(self, question: str, draft: str, issues: List[str], knowledge: KnowledgeResult) -> str:
        user = (
            f"Context passages:\n{format_context(knowledge)}\n\nPatient question: {question}\n\n"
            f"Previous draft:\n{draft}\n\nSafety issues to fix:\n- " + "\n- ".join(issues)
        )
        return self.llm.complete(REVISE_SYSTEM, user, max_tokens=500).strip()
