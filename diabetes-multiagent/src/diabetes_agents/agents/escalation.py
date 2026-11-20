import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from ..llm import LLM
from ..models import EscalationTicket

SUMMARY_SYSTEM = """[ESCALATION_SUMMARY] You write a concise handoff note for a clinician in SBAR format
(Situation, Background, Assessment, Recommendation), max 80 words. Use only facts from the transcript;
do not invent vitals or history."""

PATIENT_NOTICES = {
    "emergency": (
        "This could be a medical emergency. Please call your local emergency number (911 in the US) "
        "or go to the nearest emergency department right now. If someone is unconscious or having a "
        "seizure, do not give food or drink by mouth. Do not wait for an online answer. "
        "I have also alerted your care team."
    ),
    "self_harm": (
        "I'm really sorry you're going through this. You deserve support right now. If you are in "
        "immediate danger, call your local emergency number. In the US you can call or text 988 "
        "(Suicide & Crisis Lifeline). I have also alerted your care team so a person can reach out."
    ),
    "urgent": (
        "What you describe should be looked at by a healthcare professional today. Please contact "
        "your doctor or diabetes care team now. If symptoms get worse, seek emergency care. "
        "I have flagged this for your care team."
    ),
    "dosing": (
        "I can't recommend or change medication or insulin doses, because the right dose depends on "
        "your personal situation. I have passed your question to your care team. Please contact your "
        "doctor, diabetes educator, or pharmacist."
    ),
    "routine": (
        "I don't have reliable information to answer that safely, so I have flagged it for your care "
        "team to follow up. You can also contact your doctor, diabetes educator, or pharmacist."
    ),
}


class FileNotifier:
    """Writes tickets to a JSONL queue. Replace with email/SMS/EHR/pager integration."""

    def __init__(self, directory: str):
        self.path = Path(directory) / "queue.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def notify(self, ticket: EscalationTicket) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ticket.__dict__) + "\n")

    def list_tickets(self) -> List[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]


class EscalationAgent:
    """Hands the case to a human: builds a clinician note, queues a ticket, and tells the patient what to do."""

    def __init__(self, llm: LLM, notifier: FileNotifier):
        self.llm, self.notifier = llm, notifier

    def patient_notice(self, urgency: str, category: str = "") -> str:
        key = category if category in PATIENT_NOTICES else urgency
        return PATIENT_NOTICES.get(key, PATIENT_NOTICES["routine"])

    def escalate(self, session_id: str, urgency: str, reasons: List[str], transcript: List[dict], category: str = "") -> EscalationTicket:
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in transcript[-6:])
        try:
            summary = self.llm.complete(
                SUMMARY_SYSTEM,
                f"Urgency: {urgency}\nReasons: {'; '.join(reasons)}\n\nTranscript:\n{convo}",
                max_tokens=200,
            ).strip()
        except Exception:
            summary = f"Auto-summary unavailable. Reasons: {'; '.join(reasons)}"

        ticket = EscalationTicket(
            id=uuid.uuid4().hex[:8],
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            session_id=session_id,
            urgency=urgency,
            reasons=reasons,
            summary=summary,
            patient_message=self.patient_notice(urgency, category),
        )
        self.notifier.notify(ticket)
        return ticket
