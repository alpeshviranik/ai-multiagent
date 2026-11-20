from typing import List, Optional

from .. import rules
from ..llm import LLM, extract_json
from ..models import ReviewVerdict, SafetyVerdict

RISK_ORDER = {"none": 0, "low": 1, "moderate": 2, "high": 3, "emergency": 4}

INPUT_SYSTEM = """[SAFETY_INPUT] You are a clinical safety triage classifier for a diabetes support chatbot.
Assess the patient's latest message. Reply with JSON only:
{"risk": "none|low|moderate|high|emergency",
 "category": "none|emergency|self_harm|dosing|symptom_concern",
 "reasons": ["short reason", ...]}
- emergency: life-threatening (unconscious, seizure, DKA signs, severe low glucose, chest pain, self-harm)
- high: worrying symptoms needing same-day clinical contact
- moderate: concerning but non-urgent, or a request that needs a clinician's judgment
- low/none: general education questions
When unsure, choose the HIGHER risk."""

REVIEW_SYSTEM = """[SAFETY_REVIEW] You review a draft chatbot answer for a diabetes patient.
Reject the draft if it: gives or changes a medication/insulin dose, diagnoses the patient, contradicts the
context, adds medical claims not in the context, or tells the patient to ignore their care team.
Reply with JSON only: {"approved": true|false, "issues": ["short issue", ...]}"""


class SafetyAgent:
    """Two-layer safety: deterministic rules + LLM judgment. Fails safe on any error."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def assess_input(self, text: str, history: Optional[List[dict]] = None) -> SafetyVerdict:
        rule = rules.classify(text)
        # Layer 1: hard red flags never wait for an LLM
        if rule and rule[1] == "emergency":
            return SafetyVerdict("emergency", rule[0], [f"Red-flag rule matched: {rule[0]}"], "rules")

        verdict = SafetyVerdict()
        if rule:
            verdict = SafetyVerdict(rule[1], rule[0], [f"Rule matched: {rule[0]}"], "rules")

        # Layer 2: LLM triage
        try:
            raw = self.llm.complete(INPUT_SYSTEM, f"Patient message: {text}", history=history, max_tokens=200)
            data = extract_json(raw)
        except Exception:
            data = None

        if data is None or data.get("risk") not in RISK_ORDER:
            # Fail safe: if we cannot classify, treat as moderate and flag for human follow-up
            if RISK_ORDER[verdict.risk] < RISK_ORDER["moderate"]:
                return SafetyVerdict("moderate", verdict.category if verdict.category != "none" else "unclear",
                                     verdict.reasons + ["Safety classifier unavailable or unparseable"], "fail-safe")
            return verdict

        llm_risk = data["risk"]
        reasons = [str(r) for r in data.get("reasons", [])]
        if RISK_ORDER[llm_risk] > RISK_ORDER[verdict.risk]:
            category = data.get("category", "none")
            return SafetyVerdict(llm_risk, category, verdict.reasons + reasons, "llm")
        verdict.reasons += reasons
        return verdict

    def review_draft(self, draft: str, context: str) -> ReviewVerdict:
        issues = [f"Unsafe pattern in draft: {p}" for p in rules.draft_violations(draft)]
        try:
            raw = self.llm.complete(
                REVIEW_SYSTEM, f"Context:\n{context}\n\nDraft answer:\n{draft}", max_tokens=200
            )
            data = extract_json(raw)
        except Exception:
            data = None

        if data is None or "approved" not in data:
            issues.append("Safety reviewer unavailable or unparseable")  # fail safe: reject
            return ReviewVerdict(False, issues)

        issues += [str(i) for i in data.get("issues", [])]
        return ReviewVerdict(bool(data["approved"]) and not issues, issues)
