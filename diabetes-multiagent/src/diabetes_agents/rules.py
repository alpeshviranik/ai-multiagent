"""Deterministic red-flag rules. Cheap, fast, and independent of any LLM."""
import re
from typing import Optional, Tuple

SELF_HARM = [
    r"\b(kill myself|end my life|suicid|want to die|hurt myself)\b",
    r"\b(overdose|take all my insulin).{0,30}(on purpose|to die)\b",
]

EMERGENCY = [
    r"\b(unconscious|passed out|unresponsive|not waking up|won'?t wake)\b",
    r"\bseizure|convulsion",
    r"\b(can'?t|cannot|trouble|difficulty) breath",
    r"\bchest pain\b",
    r"\bfruity (smell|breath)\b",
    r"\b(vomiting|throwing up).{0,40}(ketone|can'?t keep|diabet)",
    r"\bketones?\b.{0,30}\b(high|large|moderate)\b",
    r"\b(severe|extreme) (confusion|drowsiness)\b",
    r"\bglucose.{0,15}\b(under|below|<)\s*(40)\b",
]

DOSING = [
    r"\b(how (much|many)|what).{0,30}\b(units?|dose|dosage)\b",
    r"\b(change|increase|decrease|adjust|skip|stop|double)\b.{0,30}\b(insulin|metformin|medication|medicine|dose)\b",
]

# Patterns that must never appear in an answer shown to a patient
UNSAFE_DRAFT = [
    r"\b(take|inject|give yourself|increase to|decrease to|use)\b.{0,25}\b\d+(\.\d+)?\s*(units?|mg|ml)\b",
    r"\byou (definitely |probably )?have (type [12] )?diabetes\b",
    r"\b(stop|skip) (taking )?your (insulin|medication|medicine)\b",
]


def _any(patterns, text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def classify(text: str) -> Optional[Tuple[str, str]]:
    """Return (category, risk) for the first matching red flag, else None."""
    if _any(SELF_HARM, text):
        return ("self_harm", "emergency")
    if _any(EMERGENCY, text):
        return ("emergency", "emergency")
    if _any(DOSING, text):
        return ("dosing", "moderate")
    return None


def draft_violations(text: str) -> list:
    return [p for p in UNSAFE_DRAFT if re.search(p, text, re.IGNORECASE)]
