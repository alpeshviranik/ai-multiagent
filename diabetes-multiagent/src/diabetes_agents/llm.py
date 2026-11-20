"""LLM abstraction: a real Anthropic client and a deterministic offline fake."""
import json
import re
from typing import List, Optional, Protocol

from .config import Settings


class LLM(Protocol):
    def complete(
        self,
        system: str,
        user: str,
        history: Optional[List[dict]] = None,
        max_tokens: int = 500,
    ) -> str: ...


class AnthropicLLM:
    def __init__(self, api_key: str, model: str):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, system, user, history=None, max_tokens=500) -> str:
        messages = list(history or []) + [{"role": "user", "content": user}]
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system, messages=messages
        )
        return "".join(b.text for b in resp.content if b.type == "text")


class FakeLLM:
    """Deterministic stand-in so the whole pipeline runs offline (DEMO_MODE=1) and in tests.

    It recognises each agent by a marker tag in the system prompt, e.g. [SAFETY_INPUT].
    """

    def complete(self, system, user, history=None, max_tokens=500) -> str:
        if "[SAFETY_INPUT]" in system:
            return json.dumps({"risk": "none", "category": "none", "reasons": []})
        if "[SAFETY_REVIEW]" in system:
            return json.dumps({"approved": True, "issues": []})
        if "[QUERY_REWRITE]" in system:
            return user.split("Question:", 1)[-1].strip()
        if "[PATIENT_DRAFT]" in system or "[PATIENT_REVISE]" in system:
            m = re.search(r"Context passages:\n(.*?)\n\nPatient question:", user, re.S)
            body = (m.group(1) if m else "").strip()
            first = body.split("\n\n")[0]
            first = re.sub(r"^\[[^\]]*\]\n", "", first)
            return "Here is what our patient information says:\n" + first[:600]
        if "[ESCALATION_SUMMARY]" in system:
            return "S: Patient message flagged by safety review. B: See reasons. A: Needs human review. R: Clinician follow-up."
        return ""


def get_llm(settings: Settings) -> LLM:
    if settings.demo_mode or not settings.api_key:
        return FakeLLM()
    return AnthropicLLM(settings.api_key, settings.model_name)


def extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of an LLM reply. Returns None if unparseable."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None
