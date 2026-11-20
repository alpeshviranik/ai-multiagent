import json

from diabetes_agents.config import Settings
from diabetes_agents.llm import FakeLLM
from diabetes_agents.orchestrator import Orchestrator


class SpyLLM(FakeLLM):
    """Fake LLM that records which agent roles were called."""
    def __init__(self):
        self.calls = []

    def complete(self, system, user, history=None, max_tokens=500):
        self.calls.append(system.split("]")[0].strip("["))
        return super().complete(system, user, history, max_tokens)


class UnsafeDraftLLM(FakeLLM):
    """Patient agent always produces a dosing instruction."""
    def complete(self, system, user, history=None, max_tokens=500):
        if "[PATIENT_DRAFT]" in system or "[PATIENT_REVISE]" in system:
            return "You should take 10 units of insulin now."
        return super().complete(system, user, history, max_tokens)


class BrokenSafetyLLM(FakeLLM):
    """Safety classifier returns garbage."""
    def complete(self, system, user, history=None, max_tokens=500):
        if "[SAFETY_INPUT]" in system:
            return "I am not JSON"
        return super().complete(system, user, history, max_tokens)


def make(tmp_path, llm=None):
    s = Settings(escalation_dir=str(tmp_path))
    return Orchestrator(s, llm or FakeLLM())


def test_normal_question_is_answered_with_sources(tmp_path):
    r = make(tmp_path).handle("s", "How do I treat low blood sugar with the 15-15 rule?")
    assert not r.escalated and r.sources and "Hypoglycemia" in r.sources[0]
    assert [e.agent for e in r.trace][:3] == ["safety", "knowledge", "patient"]


def test_emergency_skips_llm_answer_and_creates_urgent_ticket(tmp_path):
    spy = SpyLLM()
    r = make(tmp_path, spy).handle("s", "My dad is unconscious and diabetic")
    assert r.escalated and r.ticket.urgency == "emergency"
    assert "PATIENT_DRAFT" not in spy.calls and "SAFETY_INPUT" not in spy.calls
    assert "911" in r.answer


def test_self_harm_escalates(tmp_path):
    r = make(tmp_path).handle("s", "I want to die")
    assert r.ticket.urgency == "emergency" and "988" in r.answer


def test_dosing_question_goes_to_clinician(tmp_path):
    r = make(tmp_path).handle("s", "Should I increase my insulin dose tonight?")
    assert r.escalated and r.ticket.urgency == "routine" and not r.sources


def test_unknown_topic_escalates_routine(tmp_path):
    r = make(tmp_path).handle("s", "quantum chromodynamics lagrangian")
    assert r.escalated and r.ticket.urgency == "routine"


def test_unsafe_draft_is_blocked_after_revision(tmp_path):
    r = make(tmp_path, UnsafeDraftLLM()).handle("s", "How do I treat low blood sugar?")
    assert r.escalated and "10 units" not in r.answer
    actions = [e.action for e in r.trace]
    assert "revise" in actions and r.ticket.urgency == "routine"


def test_classifier_failure_fails_safe(tmp_path):
    r = make(tmp_path, BrokenSafetyLLM()).handle("s", "How do I treat low blood sugar?")
    assert r.escalated and r.ticket is not None  # answered but flagged for a human


def test_tickets_are_persisted(tmp_path):
    bot = make(tmp_path)
    bot.handle("s", "He had a seizure")
    tickets = bot.notifier.list_tickets()
    assert len(tickets) == 1 and tickets[0]["urgency"] == "emergency"
    json.dumps(tickets)  # serialisable
