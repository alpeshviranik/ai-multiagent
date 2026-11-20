from ..llm import LLM
from ..models import KnowledgeResult
from ..retriever import Retriever

SYSTEM = """[QUERY_REWRITE] You are a medical search assistant. Rewrite the patient's question as a short
list of clinical keywords and synonyms (e.g. "my sugar is low" -> "hypoglycemia low blood glucose").
Reply with the keywords only, no sentences."""


class KnowledgeAgent:
    """Retrieves vetted medical knowledge. It never talks to the patient."""

    def __init__(self, llm: LLM, retriever: Retriever, top_k: int = 3, min_score: float = 0.05):
        self.llm, self.retriever, self.top_k, self.min_score = llm, retriever, top_k, min_score

    def _expand(self, question: str) -> str:
        try:
            return self.llm.complete(SYSTEM, f"Question: {question}", max_tokens=60).strip()
        except Exception:
            return ""  # retrieval still works with the original question

    def retrieve(self, question: str) -> KnowledgeResult:
        expanded = self._expand(question)
        hits = self.retriever.search(f"{question} {expanded}", k=self.top_k, min_score=self.min_score)
        return KnowledgeResult(
            passages=[p for p, _ in hits],
            top_score=hits[0][1] if hits else 0.0,
            expanded_query=expanded,
        )
