import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    api_key: str = ""
    model_name: str = "claude-sonnet-5-5"
    demo_mode: bool = False
    knowledge_dir: str = str(ROOT / "data" / "knowledge")
    escalation_dir: str = str(ROOT / "escalations")
    top_k: int = 3
    min_retrieval_score: float = 0.05
    max_revisions: int = 1
    max_history_turns: int = 6
    escalate_on_no_knowledge: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            model_name=os.getenv("MODEL_NAME", "claude-sonnet-5-5"),
            demo_mode=_flag("DEMO_MODE", "0"),
            knowledge_dir=os.getenv("KNOWLEDGE_DIR", str(ROOT / "data" / "knowledge")),
            escalation_dir=os.getenv("ESCALATION_DIR", str(ROOT / "escalations")),
            top_k=int(os.getenv("TOP_K", "3")),
            min_retrieval_score=float(os.getenv("MIN_RETRIEVAL_SCORE", "0.05")),
            max_revisions=int(os.getenv("MAX_REVISIONS", "1")),
            max_history_turns=int(os.getenv("MAX_HISTORY_TURNS", "6")),
            escalate_on_no_knowledge=_flag("ESCALATE_ON_NO_KNOWLEDGE", "1"),
        )
