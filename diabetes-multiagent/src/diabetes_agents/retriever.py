"""TF-IDF retrieval over markdown sections (swap for embeddings in production)."""
from pathlib import Path
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Passage


def load_passages(knowledge_dir: str) -> List[Passage]:
    passages: List[Passage] = []
    for path in sorted(Path(knowledge_dir).glob("*.md")):
        title, section, buf = path.stem, "Overview", []

        def flush():
            body = "\n".join(buf).strip()
            if body:
                passages.append(Passage(f"{title} > {section}", body))

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                flush()
                buf.clear()
                section = line[3:].strip()
            else:
                buf.append(line)
        flush()
    return passages


class Retriever:
    def __init__(self, knowledge_dir: str):
        self.passages = load_passages(knowledge_dir)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        corpus = [f"{p.source} {p.text}" for p in self.passages]
        self.matrix = self.vectorizer.fit_transform(corpus) if corpus else None

    def search(self, query: str, k: int = 3, min_score: float = 0.05) -> List[Tuple[Passage, float]]:
        if self.matrix is None:
            return []
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.passages[i], float(scores[i])) for i in ranked if scores[i] >= min_score]
