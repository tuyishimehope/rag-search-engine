import os

from lib.search_utils import InvertedIndex, normalize_scores
from .semantic_search import ChunkedSemanticSearch


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()
    

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        fetch = limit * 500
        bm25_results = self._bm25_search(query, fetch)
        semantic_results = self.semantic_search.search_chunks(query, fetch)

        bm25_norm = normalize_scores([r["score"] for r in bm25_results])
        semantic_norm = normalize_scores([r["score"] for r in semantic_results])

        docs_by_id = {doc["id"]: doc for doc in self.documents}
        combined: dict[int, dict] = {}

        def entry_for(doc_id: int) -> dict:
            if doc_id not in combined:
                doc = docs_by_id[doc_id]
                combined[doc_id] = {
                    "id": doc_id,
                    "title": doc["title"],
                    "document": doc["description"],
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                }
            return combined[doc_id]

        for result, score in zip(bm25_results, bm25_norm):
            entry_for(result["doc_id"])["bm25_score"] = score
        for result, score in zip(semantic_results, semantic_norm):
            entry_for(result["id"])["semantic_score"] = score

        for entry in combined.values():
            entry["hybrid_score"] = hybrid_score(
                entry["bm25_score"], entry["semantic_score"], alpha)

        ranked = sorted(combined.values(),
                        key=lambda e: (-e["hybrid_score"], e["id"]))
        return ranked[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    