import regex as re
import numpy as np
from sentence_transformers import SentenceTransformer

from lib.search_utils import CACHE_DIR, load_movies

EMBEDDINGS_PATH = CACHE_DIR / "movie_embeddings.npy"


class SemanticSearch():
    def __init__(self):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = model
        self.embeddings = None
        self.documents = {}
        self.document_map = {}

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("Text must not be empty or whitespace only")
        embeddings = self.model.encode([text])
        return embeddings[0]

    def _register_documents(self, documents: list[dict]) -> None:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

    def build_embeddings(self, documents: list[dict]):
        self._register_documents(documents)
        texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self._register_documents(documents)
        if EMBEDDINGS_PATH.exists():
            self.embeddings = np.load(EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        scored = []
        for doc, doc_embedding in zip(self.documents, self.embeddings):
            scored.append(
                (cosine_similarity(query_embedding, doc_embedding), doc))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {
                "score": score,
                "title": doc["title"],
                "description": doc["description"],
            }
            for score, doc in scored[:limit]
        ]


def verify_model():
    semanticSearch = SemanticSearch()
    print(f"Model loaded: {semanticSearch.model}")
    print(f"Max sequence length: {semanticSearch.model.max_seq_length}")


def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    semantic_search = SemanticSearch()
    documents = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def search_query(query, limit):
    semantic_search = SemanticSearch()
    documents = load_movies()
    semantic_search.load_or_create_embeddings(documents)
    results = semantic_search.search(query, limit)
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result['title']} (score: {result['score']:.4f})")
        print(f"  {result['description'][:100]}...")
        print()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size < 1:
        raise ValueError("chunk size must be at least 1")
    words = text.split()
    chunks = []
    for start in range(0, len(words), chunk_size):
        if overlap > 0 and start > 0:
            chunks.append(" ".join(words[start - overlap:start + chunk_size]))
        else:
            chunks.append(" ".join(words[start:start + chunk_size]))
    return chunks


def semantic_chunking(text: str, max_chunk_size: int, overlap: int) -> list[str]:
    output = re.split(r"(?<=[.!?])\s+", text)
    if max_chunk_size < 1:
        raise ValueError("chunk size must be at least 1")

    chunks = []
    for start in range(0, len(output), max_chunk_size):
        if overlap > 0 and start > 0:
            chunks.append(" ".join(output[start - overlap:start + max_chunk_size]))
        else:
            chunks.append(" ".join(output[start:start + max_chunk_size]))
    return chunks
