import json

import regex as re
import numpy as np
from sentence_transformers import SentenceTransformer

from lib.search_utils import CACHE_DIR, load_movies

EMBEDDINGS_PATH = CACHE_DIR / "movie_embeddings.npy"
CHUNKS_EMBEDDINGS_PATH = CACHE_DIR / "chunk_embeddings.npy"
CHUNK_METADATA_PATH = CACHE_DIR / "chunk_metadata.json"


class SemanticSearch():
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        model = SentenceTransformer(model_name)
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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self._register_documents(documents)
        all_chunks = []
        chunk_metadata = []

        for movie_idx, doc in enumerate(documents):
            if not doc["description"] or not doc["description"].strip():
                continue
            chunks = semantic_chunking(
                doc["description"], max_chunk_size=4, overlap=1)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks),
                })

        self.chunk_embeddings = self.model.encode(
            all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(CHUNKS_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w") as file:
            json.dump({"chunks": chunk_metadata,
                      "total_chunks": len(all_chunks)}, file, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self._register_documents(documents)
        if CHUNKS_EMBEDDINGS_PATH.exists() and CHUNK_METADATA_PATH.exists():
            self.chunk_embeddings = np.load(CHUNKS_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, "r") as file:
                self.chunk_metadata = json.load(file)["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first.")

        query_embedding = self.generate_embedding(query)

        # 1. score every chunk; metadata row i describes embedding row i
        chunk_scores = []
        for chunk_embedding, meta in zip(self.chunk_embeddings, self.chunk_metadata):
            chunk_scores.append({
                "chunk_idx": meta["chunk_idx"],
                "movie_idx": meta["movie_idx"],
                "score": cosine_similarity(query_embedding, chunk_embedding),
            })

        # 2. a movie is as good as its best chunk
        best_per_movie = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            if (movie_idx not in best_per_movie
                    or chunk_score["score"] > best_per_movie[movie_idx]["score"]):
                best_per_movie[movie_idx] = chunk_score

        # 3. rank movies, keep the top `limit`, shape the output
        ranked = sorted(best_per_movie.values(),
                        key=lambda cs: cs["score"], reverse=True)
        results = []
        for chunk_score in ranked[:limit]:
            doc = self.documents[chunk_score["movie_idx"]]
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"][:100],
                "score": round(float(chunk_score["score"]), 3),
            })
        return results


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
    if max_chunk_size < 1:
        raise ValueError("chunk size must be at least 1")
    if overlap < 0 or overlap >= max_chunk_size:
        raise ValueError(
            "overlap must be at least 0 and smaller than the chunk size")

    sentences = re.split(r"(?<=[.!?])\s+", text)
    step = max_chunk_size - overlap
    chunks = []
    start = 0
    while start < len(sentences):
        window = sentences[start:start + max_chunk_size]
        if chunks and len(window) <= overlap:
            break  # nothing new in this window, only the overlap
        chunks.append(" ".join(window))
        start += step
    return chunks
