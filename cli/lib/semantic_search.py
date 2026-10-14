import numpy as np
from sentence_transformers import SentenceTransformer

from lib.search_utils import CACHE_DIR, load_movies

EMBEDDINGS_PATH = CACHE_DIR / "movie_embeddings.npy"


class SemanticSearch():
    def __init__(self):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = model
        self.embeddings = None
        self.documents = None
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
