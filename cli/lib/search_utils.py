from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import pickle
import string

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "cache"
INDEX_PATH = CACHE_DIR / "index.pkl"
DOCMAP_PATH = CACHE_DIR / "docmap.pkl"
TERM_FREQUENCIES_PATH = CACHE_DIR / 'term_frequencies.pkl'
DOC_LENGTHS_PATH = CACHE_DIR / 'doc_lengths.pkl'


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = INDEX_PATH
        self.docmap_path = DOCMAP_PATH
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
        self.term_frequencies_path = TERM_FREQUENCIES_PATH
        self.doc_lengths = {}
        self.doc_lengths_path = DOC_LENGTHS_PATH

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term, set()))

    def build(self) -> None:
        for movie in load_movies():
            doc_id = movie["id"]
            self.__add_document(
                doc_id, f"{movie['title']} {movie['description']}")
            self.docmap[doc_id] = movie

    def save(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump(self.index, f)
        with open(DOCMAP_PATH, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(TERM_FREQUENCIES_PATH, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(DOC_LENGTHS_PATH, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        if not (self.index_path.exists() and self.docmap_path.exists() and self.term_frequencies_path.exists()):
            raise FileNotFoundError(
                "Index not found. Run the 'build' command first.")
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def search(self, query: str, limit: int = 5) -> list[int]:
        results: list[int] = []
        seen: set[int] = set()
        for token in tokenize_text(query):
            for doc_id in self.get_documents(token):
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                results.append(doc_id)
                if len(results) == limit:
                    return results
        return results

    def get_tf(self, doc_id, term):
        return self.term_frequencies.get(doc_id, Counter())[term]

    def get_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index.get(term, set()))
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.index.get(term, set()))
        result = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return result

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        if tf == 0:
            return 0.0
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * \
            (doc_length / avg_doc_length) if avg_doc_length else 1.0
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)

        BM25 = bm25_tf * bm25_idf
        return BM25

    def bm25_search(self, query: str, limit: int = 5) -> list[tuple[int, float]]:
        tokens = tokenize_text(query)
        if not tokens:
            return []
        scores: dict[int, float] = {}
        for doc_id in self.docmap:
            scores[doc_id] = sum(self.bm25(doc_id, t) for t in tokens)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:limit]


def load_movies():
    with open("data/movies.json", "r") as file:
        movies = json.load(file)["movies"]
        return movies


def normalize(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return text.lower().translate(table)


def load_stop_words():
    stop_words = []
    with open("data/stopwords.txt", "r") as words_file:
        data = words_file.read().splitlines()
        stop_words = [normalize(word) for word in data]
    return stop_words


def tokenize_text(text: str) -> list[str]:
    tokens = normalize(text).split()
    stop_words = load_stop_words()
    return [stemmer.stem(t) for t in tokens if t not in stop_words]


def tokenize_term(term: str) -> str:
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError(
            f"Expected exactly one token from {term!r}, got {len(tokens)}")
    return tokens[0]


def matches(query_tokens: list[str], title_tokens: list[str]) -> bool:
    return any(q in t for q in query_tokens for t in title_tokens)


def filter_stop_words(words, stop_words) -> list[str]:
    results = []
    for word in words:
        if word in stop_words:
            pass
        else:
            results.append(stemmer.stem(word))
    return results
