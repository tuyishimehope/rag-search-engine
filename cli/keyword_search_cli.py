import argparse
import math
import pickle
from collections import defaultdict, Counter
from pathlib import Path
import sys

from nltk.stem import PorterStemmer

from lib.search_utils import BM25_K1, load_movies, tokenize_term, tokenize_text

stemmer = PorterStemmer()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "cache"
INDEX_PATH = CACHE_DIR / "index.pkl"
DOCMAP_PATH = CACHE_DIR / "docmap.pkl"
TERM_FREQUENCIES_PATH = PROJECT_ROOT / 'term_frequencies.pkl'


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = INDEX_PATH
        self.docmap_path = DOCMAP_PATH
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
        self.term_frequencies_path = TERM_FREQUENCIES_PATH

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

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
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1):
        tf = self.get_tf(doc_id, term)
        saturation_tf = (tf * (k1 + 1)) / (tf + k1)
        return saturation_tf

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def load_index_or_exit() -> InvertedIndex:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    search_parser = subparsers.add_parser(
        "search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser(
        "build", help="Build the inverted index and save it to disk")
    tf_parser = subparsers.add_parser(
        "tf", help="Show how often a term appears in a document")
    tf_parser.add_argument("document_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to look up")

    idf_parser = subparsers.add_parser("idf", help="idf parser")
    idf_parser.add_argument("term", help="term")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Show the TF-IDF score of a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to score")
    
    bm25_idf_parser = subparsers.add_parser(
    "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")

            idx = load_index_or_exit()

            for doc_id in idx.search(args.query):
                print(f"{doc_id}. {idx.docmap[doc_id]['title']}")

        case "build":
            build_command()
        case "tf":
            idx = load_index_or_exit()
            token = tokenize_term(args.term)
            print(idx.get_tf(args.document_id, token))
        case "idf":
            idx = load_index_or_exit()
            token = tokenize_term(args.term)
            idf = idx.get_idf(token)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            idx = load_index_or_exit()
            token = tokenize_term(args.term)
            tf_idf = idx.get_tf_idf(args.doc_id, token)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            idx = load_index_or_exit()
            token = tokenize_term(args.term)
            bm25idf = idx.get_bm25_idf(term=token)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()