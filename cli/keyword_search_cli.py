import argparse
import pickle
from collections import defaultdict
from pathlib import Path

from nltk.stem import PorterStemmer

from lib.search_utils import load_movies, load_stop_words, normalize, tokenize_text

stemmer = PorterStemmer()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "cache"
INDEX_PATH = CACHE_DIR / "index.pkl"
DOCMAP_PATH = CACHE_DIR / "docmap.pkl"


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        for token in set(tokenize_text(text)):
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term, set()))

    def build(self) -> None:
        for movie in load_movies():
            doc_id = movie["id"]
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}")
            self.docmap[doc_id] = movie

    def save(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump(self.index, f)
        with open(DOCMAP_PATH, "wb") as f:
            pickle.dump(self.docmap, f)


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    search_parser = subparsers.add_parser(
        "search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Build the inverted index and save it to disk")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")

          
            movies = load_movies()
            stop_words = load_stop_words()
            query_tokens = filter_stop_words(normalize(args.query).split(), stop_words)
            results = []
            for movie in movies:
                title_tokens = filter_stop_words(normalize(movie["title"]).split(), stop_words)
                if matches(query_tokens, title_tokens):
                    results.append(movie)
                    if len(results) == 5:
                        break

            for i, movie in enumerate(results, start=1):
                print(f"{i}. {movie['title']}")
        
        case "build":
            build_command()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
