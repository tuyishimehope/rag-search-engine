import argparse
import math
import pickle
from collections import defaultdict, Counter
from pathlib import Path
import sys

from nltk.stem import PorterStemmer

from lib.search_utils import BM25_B, BM25_K1, InvertedIndex, load_movies, tokenize_term, tokenize_text

stemmer = PorterStemmer()


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
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument(
        "term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return")

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
        case "bm25tf":
            idx = load_index_or_exit()
            token = tokenize_term(args.term)
            bm25tf = idx.get_bm25_tf(
                doc_id=args.doc_id, term=token, k1=args.k1,  b=args.b)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case "bm25search":
            idx = load_index_or_exit()
            results = idx.bm25_search(args.query, args.limit)
            for rank, (doc_id, score) in enumerate(results, start=1):
                print(
                    f"{rank}. ({doc_id}) {idx.docmap[doc_id]['title']} - Score: {score:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
