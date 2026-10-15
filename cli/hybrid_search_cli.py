import argparse

from lib.search_utils import load_movies, normalize_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    sub_parsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = sub_parsers.add_parser("normalize", help="List of scores")
    normalize_parser.add_argument(
        "scores", nargs="*", type=float, help="List of scores")

    weighted_parser = sub_parsers.add_parser(
        "weighted-search", help="Hybrid search weighting BM25 and semantic scores")
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Weight of the BM25 score (0 = semantic only, 1 = keyword only)")
    weighted_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()

    match args.command:
        case "normalize":
            for score in normalize_scores(args.scores):
                print(f"* {score:.4f}")
        case "weighted-search":
            from lib.hybrid_search import HybridSearch

            hybrid_search = HybridSearch(load_movies())
            results = hybrid_search.weighted_search(
                args.query, args.alpha, args.limit)
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']}")
                print(f"  Hybrid Score: {result['hybrid_score']:.3f}")
                print(
                    f"  BM25: {result['bm25_score']:.3f}, Semantic: {result['semantic_score']:.3f}")
                print(f"  {result['document'][:100]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()