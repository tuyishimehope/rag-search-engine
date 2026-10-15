import argparse

from lib.query_enhancement import ENHANCE_METHODS, enhance_query
from lib.reranking import FETCH_MULTIPLIER, RERANK_METHODS, rerank
from lib.search_utils import load_movies, normalize_scores


def print_rrf_results(results: list[dict]) -> None:
    for i, result in enumerate(results, start=1):
        print(f"{i}. {result['title']}")
        print(f"  RRF Score: {result['rrf_score']:.3f}")
        print(
            f"  BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}")
        print(f"  {result['document'][:100]}...")
        print()


def print_reranked_results(query: str, k: int, results: list[dict],
                           method: str) -> None:
    print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):\n")
    for i, result in enumerate(results, start=1):
        print(f"{i}. {result['title']}")
        match method:
            case "batch":
                print(f"   Re-rank Rank: {result['rerank_rank']}")
            case "cross_encoder":
                print(
                    f"   Cross Encoder Score: {result['cross_encoder_score']:.3f}")
            case _:
                print(f"   Re-rank Score: {result['rerank_score']:.3f}/10")
        print(f"   RRF Score: {result['rrf_score']:.3f}")
        print(
            f"   BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}")
        print(f"   {result['document'][:100]}...")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    sub_parsers = parser.add_subparsers(
        dest="command", help="Available commands")

    normalize_parser = sub_parsers.add_parser(
        "normalize", help="List of scores")
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

    rrf_parser = sub_parsers.add_parser(
        "rrf-search", help="Hybrid search using Reciprocal Rank Fusion")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument(
        "-k", type=int, default=60,
        help="RRF constant: higher values flatten the influence of top ranks")
    rrf_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return")
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=ENHANCE_METHODS,
        help="Query enhancement method",
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=RERANK_METHODS,
        help="Re-rank the RRF results with an LLM",
    )

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
        case "rrf-search":
            from lib.hybrid_search import HybridSearch

            query = args.query
            if args.enhance:
                query = enhance_query(args.query, args.enhance)
                print(
                    f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")

            hybrid_search = HybridSearch(load_movies())

            if args.rerank_method:
                # Over-fetch so the LLM has a wider pool to promote from.
                results = hybrid_search.rrf_search(
                    query, args.k, args.limit * FETCH_MULTIPLIER)
                print(
                    f"Re-ranking top {args.limit} results using {args.rerank_method} method...")
                results = rerank(query, results,
                                 args.rerank_method, args.limit)
                print_reranked_results(
                    query, args.k, results, args.rerank_method)
            else:
                print_rrf_results(hybrid_search.rrf_search(
                    query, args.k, args.limit))
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
