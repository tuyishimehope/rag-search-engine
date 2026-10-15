import argparse
import json

from lib.hybrid_search import HybridSearch
from lib.search_utils import PROJECT_ROOT, load_movies

GOLDEN_DATASET_PATH = PROJECT_ROOT / "data" / "golden_dataset.json"
RRF_K = 60


def precision_at_k(retrieved: list[str], relevant: list[str]) -> float:
    """Fraction of the retrieved titles that are in the relevant set."""
    if not retrieved:
        return 0.0
    relevant_set = set(relevant)
    relevant_retrieved = sum(1 for title in retrieved if title in relevant_set)
    return relevant_retrieved / len(retrieved)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with open(GOLDEN_DATASET_PATH, "r") as file:
        test_cases = json.load(file)["test_cases"]

    # Build the search engine once; loading embeddings is the slow part.
    hybrid_search = HybridSearch(load_movies())

    print(f"k={limit}\n")
    for test_case in test_cases:
        query = test_case["query"]
        relevant = test_case["relevant_docs"]

        results = hybrid_search.rrf_search(query, RRF_K, limit)
        retrieved = [result["title"] for result in results]
        precision = precision_at_k(retrieved, relevant)

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved)}")
        print(f"  - Relevant: {', '.join(relevant)}")
        print()


if __name__ == "__main__":
    main()
