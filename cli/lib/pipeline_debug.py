"""Opt-in debug logging for the search pipeline.

Nothing is printed unless enable_debug() is called (the CLI's --debug flag).
Logs go to stderr so they never mix with the normal results on stdout.
"""
import logging
import sys

logger = logging.getLogger("rrf_pipeline")


def enable_debug() -> None:
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[debug] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


def log_query(stage: str, query: str) -> None:
    logger.debug("%s: %r", stage, query)


def log_results(stage: str, results: list[dict]) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug("%s (%d results)", stage, len(results))
    for position, result in enumerate(results, start=1):
        details = []
        # Re-ranking scores only exist after the re-rank stage.
        if "cross_encoder_score" in result:
            details.append(f"cross_encoder={result['cross_encoder_score']:.3f}")
        if "rerank_score" in result:
            details.append(f"llm_score={result['rerank_score']:.1f}")
        if "rerank_rank" in result:
            details.append(f"llm_rank={result['rerank_rank']}")
        details.append(f"rrf={result['rrf_score']:.4f}")
        details.append(f"bm25_rank={result['bm25_rank']}")
        details.append(f"semantic_rank={result['semantic_rank']}")
        logger.debug("  %2d. %s (id=%s) | %s", position, result["title"],
                     result["id"], ", ".join(details))
