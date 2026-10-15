import re
import sys
import time
from textwrap import dedent

from lib.llm import call_llm

RERANK_METHODS = ["individual"]

# Re-rank a wider pool than we will show: the LLM can only promote a movie
# that RRF let into the pool, so a bigger pool gives it more to work with.
FETCH_MULTIPLIER = 5

# Free models are flaky (refusals, chatter instead of a number, API errors),
# so every document gets a few tries before we give up on it.
MAX_ATTEMPTS = 3

# Seconds to wait between LLM calls so the free tier doesn't rate-limit us.
DELAY_SECONDS = 3.0


def build_individual_prompt(query: str, doc: dict) -> str:
    return dedent(f"""\
        Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {doc.get("title", "")} - {doc.get("document", "")}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match).
        Output ONLY the number in your response, no other text or explanation.

        Score:""")


def parse_score(text: str | None) -> float | None:
    """Pull the first number out of the reply; None means 'unusable, retry'."""
    if not text:
        return None
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match is None:
        return None
    score = float(match.group())
    return score if 0 <= score <= 10 else None


def score_document(query: str, doc: dict, *, llm=call_llm,
                   sleep=time.sleep, delay: float = DELAY_SECONDS,
                   attempts: int = MAX_ATTEMPTS) -> float:
    prompt = build_individual_prompt(query, doc)
    for attempt in range(1, attempts + 1):
        try:
            score = parse_score(llm(prompt, show_usage=False))
        except RuntimeError:
            # Missing API key: retrying cannot fix it, so fail loudly.
            raise
        except Exception as error:
            print(f"Warning: LLM call failed for '{doc.get('title', '')}' "
                  f"(attempt {attempt}/{attempts}): {error}", file=sys.stderr)
            score = None
        if score is not None:
            return score
        if attempt < attempts:
            sleep(delay)
    print(f"Warning: no usable score for '{doc.get('title', '')}', using 0",
          file=sys.stderr)
    return 0.0


def rerank_individual(query: str, results: list[dict], limit: int, *,
                      llm=call_llm, sleep=time.sleep,
                      delay: float = DELAY_SECONDS) -> list[dict]:
    reranked = []
    for i, result in enumerate(results):
        if i > 0:
            sleep(delay)
        score = score_document(query, result, llm=llm, sleep=sleep, delay=delay)
        reranked.append({**result, "rerank_score": score})
    # sorted() is stable, so equal LLM scores keep their RRF order.
    reranked.sort(key=lambda r: r["rerank_score"], reverse=True)
    return reranked[:limit]


def rerank(query: str, results: list[dict], method: str, limit: int, **kwargs) -> list[dict]:
    match method:
        case "individual":
            return rerank_individual(query, results, limit, **kwargs)
        case _:
            raise ValueError(f"Unknown rerank method: {method}")
