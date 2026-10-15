import json
import re
import sys
import time
from textwrap import dedent

from lib.llm import call_llm

RERANK_METHODS = ["individual", "batch"]

# Re-rank a wider pool than we will show: the LLM can only promote a movie
# that RRF let into the pool, so a bigger pool gives it more to work with.
FETCH_MULTIPLIER = 5

# Free models are flaky (refusals, chatter instead of a number, API errors),
# so every request gets a few tries before we give up on it.
MAX_ATTEMPTS = 3

# Seconds to wait between LLM calls so the free tier doesn't rate-limit us.
DELAY_SECONDS = 3.0

# In a batch prompt every movie shares one context window, so each
# description is cut to this many characters to keep the prompt small.
BATCH_DESCRIPTION_CHARS = 200

# Templates are dedented *before* the values are filled in. Dedenting an
# already-filled prompt would break whenever a description contains a newline
# at column 0, because that line would cancel the common indentation.
INDIVIDUAL_TEMPLATE = dedent("""\
    Rate how well this movie matches the search query.

    Query: "{query}"
    Movie: {title} - {document}

    Consider:
    - Direct relevance to query
    - User intent (what they're looking for)
    - Content appropriateness

    Rate 0-10 (10 = perfect match).
    Output ONLY the number in your response, no other text or explanation.

    Score:""")

BATCH_TEMPLATE = dedent("""\
    Rank the movies listed below by relevance to the following search query.

    Query: "{query}"

    Movies:
    {doc_list_str}

    Return the movie IDs in order of relevance, best match first.

    Your response must be a raw JSON array of integers.
    Do not wrap the JSON in Markdown. Do not use a ```json code block.
    Do not include any explanatory text.

    For example:
    [75, 12, 34, 2, 1]

    Ranking:""")


def build_individual_prompt(query: str, doc: dict) -> str:
    return INDIVIDUAL_TEMPLATE.format(
        query=query,
        title=doc.get("title", ""),
        document=doc.get("document", ""),
    )


def build_batch_prompt(query: str, results: list[dict]) -> str:
    doc_list_str = "\n".join(
        f"{doc['id']}: {doc['title']} - "
        f"{' '.join(doc['document'].split())[:BATCH_DESCRIPTION_CHARS]}"
        for doc in results
    )
    return BATCH_TEMPLATE.format(query=query, doc_list_str=doc_list_str)


def parse_score(text: str | None) -> float | None:
    """Pull the first number out of the reply; None means 'unusable, retry'."""
    if not text:
        return None
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match is None:
        return None
    score = float(match.group())
    return score if 0 <= score <= 10 else None


def parse_ranking(text: str | None, valid_ids: set[int]) -> list[int] | None:
    """Turn the model's reply into an ordered list of known movie IDs.

    None means 'unusable, retry'. Models sometimes ignore "no Markdown", so
    we first try the reply as-is and then the first [...] inside it.
    """
    if not text:
        return None
    text = text.strip()
    candidates = [text]
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        candidates.append(match.group())

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue
        ranking: list[int] = []
        for item in data:
            # bool is an int subclass in Python; true/false are not IDs.
            if isinstance(item, int) and not isinstance(item, bool) \
                    and item in valid_ids and item not in ranking:
                ranking.append(item)
        if ranking:
            return ranking
    return None


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


def rerank_batch(query: str, results: list[dict], limit: int, *,
                 llm=call_llm, sleep=time.sleep,
                 delay: float = DELAY_SECONDS,
                 attempts: int = MAX_ATTEMPTS) -> list[dict]:
    if not results:
        return []
    prompt = build_batch_prompt(query, results)
    valid_ids = {doc["id"] for doc in results}

    ranking = None
    for attempt in range(1, attempts + 1):
        try:
            ranking = parse_ranking(llm(prompt, show_usage=False), valid_ids)
        except RuntimeError:
            raise
        except Exception as error:
            print(f"Warning: LLM call failed (attempt {attempt}/{attempts}): "
                  f"{error}", file=sys.stderr)
        if ranking is not None:
            break
        if attempt < attempts:
            sleep(delay)

    if ranking is None:
        print("Warning: no usable ranking from the LLM, keeping RRF order",
              file=sys.stderr)
        ranking = []

    # Movies the model left out keep their RRF order after the ranked ones.
    ranked_ids = ranking + [doc["id"] for doc in results
                            if doc["id"] not in set(ranking)]
    rank_by_id = {doc_id: rank for rank, doc_id in enumerate(ranked_ids, start=1)}
    reranked = [{**doc, "rerank_rank": rank_by_id[doc["id"]]} for doc in results]
    reranked.sort(key=lambda r: r["rerank_rank"])
    return reranked[:limit]


def rerank(query: str, results: list[dict], method: str, limit: int, **kwargs) -> list[dict]:
    match method:
        case "individual":
            return rerank_individual(query, results, limit, **kwargs)
        case "batch":
            return rerank_batch(query, results, limit, **kwargs)
        case _:
            raise ValueError(f"Unknown rerank method: {method}")
