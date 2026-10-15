import json
import re
import sys
import time
from textwrap import dedent

from lib.llm import call_llm

SCORE_MIN, SCORE_MAX = 0, 3

# Free models are flaky, so the single request gets a few tries.
MAX_ATTEMPTS = 3
DELAY_SECONDS = 3.0

# Each result is shown to the judge as one short line.
DESCRIPTION_CHARS = 200

# Dedented *before* values are filled in, so multi-line values can't break it.
EVALUATION_TEMPLATE = dedent("""\
    Rate how relevant each result is to this query on a 0-3 scale:

    Query: "{query}"

    Results:
    {formatted_results}

    Scale:
    - 3: Highly relevant
    - 2: Relevant
    - 1: Marginally relevant
    - 0: Not relevant

    Do NOT give any numbers other than 0, 1, 2, or 3.

    Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

    [2, 0, 3, 2, 0, 1]""")


def build_evaluation_prompt(query: str, results: list[dict]) -> str:
    formatted_results = "\n".join(
        f"{i}. {result['title']}: "
        f"{' '.join(result['document'].split())[:DESCRIPTION_CHARS]}"
        for i, result in enumerate(results, start=1)
    )
    return EVALUATION_TEMPLATE.format(
        query=query, formatted_results=formatted_results)


def parse_scores(text: str | None, expected_count: int) -> list[int] | None:
    """Parse the judge's reply into exactly one 0-3 score per result.

    None means 'unusable, retry'. A list of the wrong length is rejected
    rather than guessed at: we could not tell which score belongs to which
    movie.
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
        if (isinstance(data, list) and len(data) == expected_count
                and all(isinstance(score, int) and not isinstance(score, bool)
                        and SCORE_MIN <= score <= SCORE_MAX for score in data)):
            return data
    return None


def evaluate_results(query: str, results: list[dict], *, llm=call_llm,
                     sleep=time.sleep, delay: float = DELAY_SECONDS,
                     attempts: int = MAX_ATTEMPTS) -> list[int] | None:
    """Ask the LLM to rate every result; None if it never gave usable scores."""
    if not results:
        return []
    prompt = build_evaluation_prompt(query, results)
    for attempt in range(1, attempts + 1):
        try:
            scores = parse_scores(llm(prompt, show_usage=False), len(results))
        except RuntimeError:
            # Missing API key: retrying cannot fix it, so fail loudly.
            raise
        except Exception as error:
            print(f"Warning: LLM call failed (attempt {attempt}/{attempts}): "
                  f"{error}", file=sys.stderr)
            scores = None
        if scores is not None:
            return scores
        if attempt < attempts:
            sleep(delay)
    print("Warning: no usable evaluation scores from the LLM", file=sys.stderr)
    return None
