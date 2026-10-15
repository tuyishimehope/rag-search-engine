import sys
from textwrap import dedent

from lib.llm import call_llm

PROMPTS = {
    "spell": dedent("""\
        Fix any spelling errors in the user-provided movie search query below.
        Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
        Preserve punctuation and capitalization unless a change is required for a typo fix.
        If there are no spelling errors, or if you're unsure, output the original query unchanged.
        Output only the final query text, nothing else.

        User query: "{query}"
        """),
    "rewrite": dedent("""\
        Rewrite the user-provided movie search query below to be more specific and searchable.

        Consider:
        - Common movie knowledge (famous actors, popular films)
        - Genre conventions (horror = scary, animation = cartoon)
        - Keep the rewritten query concise (under 10 words)
        - It should be a Google-style search query, specific enough to yield relevant results
        - Don't use boolean logic

        Examples:
        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

        If you cannot improve the query, output the original unchanged.
        Output only the rewritten query text, nothing else.

        User query: "{query}"
        """),
    "expand": dedent("""\
        Expand the user-provided movie search query below with related terms.

        Add synonyms and related concepts that might appear in movie descriptions.
        Keep expansions relevant and focused.
        Output only the additional terms; they will be appended to the original query.

        Examples:
        - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
        - "action movie with bear" -> "action thriller bear chase fight adventure"
        - "comedy with bear" -> "comedy funny bear humor lighthearted"

        User query: "{query}"
        """),
}

ENHANCE_METHODS = list(PROMPTS)


def _clean(response: str | None) -> str:
    """One line, no surrounding whitespace or quotes (LLMs love to add both)."""
    return " ".join((response or "").split()).strip("\"'").strip()


def _append_new_terms(query: str, terms: str) -> str:
    """Original query first, then only the words it doesn't already contain."""
    seen = {word.lower() for word in query.split()}
    new_terms = []
    for word in terms.split():
        if word.lower() not in seen:
            seen.add(word.lower())
            new_terms.append(word)
    return " ".join([query, *new_terms])


def enhance_query(query: str, method: str) -> str:
    if method not in PROMPTS:
        raise ValueError(
            f"Unknown enhancement method {method!r}; choose from {ENHANCE_METHODS}")
    text = _clean(call_llm(PROMPTS[method].format(query=query)))
    if not text:
        print("Warning: the LLM returned no text; using the original query.",
              file=sys.stderr)
        return query
    if method == "expand":
        return _append_new_terms(query, text)
    return text
