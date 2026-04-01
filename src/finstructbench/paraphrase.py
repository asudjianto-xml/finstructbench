"""Hybrid question generation: deterministic logic + LLM paraphrase.

Preserves the deterministic ground truth, graph answer function, and
LLM evaluation prompt while replacing the template-generated natural
language with a more diverse, realistic phrasing produced by an LLM.

Usage:
    from finstructbench.paraphrase import paraphrase_questions

    # questions = list of GeneratedQuestion from deterministic generators
    paraphrased = paraphrase_questions(questions, client, model=...)
"""

import hashlib
import json
import os
from pathlib import Path

from finstructbench.generators.base import GeneratedQuestion

# Cache directory for paraphrased questions (avoids redundant API calls)
_CACHE_DIR = Path(__file__).parent / ".paraphrase_cache"

_SYSTEM_PROMPT = (
    "You are a financial analyst rewriting benchmark questions. "
    "Rewrite the given question into natural, professional English that "
    "a financial analyst would ask. Preserve the EXACT meaning — the "
    "rewritten question must ask for precisely the same information. "
    "Do NOT change entity names, column names, section names, numeric "
    "values, or thresholds. Do NOT add information not present in the "
    "original. Reply with ONLY the rewritten question, nothing else."
)


def _cache_key(question: GeneratedQuestion) -> str:
    """Compute a stable cache key from the question's identity."""
    h = hashlib.sha256()
    h.update(question.qid.encode())
    h.update(question.natural_language.encode())
    h.update(question.category.encode())
    return h.hexdigest()[:16]


def _load_cache() -> dict[str, str]:
    """Load the paraphrase cache from disk."""
    cache_file = _CACHE_DIR / "cache.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict[str, str]):
    """Persist the paraphrase cache to disk."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / "cache.json"
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)


def paraphrase_single(
    client,
    question: GeneratedQuestion,
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """Paraphrase a single question's natural language using an LLM.

    Args:
        client: Anthropic client instance.
        question: The question to paraphrase.
        model: Model to use for paraphrasing.

    Returns:
        The paraphrased natural language string.
    """
    response = client.messages.create(
        model=model,
        max_tokens=256,
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Category: {question.category}\n"
                    f"Original question: {question.natural_language}\n\n"
                    f"Rewrite this question in natural, professional English."
                ),
            }
        ],
    )
    return response.content[0].text.strip()


def paraphrase_questions(
    questions: list[GeneratedQuestion],
    client,
    model: str = "claude-sonnet-4-20250514",
    use_cache: bool = True,
    verbose: bool = True,
) -> list[GeneratedQuestion]:
    """Paraphrase the natural language of questions while preserving ground truth.

    This implements the hybrid mode described in the paper: deterministic
    generators produce question logic and ground truth, then an LLM
    paraphrases the template-generated natural language into diverse,
    realistic phrasing.

    Only the `natural_language` field is modified. The `llm_prompt`,
    `graph_answer_fn`, `ground_truth`, and all other fields are preserved
    unchanged, ensuring that:
      - Ground truth remains provably correct by construction.
      - The LLM under test sees the same structured evaluation prompt.
      - Results are directly comparable to deterministic mode.

    Args:
        questions: List of GeneratedQuestion from deterministic generators.
        client: Anthropic client instance.
        model: Model to use for paraphrasing (not the model under test).
        use_cache: If True, cache paraphrases to avoid redundant API calls.
        verbose: If True, print progress.

    Returns:
        New list of GeneratedQuestion with paraphrased natural_language.
    """
    cache = _load_cache() if use_cache else {}
    result = []
    hits, misses = 0, 0

    for i, q in enumerate(questions):
        key = _cache_key(q)

        if use_cache and key in cache:
            paraphrased_nl = cache[key]
            hits += 1
        else:
            paraphrased_nl = paraphrase_single(client, q, model)
            cache[key] = paraphrased_nl
            misses += 1
            if verbose:
                print(f"  [{i+1}/{len(questions)}] {q.qid}: paraphrased")

        # Create a new GeneratedQuestion with only natural_language changed
        new_q = GeneratedQuestion(
            qid=q.qid,
            category=q.category,
            natural_language=paraphrased_nl,
            graph_answer_fn=q.graph_answer_fn,
            ground_truth=q.ground_truth,
            llm_prompt=q.llm_prompt,          # unchanged
            answer_type=q.answer_type,
            metadata={
                **q.metadata,
                "original_natural_language": q.natural_language,
                "paraphrase_model": model,
            },
        )
        result.append(new_q)

    if use_cache and misses > 0:
        _save_cache(cache)

    if verbose:
        print(f"  Paraphrase: {hits} cached, {misses} generated")

    return result
