"""Optional LLM re-ranking of the shortlist produced by the heuristic scorer.

The heuristics are good at "is this a well-formed, well-paced chunk of speech";
they cannot tell whether the idea inside it is actually interesting. When an
Anthropic API key is available, this module re-scores the top candidates and
writes a title for each. It is strictly optional — the pipeline degrades to the
heuristic score if anything here fails.
"""

from __future__ import annotations

import json
import os
import re

from .segmenter import Candidate

MODEL = "claude-sonnet-5"
MAX_CANDIDATES = 25
MAX_CHARS_PER_CANDIDATE = 1200

SYSTEM_PROMPT = """You rank excerpts from a long video for use as standalone short-form clips.

For each excerpt, judge how well it works on its own, with no surrounding context:
- Does the first sentence make a viewer want the next one?
- Is there a complete idea, with a payoff, inside the excerpt?
- Would someone send it to a friend?

Penalise excerpts that depend on something said earlier, trail off, or are pure filler.

Reply with JSON only, no prose, in this shape:
{"clips": [{"id": 0, "score": 0-100, "title": "<=70 char hook title", "reason": "one short sentence"}]}
Include every id you were given."""


class RankerError(RuntimeError):
    pass


def _payload(candidates: list[Candidate]) -> str:
    items = []
    for index, cand in enumerate(candidates):
        text = cand.text[:MAX_CHARS_PER_CANDIDATE]
        items.append(f"[id {index}] ({cand.duration:.0f}s)\n{text}")
    return "\n\n".join(items)


def _parse(raw: str) -> dict[int, dict]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise RankerError(f"model did not return JSON: {raw[:200]}")
    data = json.loads(match.group(0))
    out: dict[int, dict] = {}
    for item in data.get("clips", []):
        try:
            out[int(item["id"])] = item
        except (KeyError, TypeError, ValueError):
            continue
    return out


def rerank(
    candidates: list[Candidate],
    *,
    model: str = MODEL,
    weight: float = 0.6,
    api_key: str | None = None,
) -> list[Candidate]:
    """Blend an LLM judgement into each candidate's score, in place.

    ``weight`` is how much the model's opinion counts against the heuristic
    score (0 = ignore the model, 1 = trust it entirely).
    """
    if not candidates:
        return candidates
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RankerError("anthropic is not installed. Run: pip install anthropic") from exc

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RankerError("ANTHROPIC_API_KEY is not set")

    shortlist = sorted(candidates, key=lambda c: c.score, reverse=True)[:MAX_CANDIDATES]
    client = anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _payload(shortlist)}],
    )
    raw = "".join(block.text for block in response.content if block.type == "text")
    verdicts = _parse(raw)

    for index, cand in enumerate(shortlist):
        verdict = verdicts.get(index)
        if not verdict:
            continue
        llm_score = max(0.0, min(100.0, float(verdict.get("score", 0)))) / 100.0
        cand.score = round((1 - weight) * cand.score + weight * llm_score, 4)
        title = str(verdict.get("title") or "").strip()
        reason = str(verdict.get("reason") or "").strip()
        if title:
            cand.reasons.insert(0, f"title::{title}")
        if reason:
            cand.reasons.append(f"llm: {reason}")
    return candidates


def title_from(candidate: Candidate) -> str | None:
    """Pull a title the ranker attached to a candidate, if any."""
    for reason in candidate.reasons:
        if reason.startswith("title::"):
            return reason[len("title::"):]
    return None
