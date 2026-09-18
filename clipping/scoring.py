"""Heuristic ranking of clip candidates.

No model required: the score combines signals that correlate with clips people
actually watch — a hook in the opening line, a self-contained payoff, steady
speech, and a duration in the sweet spot for short-form feeds.

Each signal returns roughly 0..1 and is combined with the weights in WEIGHTS,
so the final score is comparable across videos.
"""

from __future__ import annotations

import math
import re

from .segmenter import Candidate

HOOK_PATTERNS = [
    r"\bhere'?s (?:the|why|how|what)\b",
    r"\bthe (?:truth|secret|problem|reason|trick|mistake|thing)\b",
    r"\bmost people\b",
    r"\bnobody (?:tells|talks)\b",
    r"\bwhat (?:if|most|nobody)\b",
    r"\byou (?:should|need to|have to|never|always)\b",
    r"\bi (?:was|used to|never|learned|realized)\b",
    r"\bthe biggest\b",
    r"\blet me (?:tell|explain|show)\b",
    r"\bthis is (?:why|how|what)\b",
    r"\bimagine\b",
    r"\bturns out\b",
]

EMPHASIS_WORDS = {
    "never", "always", "everyone", "nobody", "everything", "nothing",
    "huge", "insane", "crazy", "wild", "shocking", "brutal", "obsessed",
    "literally", "actually", "honestly", "seriously", "completely",
    "best", "worst", "first", "only", "biggest", "hardest", "fastest",
    "free", "secret", "mistake", "truth", "proof", "wrong", "right",
}

FILLER_WORDS = {"um", "uh", "erm", "hmm", "mm", "like", "yeah", "okay", "so", "right"}

WEIGHTS = {
    "hook": 0.26,
    "emphasis": 0.14,
    "question": 0.08,
    "numbers": 0.08,
    "duration": 0.18,
    "pace": 0.12,
    "completeness": 0.10,
    "cleanliness": 0.04,
}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def hook_score(text: str) -> float:
    """Does the opening line promise something? Only the first ~15 words count."""
    opening = " ".join(_words(text)[:15])
    hits = sum(1 for pattern in HOOK_PATTERNS if re.search(pattern, opening))
    return min(1.0, hits / 2)


def emphasis_score(text: str) -> float:
    words = _words(text)
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in EMPHASIS_WORDS)
    # ~1 emphatic word per 12 spoken words already reads as energetic.
    return min(1.0, hits / max(1.0, len(words) / 12))


def question_score(text: str) -> float:
    return min(1.0, text.count("?") / 2)


def number_score(text: str) -> float:
    hits = len(re.findall(r"\b\d[\d,.]*\b|\b(?:one|two|three|five|ten|hundred|thousand|million|billion|percent)\b", text.lower()))
    return min(1.0, hits / 3)


def duration_score(duration: float, *, ideal: float = 32.0, spread: float = 18.0) -> float:
    """Bell curve around the length that performs best in short-form feeds."""
    return math.exp(-((duration - ideal) ** 2) / (2 * spread**2))


def pace_score(word_count: int, duration: float) -> float:
    """Reward a natural 2.2-3.4 words/sec; punish dead air and auctioneering."""
    if duration <= 0:
        return 0.0
    wps = word_count / duration
    if wps < 1.2 or wps > 5.0:
        return 0.0
    return max(0.0, 1.0 - abs(wps - 2.8) / 1.6)


def completeness_score(text: str) -> float:
    """A clip that ends on a finished thought beats one that trails off."""
    stripped = text.strip()
    if not stripped:
        return 0.0
    score = 0.5 if stripped[-1] in ".?!…" else 0.0
    first = _words(stripped)[:1]
    # Opening on a dangling conjunction means we joined mid-thought.
    if first and first[0] not in {"and", "but", "so", "because", "or", "then", "which"}:
        score += 0.5
    return score


def cleanliness_score(text: str) -> float:
    words = _words(text)
    if not words:
        return 0.0
    fillers = sum(1 for w in words if w in FILLER_WORDS)
    return max(0.0, 1.0 - fillers / max(1.0, len(words) / 10))


def score_candidate(candidate: Candidate, *, ideal_duration: float = 32.0) -> Candidate:
    text = candidate.text
    word_count = len(candidate.words)
    signals = {
        "hook": hook_score(text),
        "emphasis": emphasis_score(text),
        "question": question_score(text),
        "numbers": number_score(text),
        "duration": duration_score(candidate.duration, ideal=ideal_duration),
        "pace": pace_score(word_count, candidate.duration),
        "completeness": completeness_score(text),
        "cleanliness": cleanliness_score(text),
    }
    candidate.score = round(sum(WEIGHTS[k] * v for k, v in signals.items()), 4)
    candidate.reasons = [
        f"{name} {value:.2f}"
        for name, value in sorted(signals.items(), key=lambda kv: -kv[1])
        if value > 0.15
    ]
    return candidate


def score_all(candidates: list[Candidate], *, ideal_duration: float = 32.0) -> list[Candidate]:
    return [score_candidate(c, ideal_duration=ideal_duration) for c in candidates]


def suggest_title(text: str, *, max_chars: int = 70) -> str:
    """A short, plain title taken from the clip's strongest opening sentence."""
    sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s.strip()]
    if not sentences:
        return "Clip"
    best = max(sentences[:3], key=lambda s: hook_score(s) + emphasis_score(s) + question_score(s))
    best = re.sub(r"\s+", " ", best).strip(" .,-")
    if len(best) > max_chars:
        best = best[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    return best[:1].upper() + best[1:]
