import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from clipping.transcript import Transcript, Word


def speak(script: str, *, wps: float = 2.8, gap: float = 0.06) -> Transcript:
    """Build a transcript from text with plausible, monotonic word timings."""
    words: list[Word] = []
    t = 0.0
    for token in script.split():
        duration = len(token) / (wps * 5)
        words.append(Word(token, round(t, 3), round(t + duration, 3)))
        t += duration + (0.35 if token.endswith((".", "?", "!")) else gap)
    return Transcript(words)


@pytest.fixture
def sample_transcript() -> Transcript:
    return speak(
        "Here's the truth nobody tells you about starting a business. "
        "Most people wait until they feel ready. I was that person for three years. "
        "Then I realized readiness arrives after you start, never before. "
        "So what should you actually do? Pick the smallest version and ship it. "
        "The biggest mistake is building for six months in silence. "
        "Um so yeah like that is kind of the whole idea really."
    )
