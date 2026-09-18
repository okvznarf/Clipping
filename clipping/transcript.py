"""Transcript data model.

A transcript is a flat list of words with timestamps. Everything downstream
(sentence splitting, clip selection, caption timing) is derived from it, so the
model stays deliberately small and JSON round-trippable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Iterator, Sequence


@dataclass
class Word:
    """A single spoken word with its time span, in seconds."""

    text: str
    start: float
    end: float
    probability: float = 1.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Sentence:
    """Consecutive words that form one utterance."""

    words: list[Word] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words).strip()

    def __len__(self) -> int:
        return len(self.words)


@dataclass
class Transcript:
    words: list[Word] = field(default_factory=list)
    language: str = "en"

    def __len__(self) -> int:
        return len(self.words)

    def __iter__(self) -> Iterator[Word]:
        return iter(self.words)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words).strip()

    @property
    def duration(self) -> float:
        return self.words[-1].end if self.words else 0.0

    def slice(self, start: float, end: float) -> list[Word]:
        """Words whose midpoint falls inside ``[start, end]``."""
        out = []
        for w in self.words:
            mid = (w.start + w.end) / 2
            if start <= mid <= end:
                out.append(w)
        return out

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        return {"language": self.language, "words": [asdict(w) for w in self.words]}

    @classmethod
    def from_dict(cls, data: dict) -> "Transcript":
        words = [
            Word(
                text=str(w["text"]),
                start=float(w["start"]),
                end=float(w["end"]),
                probability=float(w.get("probability", 1.0)),
            )
            for w in data.get("words", [])
        ]
        return cls(words=words, language=str(data.get("language", "en")))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Transcript":
        return cls.from_dict(json.loads(Path(path).read_text()))


def shift(words: Sequence[Word], offset: float) -> list[Word]:
    """Re-base word timings so ``offset`` becomes t=0 (used per clip)."""
    return [
        Word(
            text=w.text,
            start=max(0.0, w.start - offset),
            end=max(0.0, w.end - offset),
            probability=w.probability,
        )
        for w in words
    ]


def merge_short_words(words: Iterable[Word], min_duration: float = 0.04) -> list[Word]:
    """Give degenerate zero-length words a floor so captions never flash."""
    out: list[Word] = []
    for w in words:
        if w.duration < min_duration:
            w = Word(w.text, w.start, w.start + min_duration, w.probability)
        out.append(w)
    return out
