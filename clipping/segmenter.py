"""Turn a flat word stream into sentences, then into candidate clip windows."""

from __future__ import annotations

from dataclasses import dataclass

from .transcript import Sentence, Transcript, Word

# A word ending in one of these closes a sentence.
TERMINATORS = ".?!…"
# A silence at least this long also closes one, even mid-punctuation.
PAUSE_SPLIT = 0.65


def split_sentences(
    transcript: Transcript,
    *,
    pause_split: float = PAUSE_SPLIT,
    max_words: int = 60,
) -> list[Sentence]:
    """Group words into sentences using punctuation and pauses."""
    sentences: list[Sentence] = []
    current: list[Word] = []
    prev: Word | None = None

    for word in transcript.words:
        if current and prev is not None and word.start - prev.end >= pause_split:
            sentences.append(Sentence(current))
            current = []
        current.append(word)
        stripped = word.text.strip().rstrip('"\'")]')
        if stripped.endswith(tuple(TERMINATORS)) or len(current) >= max_words:
            sentences.append(Sentence(current))
            current = []
        prev = word

    if current:
        sentences.append(Sentence(current))
    return [s for s in sentences if len(s) > 0]


@dataclass
class Candidate:
    """A contiguous run of sentences that could become a clip."""

    start: float
    end: float
    sentences: list[Sentence]
    score: float = 0.0
    reasons: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.reasons is None:
            self.reasons = []

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences).strip()

    @property
    def words(self) -> list[Word]:
        return [w for s in self.sentences for w in s.words]

    def overlap(self, other: "Candidate") -> float:
        """Fraction of the shorter window that the two windows share."""
        span = min(self.end, other.end) - max(self.start, other.start)
        if span <= 0:
            return 0.0
        shortest = min(self.duration, other.duration)
        return span / shortest if shortest else 0.0


def build_candidates(
    sentences: list[Sentence],
    *,
    min_duration: float = 15.0,
    max_duration: float = 60.0,
    stride: int = 1,
) -> list[Candidate]:
    """Every run of consecutive sentences whose length fits the target window.

    Clips always start and end on a sentence boundary, which is what keeps them
    from opening mid-thought.
    """
    candidates: list[Candidate] = []
    for i in range(0, len(sentences), stride):
        run: list[Sentence] = []
        for j in range(i, len(sentences)):
            run.append(sentences[j])
            duration = run[-1].end - run[0].start
            if duration > max_duration:
                # A single sentence longer than the window still deserves a shot.
                if len(run) == 1:
                    candidates.append(
                        Candidate(run[0].start, run[0].start + max_duration, list(run))
                    )
                break
            if duration >= min_duration:
                candidates.append(Candidate(run[0].start, run[-1].end, list(run)))
    return candidates


def select(
    candidates: list[Candidate],
    *,
    count: int,
    max_overlap: float = 0.25,
) -> list[Candidate]:
    """Greedy non-maximum suppression: best first, skipping overlaps."""
    chosen: list[Candidate] = []
    for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
        if any(cand.overlap(other) > max_overlap for other in chosen):
            continue
        chosen.append(cand)
        if len(chosen) >= count:
            break
    return sorted(chosen, key=lambda c: c.start)
