"""Speech-to-text with word-level timestamps, via faster-whisper."""

from __future__ import annotations

from pathlib import Path

from .media import extract_audio
from .transcript import Transcript, Word

DEFAULT_MODEL = "small"


class TranscriptionError(RuntimeError):
    pass


def transcribe(
    source: str | Path,
    *,
    model_size: str = DEFAULT_MODEL,
    language: str | None = None,
    device: str = "auto",
    compute_type: str = "default",
    vad_filter: bool = True,
    workdir: str | Path | None = None,
    progress: bool = True,
) -> Transcript:
    """Transcribe ``source`` (any media file) into a word-level Transcript."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - depends on env
        raise TranscriptionError(
            "faster-whisper is not installed. Run: pip install faster-whisper\n"
            "Alternatively pass --transcript to reuse a transcript JSON."
        ) from exc

    source = Path(source)
    workdir = Path(workdir) if workdir else source.parent
    audio = extract_audio(source, workdir / f"{source.stem}.16k.wav")

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(audio),
        language=language,
        word_timestamps=True,
        vad_filter=vad_filter,
        beam_size=5,
    )

    words: list[Word] = []
    for segment in segments:
        if progress:
            print(f"  [{segment.start:7.1f}s] {segment.text.strip()[:70]}", flush=True)
        for word in segment.words or []:
            text = word.word.strip()
            if not text:
                continue
            words.append(
                Word(
                    text=text,
                    start=float(word.start),
                    end=float(word.end),
                    probability=float(getattr(word, "probability", 1.0) or 1.0),
                )
            )

    if not words:
        raise TranscriptionError(
            f"no speech detected in {source}. Try --model medium or --no-vad."
        )
    return Transcript(words=words, language=getattr(info, "language", language or "en"))
