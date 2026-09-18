"""Speech-to-text with word-level timestamps, via faster-whisper."""

from __future__ import annotations

from pathlib import Path

from .media import extract_audio
from .transcript import Transcript, Word

DEFAULT_MODEL = "small"

#: Substrings that mark a failure to load the CUDA runtime rather than a real
#: transcription problem. CTranslate2 raises these lazily, during generation,
#: and only names the missing library (e.g. "cublas64_12.dll is not found").
CUDA_LOAD_MARKERS = ("cublas", "cudnn", "cuda", "cudart", "libcu")


class TranscriptionError(RuntimeError):
    pass


def _is_cuda_load_failure(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in CUDA_LOAD_MARKERS)


def _cpu_compute_type(compute_type: str) -> str:
    # int8 is several times faster than float32 on CPU at no real accuracy cost
    # for this use, so prefer it whenever the caller has not chosen explicitly.
    return "int8" if compute_type == "default" else compute_type


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
    """Transcribe ``source`` (any media file) into a word-level Transcript.

    With ``device="auto"`` a GPU is used when one is usable, falling back to CPU
    if the CUDA runtime libraries turn out to be missing. An explicit
    ``device="cuda"`` never falls back silently — it reports what is missing.
    """
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

    def run(on_device: str, with_compute: str) -> tuple[list[Word], str]:
        """Transcribe on one device, consuming the generator so that a lazy
        CUDA load failure surfaces here rather than in the caller."""
        model = WhisperModel(model_size, device=on_device, compute_type=with_compute)
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
        return words, getattr(info, "language", language or "en")

    try:
        words, detected = run(device, compute_type)
    except RuntimeError as exc:
        if not _is_cuda_load_failure(exc):
            raise
        if device == "cuda":
            raise TranscriptionError(
                f"CUDA was requested but its runtime is not usable: {exc}\n"
                "Install the NVIDIA cuBLAS and cuDNN libraries, or re-run with "
                "--device cpu."
            ) from exc
        print(
            f"! GPU unusable ({exc}); falling back to CPU. "
            "Pass --device cpu to skip this attempt.",
            flush=True,
        )
        words, detected = run("cpu", _cpu_compute_type(compute_type))

    if not words:
        raise TranscriptionError(
            f"no speech detected in {source}. Try --model medium or --no-vad."
        )
    return Transcript(words=words, language=detected)
