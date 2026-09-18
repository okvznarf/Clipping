"""Device-fallback behaviour, exercised with a stand-in for faster-whisper."""

import sys
import types

import pytest

from clipping import transcribe as mod
from clipping.transcribe import TranscriptionError, transcribe


class FakeWord:
    def __init__(self, word, start, end):
        self.word, self.start, self.end, self.probability = word, start, end, 0.9


class FakeSegment:
    def __init__(self):
        self.start, self.text = 0.0, "hello there"
        self.words = [FakeWord(" hello", 0.0, 0.5), FakeWord(" there", 0.5, 1.0)]


class FakeModel:
    """Raises the real CTranslate2 error, lazily during generation, on cuda."""

    created: list[tuple[str, str]] = []

    def __init__(self, size, device="auto", compute_type="default"):
        FakeModel.created.append((device, compute_type))
        self.device = device

    def transcribe(self, audio, **kwargs):
        def generate():
            if self.device in ("auto", "cuda"):
                raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
            yield FakeSegment()

        return generate(), types.SimpleNamespace(language="en")


@pytest.fixture
def fake_whisper(monkeypatch, tmp_path):
    FakeModel.created = []
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    # Skip the real ffmpeg audio extraction.
    monkeypatch.setattr(mod, "extract_audio", lambda src, dest, **kw: dest)
    source = tmp_path / "video.mp4"
    source.write_bytes(b"stub")
    return source


def test_auto_falls_back_to_cpu_when_cuda_runtime_is_missing(fake_whisper, capsys):
    """Regression: a missing cublas DLL used to abort the whole run."""
    result = transcribe(fake_whisper, device="auto", progress=False)

    assert result.text == "hello there"
    assert [d for d, _ in FakeModel.created] == ["auto", "cpu"]
    # The retry picks a compute type that is actually fast on CPU.
    assert FakeModel.created[1][1] == "int8"
    assert "falling back to CPU" in capsys.readouterr().out


def test_explicit_cuda_reports_instead_of_falling_back(fake_whisper):
    with pytest.raises(TranscriptionError, match="--device cpu"):
        transcribe(fake_whisper, device="cuda", progress=False)
    assert [d for d, _ in FakeModel.created] == ["cuda"]


def test_explicit_compute_type_survives_the_fallback(fake_whisper):
    transcribe(fake_whisper, device="auto", compute_type="float32", progress=False)
    assert FakeModel.created[1] == ("cpu", "float32")


def test_cpu_is_used_directly_without_a_probe(fake_whisper):
    transcribe(fake_whisper, device="cpu", progress=False)
    assert FakeModel.created == [("cpu", "default")]


def test_unrelated_runtime_errors_are_not_swallowed(fake_whisper, monkeypatch):
    class Broken(FakeModel):
        def transcribe(self, audio, **kwargs):
            def generate():
                raise RuntimeError("model file is corrupt")
                yield

            return generate(), types.SimpleNamespace(language="en")

    sys.modules["faster_whisper"].WhisperModel = Broken
    with pytest.raises(RuntimeError, match="corrupt"):
        transcribe(fake_whisper, device="auto", progress=False)


def test_silence_is_reported_clearly(fake_whisper):
    class Silent(FakeModel):
        def transcribe(self, audio, **kwargs):
            return iter(()), types.SimpleNamespace(language="en")

    sys.modules["faster_whisper"].WhisperModel = Silent
    with pytest.raises(TranscriptionError, match="no speech detected"):
        transcribe(fake_whisper, device="cpu", progress=False)
