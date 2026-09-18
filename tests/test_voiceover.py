import shutil

import pytest

from clipping import voiceover as vo
from clipping.voiceover import VoiceoverError, _wpm, available_engines, synthesize


def test_rate_maps_to_a_sane_word_rate():
    assert _wpm(0) == vo.BASE_WPM
    assert _wpm(20) > _wpm(0) > _wpm(-20)
    # Absurd inputs are clamped rather than producing unusable audio.
    assert 80 <= _wpm(-999) and _wpm(999) <= 400


def test_available_engines_is_a_subset_of_known_engines():
    assert set(available_engines()) <= set(vo.ENGINES)


def test_empty_text_is_rejected(tmp_path):
    with pytest.raises(VoiceoverError, match="empty"):
        synthesize("   ", tmp_path / "vo.wav")


def test_unknown_engine_is_rejected(tmp_path):
    with pytest.raises(VoiceoverError, match="unknown engine"):
        synthesize("hello", tmp_path / "vo.wav", engine="robot")


def test_engines_are_tried_in_order_until_one_works(tmp_path, monkeypatch):
    tried: list[str] = []

    def fail(name):
        def _f(text, dest, voice, rate):
            tried.append(name)
            raise RuntimeError(f"{name} unavailable")

        return _f

    def succeed(text, dest, voice, rate):
        tried.append("espeak")
        dest.write_bytes(b"raw")
        return dest

    monkeypatch.setattr(vo, "available_engines", lambda: ["edge", "sapi", "espeak"])
    monkeypatch.setattr(vo, "_SYNTHS", {"edge": fail("edge"), "sapi": fail("sapi"), "espeak": succeed})
    monkeypatch.setattr(vo, "_to_wav", lambda raw, dest: dest)

    synthesize("hello", tmp_path / "vo.wav")
    assert tried == ["edge", "sapi", "espeak"]


def test_all_engines_failing_reports_each_reason(tmp_path, monkeypatch):
    def fail(text, dest, voice, rate):
        raise RuntimeError("nope")

    monkeypatch.setattr(vo, "available_engines", lambda: ["edge", "espeak"])
    monkeypatch.setattr(vo, "_SYNTHS", {"edge": fail, "espeak": fail})
    with pytest.raises(VoiceoverError, match="every text-to-speech engine failed"):
        synthesize("hello", tmp_path / "vo.wav")


def test_no_engine_at_all_explains_how_to_install_one(tmp_path, monkeypatch):
    monkeypatch.setattr(vo, "available_engines", lambda: [])
    with pytest.raises(VoiceoverError, match="edge-tts"):
        synthesize("hello", tmp_path / "vo.wav")


def test_sapi_passes_text_via_a_file_not_the_command_line(tmp_path, monkeypatch):
    """Quotes in narration must never reach PowerShell as syntax."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return None

    monkeypatch.setattr(vo, "run", fake_run)
    monkeypatch.setattr(vo.shutil, "which", lambda name: "powershell" if "shell" in name else None)

    nasty = 'He said "stop"; rm -rf /'
    vo._synth_sapi(nasty, tmp_path / "vo.wav", None, 0)

    script = captured["cmd"][-1]
    assert nasty not in script
    assert "ReadAllText" in script


@pytest.mark.skipif(
    not (shutil.which("espeak-ng") or shutil.which("espeak")) or not shutil.which("ffmpeg"),
    reason="espeak and ffmpeg required",
)
def test_espeak_produces_real_audio(tmp_path):
    from clipping.media import audio_duration

    dest = synthesize(
        "Isaiah Rashad sang his heart out on this tiny concert performance.",
        tmp_path / "vo.wav",
        engine="espeak",
    )
    assert dest.exists()
    # A sentence of that length should be a few seconds, not silence.
    assert 1.5 < audio_duration(dest) < 12.0
