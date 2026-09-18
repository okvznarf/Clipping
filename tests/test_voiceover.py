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
        def _f(text, dest, voice, rate, pitch):
            tried.append(name)
            raise RuntimeError(f"{name} unavailable")

        return _f

    def succeed(text, dest, voice, rate, pitch):
        tried.append("espeak")
        dest.write_bytes(b"\x00" * 4096)
        return dest

    monkeypatch.setattr(vo, "available_engines", lambda: ["edge", "sapi", "espeak"])
    monkeypatch.setattr(vo, "_SYNTHS", {"edge": fail("edge"), "sapi": fail("sapi"), "espeak": succeed})
    monkeypatch.setattr(vo, "_to_wav", lambda raw, dest: dest)

    synthesize("hello", tmp_path / "vo.wav")
    assert tried == ["edge", "sapi", "espeak"]


def test_all_engines_failing_reports_each_reason(tmp_path, monkeypatch):
    def fail(text, dest, voice, rate, pitch):
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
    vo._synth_sapi(nasty, tmp_path / "vo.wav", None, 0, 0)

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


def test_styles_set_rate_and_pitch():
    from clipping.voiceover import resolve_style

    assert resolve_style("energetic", None, None) == (12, 10)
    assert resolve_style("natural", None, None) == (0, 0)
    calm_rate, calm_pitch = resolve_style("calm", None, None)
    assert calm_rate < 0 and calm_pitch < 0


def test_explicit_rate_and_pitch_override_the_style():
    from clipping.voiceover import resolve_style

    assert resolve_style("energetic", 40, None) == (40, 10)
    assert resolve_style("energetic", None, 0) == (12, 0)
    # An override of zero is honoured, not treated as "unset".
    assert resolve_style("energetic", 0, 0) == (0, 0)


def test_unknown_style_is_rejected():
    from clipping.voiceover import resolve_style

    with pytest.raises(VoiceoverError, match="unknown style"):
        resolve_style("shouty", None, None)


def test_the_default_style_has_lift():
    """A short-form hook should not default to a flat read."""
    rate, pitch = vo.resolve_style(vo.DEFAULT_STYLE, None, None)
    assert rate > 0 and pitch > 0


def test_empty_engine_output_is_treated_as_failure(tmp_path, monkeypatch):
    """Regression: edge-tts exits 0 but writes 0 bytes when its service says no."""
    def writes_nothing(text, dest, voice, rate, pitch):
        dest.write_bytes(b"")
        return dest

    def works(text, dest, voice, rate, pitch):
        dest.write_bytes(b"\x00" * 4096)
        return dest

    monkeypatch.setattr(vo, "available_engines", lambda: ["edge", "espeak"])
    monkeypatch.setattr(vo, "_SYNTHS", {"edge": writes_nothing, "espeak": works})
    monkeypatch.setattr(vo, "_to_wav", lambda raw, dest: dest)

    assert synthesize("hello", tmp_path / "vo.wav").exists()


def test_all_engines_producing_nothing_is_an_error(tmp_path, monkeypatch):
    def writes_nothing(text, dest, voice, rate, pitch):
        dest.write_bytes(b"")
        return dest

    monkeypatch.setattr(vo, "available_engines", lambda: ["edge"])
    monkeypatch.setattr(vo, "_SYNTHS", {"edge": writes_nothing})
    with pytest.raises(VoiceoverError, match="produced no audio"):
        synthesize("hello", tmp_path / "vo.wav")


def test_ssml_escapes_the_narration():
    from clipping.voiceover import _ssml

    out = _ssml('He said "go" & <ran>', 10, 5)
    assert "&amp;" in out and "&lt;ran&gt;" in out
    assert 'rate="+10%"' in out and 'pitch="+5%"' in out
