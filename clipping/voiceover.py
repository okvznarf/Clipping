"""Text-to-speech narration, for a spoken hook over the start of a clip.

There is no portable TTS engine, so this tries what a machine is likely to
have, best sounding first:

  edge  — the ``edge-tts`` package: neural voices, free, needs a network call
  sapi  — Windows' built-in speech synthesiser, via PowerShell, no install
  say   — macOS' built-in ``say``
  espeak— ``espeak-ng``: robotic, but present on most Linux boxes

Whatever produces the audio, the result is normalised to a mono 48 kHz WAV so
the render step never has to care which engine ran.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .media import require_tool, run

ENGINES = ("edge", "sapi", "say", "espeak")

DEFAULT_EDGE_VOICE = "en-US-AndrewMultilingualNeural"
#: Words per minute the CLI engines use when no rate is given.
BASE_WPM = 175
#: Anything smaller than this is not audio, whatever the engine's exit code said.
MIN_AUDIO_BYTES = 1024

#: Delivery presets. A short-form hook wants a bit of lift and pace, so
#: "energetic" is the default; the others are there when narration should sit
#: back instead. rate and pitch are percentages relative to the voice's normal.
STYLES: dict[str, dict[str, int]] = {
    "energetic": {"rate": 12, "pitch": 10},
    "natural": {"rate": 0, "pitch": 0},
    "calm": {"rate": -8, "pitch": -5},
}
DEFAULT_STYLE = "energetic"

#: Voices that carry a lift well, per engine, when the caller names none.
ENERGETIC_VOICES = {
    "edge": "en-US-AndrewMultilingualNeural",
    "sapi": None,       # whatever the system default is; SSML supplies the lift
    "say": None,
    "espeak": "en-us+m3",
}


def resolve_style(style: str, rate: int | None, pitch: int | None) -> tuple[int, int]:
    """Turn a style name plus optional overrides into (rate%, pitch%)."""
    if style not in STYLES:
        raise VoiceoverError(
            f"unknown style {style!r}; expected one of {tuple(STYLES)}"
        )
    preset = STYLES[style]
    return (
        preset["rate"] if rate is None else rate,
        preset["pitch"] if pitch is None else pitch,
    )


class VoiceoverError(RuntimeError):
    pass


def _edge_available() -> bool:
    if shutil.which("edge-tts"):
        return True
    try:  # the package may be importable without its script on PATH
        import edge_tts  # noqa: F401

        return True
    except ImportError:
        return False


def available_engines() -> list[str]:
    """Engines usable on this machine, in preference order."""
    found = []
    if _edge_available():
        found.append("edge")
    if sys.platform == "win32" and (shutil.which("powershell") or shutil.which("pwsh")):
        found.append("sapi")
    if sys.platform == "darwin" and shutil.which("say"):
        found.append("say")
    if shutil.which("espeak-ng") or shutil.which("espeak"):
        found.append("espeak")
    return found


def _wpm(rate_percent: int) -> int:
    return max(80, min(400, int(BASE_WPM * (1 + rate_percent / 100))))


def _synth_edge(
    text: str, dest: Path, voice: str | None, rate_percent: int, pitch_percent: int
) -> Path:
    raw = dest.with_suffix(".mp3")
    cmd = [
        shutil.which("edge-tts") or sys.executable,
        *([] if shutil.which("edge-tts") else ["-m", "edge_tts"]),
        "--voice", voice or DEFAULT_EDGE_VOICE,
        "--text", text,
        "--write-media", str(raw),
    ]
    if rate_percent:
        cmd += ["--rate", f"{rate_percent:+d}%"]
    if pitch_percent:
        # edge-tts takes pitch in Hz; roughly 1 Hz per percent reads naturally
        # over the small range a narration hook needs.
        cmd += ["--pitch", f"{pitch_percent:+d}Hz"]
    run(cmd)
    return raw


def _synth_sapi(
    text: str, dest: Path, voice: str | None, rate_percent: int, pitch_percent: int = 0
) -> Path:
    raw = dest.with_suffix(".sapi.wav")
    # The text goes via a file so that quotes and apostrophes in it can never
    # be parsed as PowerShell syntax.
    text_file = dest.with_suffix(".txt")
    text_file.write_text(text, encoding="utf-8")
    select = f'$s.SelectVoice("{voice}");' if voice else ""

    if pitch_percent:
        # SpeechSynthesizer has no pitch property, so prosody has to go through
        # SSML. The text is XML-escaped into the same side file.
        text_file.write_text(_ssml(text, rate_percent, pitch_percent), encoding="utf-8")
        speak = f'$s.SpeakSsml([IO.File]::ReadAllText("{text_file}"));'
        rate = ""
    else:
        speak = f'$s.Speak([IO.File]::ReadAllText("{text_file}"));'
        rate = f"$s.Rate = {max(-10, min(10, round(rate_percent / 10)))};"

    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"{rate}"
        f"{select}"
        f'$s.SetOutputToWaveFile("{raw}");'
        f"{speak}"
        "$s.Dispose();"
    )
    shell = shutil.which("powershell") or shutil.which("pwsh")
    run([shell, "-NoProfile", "-NonInteractive", "-Command", script])
    text_file.unlink(missing_ok=True)
    return raw


def _ssml(text: str, rate_percent: int, pitch_percent: int) -> str:
    from xml.sax.saxutils import escape

    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        'xml:lang="en-US">'
        f'<prosody rate="{rate_percent:+d}%" pitch="{pitch_percent:+d}%">'
        f"{escape(text)}"
        "</prosody></speak>"
    )


def _synth_say(
    text: str, dest: Path, voice: str | None, rate_percent: int, pitch_percent: int = 0
) -> Path:
    raw = dest.with_suffix(".aiff")
    cmd = ["say", "-o", str(raw), "-r", str(_wpm(rate_percent))]
    if voice:
        cmd += ["-v", voice]
    cmd += ["--", text]
    run(cmd)
    return raw


def _synth_espeak(
    text: str, dest: Path, voice: str | None, rate_percent: int, pitch_percent: int = 0
) -> Path:
    raw = dest.with_suffix(".espeak.wav")
    binary = shutil.which("espeak-ng") or shutil.which("espeak")
    # espeak pitch is 0-99 around a default of 50.
    pitch = max(0, min(99, 50 + pitch_percent))
    cmd = [binary, "-s", str(_wpm(rate_percent)), "-p", str(pitch), "-w", str(raw)]
    cmd += ["-v", voice or ENERGETIC_VOICES["espeak"]]
    # Read from stdin so the text is never part of the command line.
    proc = subprocess.run(cmd + ["--stdin"], input=text, text=True, capture_output=True)
    if proc.returncode != 0:
        raise VoiceoverError(f"espeak failed: {proc.stderr.strip()[:200]}")
    return raw


_SYNTHS = {
    "edge": _synth_edge,
    "sapi": _synth_sapi,
    "say": _synth_say,
    "espeak": _synth_espeak,
}


def _to_wav(raw: Path, dest: Path) -> Path:
    """Normalise any engine's output to mono 48 kHz WAV at a consistent level."""
    require_tool("ffmpeg")
    run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(raw),
            "-ac", "1", "-ar", "48000",
            # Narration has to sit clearly above a music bed, so level it.
            "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
            "-c:a", "pcm_s16le",
            str(dest),
        ]
    )
    if raw != dest:
        raw.unlink(missing_ok=True)
    return dest


def synthesize(
    text: str,
    dest: str | Path,
    *,
    engine: str = "auto",
    voice: str | None = None,
    style: str = DEFAULT_STYLE,
    rate_percent: int | None = None,
    pitch_percent: int | None = None,
) -> Path:
    """Speak ``text`` into ``dest`` as a WAV. Returns the path written.

    ``style`` sets the delivery; ``rate_percent`` and ``pitch_percent`` override
    it when given.
    """
    text = text.strip()
    if not text:
        raise VoiceoverError("voiceover text is empty")

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rate, pitch = resolve_style(style, rate_percent, pitch_percent)

    if engine == "auto":
        candidates = available_engines()
        if not candidates:
            raise VoiceoverError(
                "no text-to-speech engine found. Install one of: "
                "pip install edge-tts (best quality), or espeak-ng on Linux. "
                "Windows and macOS have one built in."
            )
    elif engine in _SYNTHS:
        candidates = [engine]
    else:
        raise VoiceoverError(f"unknown engine {engine!r}; expected one of {ENGINES}")

    errors: list[str] = []
    for name in candidates:
        try:
            raw = _SYNTHS[name](text, dest, voice or ENERGETIC_VOICES.get(name), rate, pitch)
            # An engine can fail while still exiting 0 — edge-tts leaves an empty
            # file behind when its service refuses the connection.
            if not raw.exists() or raw.stat().st_size < MIN_AUDIO_BYTES:
                raise VoiceoverError("produced no audio")
            return _to_wav(raw, dest)
        except Exception as exc:  # noqa: BLE001 - try the next engine
            errors.append(f"{name}: {exc}")
            continue

    raise VoiceoverError("every text-to-speech engine failed:\n  " + "\n  ".join(errors))
