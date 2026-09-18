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

DEFAULT_EDGE_VOICE = "en-US-AndrewNeural"
#: Words per minute the CLI engines use when no rate is given.
BASE_WPM = 175


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


def _synth_edge(text: str, dest: Path, voice: str | None, rate_percent: int) -> Path:
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
    run(cmd)
    return raw


def _synth_sapi(text: str, dest: Path, voice: str | None, rate_percent: int) -> Path:
    raw = dest.with_suffix(".sapi.wav")
    # The text goes via a file so that quotes and apostrophes in it can never
    # be parsed as PowerShell syntax.
    text_file = dest.with_suffix(".txt")
    text_file.write_text(text, encoding="utf-8")
    select = f'$s.SelectVoice("{voice}");' if voice else ""
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$s.Rate = {max(-10, min(10, round(rate_percent / 10)))};"
        f"{select}"
        f'$s.SetOutputToWaveFile("{raw}");'
        f'$s.Speak([IO.File]::ReadAllText("{text_file}"));'
        "$s.Dispose();"
    )
    shell = shutil.which("powershell") or shutil.which("pwsh")
    run([shell, "-NoProfile", "-NonInteractive", "-Command", script])
    text_file.unlink(missing_ok=True)
    return raw


def _synth_say(text: str, dest: Path, voice: str | None, rate_percent: int) -> Path:
    raw = dest.with_suffix(".aiff")
    cmd = ["say", "-o", str(raw), "-r", str(_wpm(rate_percent))]
    if voice:
        cmd += ["-v", voice]
    cmd += ["--", text]
    run(cmd)
    return raw


def _synth_espeak(text: str, dest: Path, voice: str | None, rate_percent: int) -> Path:
    raw = dest.with_suffix(".espeak.wav")
    binary = shutil.which("espeak-ng") or shutil.which("espeak")
    cmd = [binary, "-s", str(_wpm(rate_percent)), "-w", str(raw)]
    cmd += ["-v", voice or "en-us+m3"]
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
    rate_percent: int = 0,
) -> Path:
    """Speak ``text`` into ``dest`` as a WAV. Returns the path written."""
    text = text.strip()
    if not text:
        raise VoiceoverError("voiceover text is empty")

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

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
            raw = _SYNTHS[name](text, dest, voice, rate_percent)
        except Exception as exc:  # noqa: BLE001 - try the next engine
            errors.append(f"{name}: {exc}")
            continue
        return _to_wav(raw, dest)

    raise VoiceoverError("every text-to-speech engine failed:\n  " + "\n  ".join(errors))
