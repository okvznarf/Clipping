"""Loudness envelope of a track, used to find the sections with energy.

Speech clips are chosen on what is said; music clips also depend on where the
track actually goes somewhere — the drop, the full-band chorus, the point the
drums come in. That is visible in a coarse RMS envelope, which is cheap to get.
"""

from __future__ import annotations

import array
import math
import subprocess
from pathlib import Path

from .media import require_tool

#: Sample rate to decode at. The envelope only needs gross loudness, so this is
#: deliberately low — it keeps a 20 minute track to a few MB of PCM.
ENVELOPE_RATE = 8000


def rms_envelope(pcm: bytes, *, sample_rate: int = ENVELOPE_RATE, hop: float = 1.0) -> list[float]:
    """RMS of signed 16-bit mono PCM, one value per ``hop`` seconds."""
    if hop <= 0:
        raise ValueError("hop must be positive")

    samples = array.array("h")
    usable = len(pcm) - (len(pcm) % 2)
    samples.frombytes(pcm[:usable])

    window = max(1, int(sample_rate * hop))
    envelope: list[float] = []
    for start in range(0, len(samples), window):
        chunk = samples[start : start + window]
        if not chunk:
            break
        total = sum(float(s) * float(s) for s in chunk)
        envelope.append(math.sqrt(total / len(chunk)) / 32768.0)
    return envelope


def decode_pcm(source: str | Path, *, sample_rate: int = ENVELOPE_RATE) -> bytes:
    """Decode any media file to raw mono 16-bit PCM on stdout."""
    require_tool("ffmpeg")
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-i", str(source),
            "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-f", "s16le", "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-5:]
        raise RuntimeError("ffmpeg could not decode audio:\n" + "\n".join(tail))
    return proc.stdout


class Energy:
    """A loudness envelope that can be queried over a time span."""

    def __init__(self, values: list[float], hop: float = 1.0):
        self.values = values
        self.hop = hop
        self.reference = _percentile(values, 0.9) or max(values, default=0.0)

    @classmethod
    def from_file(cls, source: str | Path, *, hop: float = 1.0) -> "Energy":
        return cls(rms_envelope(decode_pcm(source), hop=hop), hop=hop)

    @classmethod
    def flat(cls) -> "Energy":
        """A neutral envelope, for when no audio is available."""
        return cls([], 1.0)

    def mean(self, start: float, end: float) -> float:
        """Mean loudness over a span, relative to the track's loud sections.

        1.0 means 'as loud as this track usually gets at its peak'; a flat or
        unavailable envelope returns a neutral 0.5 so it cannot skew a ranking.
        """
        if not self.values or self.reference <= 0:
            return 0.5
        lo = max(0, int(start / self.hop))
        hi = min(len(self.values), max(lo + 1, int(math.ceil(end / self.hop))))
        window = self.values[lo:hi]
        if not window:
            return 0.5
        return min(1.0, (sum(window) / len(window)) / self.reference)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]
