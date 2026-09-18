"""Thin wrappers around ffmpeg/ffprobe.

Everything that shells out lives here so the rest of the package stays pure
Python and testable without media tooling installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class MediaError(RuntimeError):
    """Raised when ffmpeg/ffprobe is missing or fails."""


@dataclass
class VideoInfo:
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise MediaError(
            f"{name!r} was not found on PATH. Install FFmpeg "
            "(https://ffmpeg.org/download.html) and try again."
        )
    return path


def run(
    cmd: list[str], *, quiet: bool = True, cwd: str | Path | None = None
) -> subprocess.CompletedProcess:
    """Run a command, raising MediaError with the tail of stderr on failure."""
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd) if cwd else None,
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-15:])
        raise MediaError(f"command failed: {' '.join(cmd[:3])} ...\n{tail}")
    return proc


def probe(path: str | Path) -> VideoInfo:
    require_tool("ffprobe")
    proc = run(
        [
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ]
    )
    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError(f"no video stream found in {path}")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0.0)
    return VideoInfo(
        duration=duration,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=_parse_fps(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        has_audio=has_audio,
    )


def _parse_fps(rate: str | None) -> float:
    if not rate:
        return 0.0
    if "/" in rate:
        num, _, den = rate.partition("/")
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        except ValueError:
            return 0.0
    try:
        return float(rate)
    except ValueError:
        return 0.0


def extract_audio(src: str | Path, dest: str | Path, sample_rate: int = 16000) -> Path:
    """Decode to the mono 16 kHz WAV that Whisper wants."""
    require_tool("ffmpeg")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            str(dest),
        ]
    )
    return dest


def download(url: str, dest_dir: str | Path) -> Path:
    """Fetch a source video with yt-dlp (optional dependency)."""
    require_tool("yt-dlp")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    template = str(dest_dir / "source.%(ext)s")
    run(
        [
            "yt-dlp",
            "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
            "--merge-output-format", "mp4",
            "-o", template,
            url,
        ]
    )
    files = sorted(dest_dir.glob("source.*"))
    if not files:
        raise MediaError(f"yt-dlp produced no output for {url}")
    return files[0]
