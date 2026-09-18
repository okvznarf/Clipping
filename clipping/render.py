"""Cut, reframe and burn captions into a clip with a single ffmpeg pass."""

from __future__ import annotations

from pathlib import Path

from .media import require_tool, run

LAYOUTS = ("crop", "blur", "fit", "original")


def escape_filter_path(path: str | Path) -> str:
    """Escape a path for use inside a quoted ffmpeg filtergraph argument.

    ffmpeg parses the filtergraph before the filter sees its options, so a colon
    or a quote in the name has to survive both passes. render_clip sidesteps the
    worst of it by running ffmpeg from the subtitle's own directory and passing
    only the basename.
    """
    text = str(path).replace("\\", "/")
    text = text.replace("'", r"\'")
    text = text.replace(":", r"\:")
    return text


def build_filter(
    layout: str,
    width: int,
    height: int,
    *,
    subtitles: str | Path | None = None,
    blur_sigma: int = 28,
) -> str:
    """Build the filtergraph that reframes the source and burns in captions."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")

    subs = f"subtitles='{escape_filter_path(subtitles)}'" if subtitles else None

    if layout == "original":
        stages = ["null"]
    elif layout == "crop":
        stages = [
            f"scale={width}:{height}:force_original_aspect_ratio=increase",
            f"crop={width}:{height}",
        ]
    elif layout == "fit":
        stages = [
            f"scale={width}:{height}:force_original_aspect_ratio=decrease",
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black",
        ]
    else:  # blur — fitted foreground over a blurred fill of the same frame
        graph = (
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},gblur=sigma={blur_sigma}[bgb];"
            f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2"
        )
        graph += ",setsar=1"
        if subs:
            graph += f",{subs}"
        return graph + "[v]"

    graph = "[0:v]" + ",".join(stages) + ",setsar=1"
    if subs:
        graph += f",{subs}"
    return graph + "[v]"


def render_clip(
    source: str | Path,
    dest: str | Path,
    *,
    start: float,
    duration: float,
    layout: str = "crop",
    width: int = 1080,
    height: int = 1920,
    subtitles: str | Path | None = None,
    crf: int = 20,
    preset: str = "veryfast",
    fps: int | None = 30,
    normalize_audio: bool = True,
    has_audio: bool = True,
) -> Path:
    """Encode one clip. Returns the written path."""
    require_tool("ffmpeg")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Run from the subtitle's directory and reference it by name: it keeps the
    # filtergraph free of absolute paths, which are the source of most
    # subtitles= escaping problems.
    cwd = None
    sub_arg = None
    if subtitles is not None:
        subtitles = Path(subtitles).resolve()
        cwd, sub_arg = subtitles.parent, subtitles.name
    source = Path(source).resolve()
    dest = dest.resolve()

    graph = build_filter(layout, width, height, subtitles=sub_arg)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, start):.3f}",
        "-t", f"{max(0.1, duration):.3f}",
        "-i", str(source),
        "-filter_complex", graph,
        "-map", "[v]",
    ]
    if has_audio:
        cmd += ["-map", "0:a?"]
        if normalize_audio:
            cmd += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]
        cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    else:
        cmd += ["-an"]
    if fps:
        cmd += ["-r", str(fps)]
    cmd += [
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-movflags", "+faststart",
        str(dest),
    ]
    run(cmd, cwd=cwd)
    return dest
