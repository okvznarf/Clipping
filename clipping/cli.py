"""Command-line interface: ``python -m clipping <video> [options]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .captions import CaptionStyle
from .media import MediaError, download
from .pipeline import Options, process
from .transcript import Transcript

PRESETS: dict[str, dict] = {
    "bold": {},  # the defaults
    "clean": {
        "font": "Helvetica",
        "font_size": 72,
        "highlight": "#00E5FF",
        "uppercase": False,
        "outline_width": 3.5,
        "pop_scale": 106,
        "max_words_per_line": 5,
    },
    "minimal": {
        "font": "Helvetica",
        "font_size": 64,
        "highlight": "#FFFFFF",
        "uppercase": False,
        "outline_width": 2.5,
        "pop_scale": 100,
        "fade_in_ms": 0,
        "max_words_per_line": 6,
        "position": 0.85,
    },
    "hormozi": {
        "font": "Impact",
        "font_size": 96,
        "highlight": "#39FF14",
        "outline_width": 6.0,
        "pop_scale": 118,
        "max_words_per_line": 3,
        "max_chars_per_line": 16,
        "position": 0.62,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipping",
        description="Turn a long video into short, captioned vertical clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source", help="path to a video/audio file, or a URL (needs yt-dlp)")
    parser.add_argument("-o", "--output", default="out", help="output directory")
    parser.add_argument("-n", "--clips", type=int, default=5, help="how many clips to produce")
    parser.add_argument("--version", action="version", version=f"clipping {__version__}")

    timing = parser.add_argument_group("clip selection")
    timing.add_argument("--min-duration", type=float, default=15.0)
    timing.add_argument("--max-duration", type=float, default=60.0)
    timing.add_argument("--ideal-duration", type=float, default=32.0,
                        help="length the scorer treats as ideal")
    timing.add_argument("--padding", type=float, default=0.25,
                        help="seconds of breathing room added to each end")
    timing.add_argument("--max-overlap", type=float, default=0.25,
                        help="max overlap allowed between two selected clips (0-1)")
    timing.add_argument("--ranker", choices=("heuristic", "claude"), default="heuristic",
                        help="'claude' re-ranks with the Anthropic API (needs ANTHROPIC_API_KEY)")
    timing.add_argument("--llm-weight", type=float, default=0.6)
    timing.add_argument("--llm-model", default="claude-sonnet-5")

    speech = parser.add_argument_group("transcription")
    speech.add_argument("--model", default="small",
                        help="faster-whisper model: tiny/base/small/medium/large-v3")
    speech.add_argument("--language", default=None, help="force a language code, e.g. en")
    speech.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    speech.add_argument("--compute-type", default="default",
                        help="faster-whisper compute type, e.g. int8, float16")
    speech.add_argument("--no-vad", action="store_true", help="disable voice-activity filtering")
    speech.add_argument("--transcript", default=None,
                        help="reuse a transcript.json instead of transcribing again")

    look = parser.add_argument_group("look")
    look.add_argument("--layout", choices=("crop", "blur", "fit", "original"), default="crop")
    look.add_argument("--size", default="1080x1920", help="output resolution, WxH")
    look.add_argument("--fps", type=int, default=30)
    look.add_argument("--crf", type=int, default=20, help="x264 quality, lower is better")
    look.add_argument("--preset", default="veryfast", help="x264 speed preset")
    look.add_argument("--no-normalize", action="store_true", help="skip loudness normalisation")

    caption = parser.add_argument_group("captions")
    caption.add_argument("--caption-preset", choices=tuple(PRESETS), default="bold")
    caption.add_argument("--font", default=None)
    caption.add_argument("--font-size", type=int, default=None)
    caption.add_argument("--highlight", default=None, help="active-word colour, #RRGGBB")
    caption.add_argument("--text-colour", "--text-color", dest="text_colour", default=None)
    caption.add_argument("--words-per-line", type=int, default=None)
    caption.add_argument("--caption-position", type=float, default=None,
                         help="vertical placement, 0 (top) to 1 (bottom)")
    caption.add_argument("--no-uppercase", action="store_true")

    voice = parser.add_argument_group("voiceover")
    voice.add_argument("--voiceover", default=None,
                       help="speak this line over the start of each clip")
    voice.add_argument("--voiceover-engine", default="auto",
                       choices=("auto", "edge", "sapi", "say", "espeak"))
    voice.add_argument("--voiceover-voice", default=None,
                       help="engine-specific voice name, e.g. en-US-AriaNeural")
    voice.add_argument("--voiceover-rate", type=int, default=0,
                       help="speech rate change, in percent")
    voice.add_argument("--voiceover-delay", type=float, default=0.3,
                       help="seconds before the narration starts")
    voice.add_argument("--duck", type=float, default=0.35,
                       help="clip volume under the narration (0-1)")

    modes = parser.add_argument_group("modes")
    modes.add_argument("--dry-run", action="store_true",
                       help="pick clips and write captions, but don't encode video")
    modes.add_argument("-q", "--quiet", action="store_true")
    return parser


def style_from_args(args: argparse.Namespace) -> CaptionStyle:
    style = CaptionStyle(**PRESETS[args.caption_preset])
    if args.font:
        style.font = args.font
    if args.font_size:
        style.font_size = args.font_size
    if args.highlight:
        style.highlight = args.highlight
    if args.text_colour:
        style.primary = args.text_colour
    if args.words_per_line:
        style.max_words_per_line = args.words_per_line
        style.max_chars_per_line = max(style.max_chars_per_line, args.words_per_line * 7)
    if args.caption_position is not None:
        style.position = args.caption_position
    if args.no_uppercase:
        style.uppercase = False
    style.__post_init__()  # re-validate after the overrides
    return style


def parse_size(value: str) -> tuple[int, int]:
    match = value.lower().replace("×", "x").split("x")
    if len(match) != 2:
        raise ValueError(f"expected WxH, got {value!r}")
    width, height = (int(part) for part in match)
    if width <= 0 or height <= 0:
        raise ValueError(f"resolution must be positive, got {value!r}")
    return width, height


def options_from_args(args: argparse.Namespace) -> Options:
    width, height = parse_size(args.size)
    return Options(
        output_dir=Path(args.output),
        clips=args.clips,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        ideal_duration=args.ideal_duration,
        padding=args.padding,
        layout=args.layout,
        width=width,
        height=height,
        fps=args.fps,
        crf=args.crf,
        preset=args.preset,
        normalize_audio=not args.no_normalize,
        max_overlap=args.max_overlap,
        style=style_from_args(args),
        ranker=args.ranker,
        llm_weight=args.llm_weight,
        llm_model=args.llm_model,
        voiceover=args.voiceover,
        voiceover_engine=args.voiceover_engine,
        voiceover_voice=args.voiceover_voice,
        voiceover_rate=args.voiceover_rate,
        voiceover_delay=args.voiceover_delay,
        duck=args.duck,
        render_video=not args.dry_run,
        verbose=not args.quiet,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        options = options_from_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    options.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        source = args.source
        if source.startswith(("http://", "https://")):
            if not args.quiet:
                print(f"→ downloading {source}")
            source = download(source, options.output_dir / "source")

        transcript = Transcript.load(args.transcript) if args.transcript else None
        specs = process(
            source,
            options,
            transcript=transcript,
            transcribe_kwargs={
                "model_size": args.model,
                "language": args.language,
                "device": args.device,
                "compute_type": args.compute_type,
                "vad_filter": not args.no_vad,
                "progress": not args.quiet,
            },
        )
    except (MediaError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    if not specs:
        print("No clips were produced. Try lowering --min-duration.", file=sys.stderr)
        return 1

    if not args.quiet:
        print("\nClips:")
        for spec in specs:
            print(
                f"  {spec.index:2d}. {spec.start:8.1f}s  {spec.duration:5.1f}s  "
                f"score {spec.score:.3f}  {spec.title}"
            )
    return 0
