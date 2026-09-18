"""End-to-end pipeline: source video -> ranked, captioned vertical clips."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from . import captions as caps
from . import media, render, scoring, segmenter
from .captions import CaptionStyle
from .segmenter import Candidate
from .transcript import Transcript, shift


@dataclass
class ClipSpec:
    """One chosen clip and everything written out for it."""

    index: int
    start: float
    end: float
    score: float
    title: str
    text: str
    reasons: list[str] = field(default_factory=list)
    video_path: str | None = None
    subtitle_path: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Options:
    output_dir: Path = Path("out")
    clips: int = 5
    min_duration: float = 15.0
    max_duration: float = 60.0
    ideal_duration: float = 32.0
    padding: float = 0.25
    layout: str = "crop"
    width: int = 1080
    height: int = 1920
    fps: int | None = 30
    crf: int = 20
    preset: str = "veryfast"
    normalize_audio: bool = True
    max_overlap: float = 0.25
    style: CaptionStyle = field(default_factory=CaptionStyle)
    ranker: str = "heuristic"   # or "claude"
    llm_weight: float = 0.6
    llm_model: str = "claude-sonnet-5"
    voiceover: str | None = None
    voiceover_engine: str = "auto"
    voiceover_voice: str | None = None
    voiceover_style: str = "energetic"
    voiceover_rate: int | None = None
    voiceover_pitch: int | None = None
    voiceover_delay: float = 0.3
    duck: float = 0.35
    render_video: bool = True
    keep_ass: bool = True
    verbose: bool = True


def _log(options: Options, message: str) -> None:
    if options.verbose:
        print(message, flush=True)


def plan_clips(transcript: Transcript, options: Options) -> list[Candidate]:
    """Everything up to encoding: split, score, rank, pick."""
    sentences = segmenter.split_sentences(transcript)
    _log(options, f"→ {len(sentences)} sentences from {len(transcript)} words")

    candidates = segmenter.build_candidates(
        sentences,
        min_duration=options.min_duration,
        max_duration=options.max_duration,
    )
    if not candidates:
        return []
    _log(options, f"→ {len(candidates)} candidate windows")

    scoring.score_all(candidates, ideal_duration=options.ideal_duration)

    if options.ranker == "claude":
        from . import ranker_llm

        try:
            _log(options, f"→ re-ranking with {options.llm_model}")
            ranker_llm.rerank(
                candidates, model=options.llm_model, weight=options.llm_weight
            )
        except Exception as exc:  # noqa: BLE001 - ranking is best effort
            _log(options, f"! LLM ranking unavailable ({exc}); using heuristic scores")

    return segmenter.select(
        candidates, count=options.clips, max_overlap=options.max_overlap
    )


def process(
    source: str | Path,
    options: Options | None = None,
    *,
    transcript: Transcript | None = None,
    transcribe_kwargs: dict | None = None,
) -> list[ClipSpec]:
    """Run the whole pipeline and return the clips that were produced."""
    options = options or Options()
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(source)

    out_dir = Path(options.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = media.probe(source) if options.render_video else None
    if info:
        _log(
            options,
            f"→ source {info.width}x{info.height} @ {info.fps:.2f}fps, "
            f"{info.duration / 60:.1f} min",
        )

    if transcript is None:
        from .transcribe import transcribe as run_transcribe

        started = time.time()
        _log(options, "→ transcribing (this is the slow part)")
        transcript = run_transcribe(source, workdir=out_dir, **(transcribe_kwargs or {}))
        _log(options, f"→ transcribed in {time.time() - started:.0f}s")
        transcript.save(out_dir / "transcript.json")

    chosen = plan_clips(transcript, options)
    if not chosen:
        _log(options, "! no candidate windows matched the duration range")
        return []

    # One narration file serves every clip in the run.
    narration = None
    if options.voiceover and options.render_video:
        from .voiceover import synthesize

        _log(options, f"→ narrating: {options.voiceover!r}")
        narration = synthesize(
            options.voiceover,
            out_dir / "voiceover.wav",
            engine=options.voiceover_engine,
            voice=options.voiceover_voice,
            style=options.voiceover_style,
            rate_percent=options.voiceover_rate,
            pitch_percent=options.voiceover_pitch,
        )

    source_duration = info.duration if info else transcript.duration

    # 'original' keeps the source framing, so captions must be laid out for the
    # source resolution rather than the (unused) target one.
    width, height = options.width, options.height
    if options.layout == "original" and info and info.width and info.height:
        width, height = info.width, info.height

    specs: list[ClipSpec] = []

    for index, cand in enumerate(chosen, start=1):
        start = max(0.0, cand.start - options.padding)
        end = cand.end + options.padding
        if source_duration:
            end = min(end, source_duration)

        title = _title_for(cand)
        stem = f"clip_{index:02d}"
        ass_path = out_dir / f"{stem}.ass"
        clip_words = shift(cand.words, start)

        ass_path.write_text(
            caps.build_ass(clip_words, options.style, width=width, height=height),
            encoding="utf-8",
        )
        (out_dir / f"{stem}.srt").write_text(
            caps.build_srt(clip_words, options.style, width=width),
            encoding="utf-8",
        )

        spec = ClipSpec(
            index=index,
            start=round(start, 3),
            end=round(end, 3),
            score=cand.score,
            title=title,
            text=cand.text,
            reasons=[r for r in cand.reasons if not r.startswith("title::")],
            subtitle_path=str(ass_path),
        )

        if options.render_video:
            dest = out_dir / f"{stem}.mp4"
            _log(
                options,
                f"→ [{index}/{len(chosen)}] {start:7.1f}s +{end - start:4.1f}s "
                f"score {cand.score:.3f}  {title[:48]}",
            )
            render.render_clip(
                source,
                dest,
                start=start,
                duration=end - start,
                layout=options.layout,
                width=width,
                height=height,
                subtitles=ass_path,
                crf=options.crf,
                preset=options.preset,
                fps=options.fps,
                normalize_audio=options.normalize_audio,
                has_audio=bool(info.has_audio) if info else True,
                voiceover=narration,
                voiceover_delay=options.voiceover_delay,
                duck=options.duck,
            )
            spec.video_path = str(dest)

        if not options.keep_ass:
            ass_path.unlink(missing_ok=True)
            spec.subtitle_path = None

        specs.append(spec)

    manifest = out_dir / "clips.json"
    manifest.write_text(
        json.dumps({"source": str(source), "clips": [asdict(s) for s in specs]}, indent=2),
        encoding="utf-8",
    )
    _log(options, f"→ wrote {len(specs)} clips and {manifest}")
    return specs


def _title_for(candidate: Candidate) -> str:
    from .ranker_llm import title_from

    return title_from(candidate) or scoring.suggest_title(candidate.text)
