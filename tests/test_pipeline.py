import json
from pathlib import Path

import pytest

from clipping.captions import CaptionStyle
from clipping.pipeline import Options, plan_clips, process
from clipping.segmenter import Candidate


@pytest.fixture
def options(tmp_path) -> Options:
    return Options(
        output_dir=tmp_path / "out",
        clips=2,
        min_duration=4,
        max_duration=15,
        ideal_duration=8,
        render_video=False,
        verbose=False,
    )


def test_plan_respects_the_clip_count(sample_transcript, options):
    chosen = plan_clips(sample_transcript, options)
    assert 0 < len(chosen) <= options.clips
    assert all(options.min_duration <= c.duration <= options.max_duration for c in chosen)


def test_plan_returns_nothing_when_nothing_fits(sample_transcript, options):
    options.min_duration = 600
    options.max_duration = 900
    assert plan_clips(sample_transcript, options) == []


def test_dry_run_writes_captions_and_a_manifest(tmp_path, sample_transcript, options):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"stub")

    specs = process(source, options, transcript=sample_transcript)

    assert specs
    out = options.output_dir
    for spec in specs:
        assert (out / f"clip_{spec.index:02d}.ass").exists()
        assert (out / f"clip_{spec.index:02d}.srt").exists()
        assert spec.video_path is None       # --dry-run encodes nothing
        assert spec.title and spec.text
        assert spec.duration > 0

    manifest = json.loads((out / "clips.json").read_text())
    assert manifest["source"] == str(source)
    assert len(manifest["clips"]) == len(specs)


def test_captions_are_rebased_to_the_clip(tmp_path, sample_transcript, options):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"stub")
    options.clips = 1
    options.padding = 0.0

    process(source, options, transcript=sample_transcript)
    ass = (options.output_dir / "clip_01.ass").read_text()
    first_event = next(l for l in ass.splitlines() if l.startswith("Dialogue:"))
    # The first caption sits at the top of the clip, not at its source timecode.
    assert first_event.split(",")[1] == "0:00:00.00"


def test_padding_extends_the_cut_but_never_before_zero(tmp_path, sample_transcript, options):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"stub")
    options.padding = 2.0
    for spec in process(source, options, transcript=sample_transcript):
        assert spec.start >= 0.0


def test_missing_source_is_reported(options):
    with pytest.raises(FileNotFoundError):
        process("does-not-exist.mp4", options)


def test_llm_ranker_failure_falls_back_to_heuristics(sample_transcript, options, monkeypatch):
    options.ranker = "claude"
    import clipping.ranker_llm as ranker

    def boom(*args, **kwargs):
        raise ranker.RankerError("no API key")

    monkeypatch.setattr(ranker, "rerank", boom)
    assert plan_clips(sample_transcript, options)


def test_llm_title_wins_over_the_heuristic_one(tmp_path, sample_transcript, options, monkeypatch):
    import clipping.ranker_llm as ranker

    options.ranker = "claude"
    options.clips = 1

    def fake_rerank(candidates, **kwargs):
        for cand in candidates:
            cand.reasons.insert(0, "title::A Better Title")
        return candidates

    monkeypatch.setattr(ranker, "rerank", fake_rerank)
    source = tmp_path / "video.mp4"
    source.write_bytes(b"stub")
    assert process(source, options, transcript=sample_transcript)[0].title == "A Better Title"


def test_style_reaches_the_caption_file(tmp_path, sample_transcript, options):
    options.style = CaptionStyle(highlight="#FF00FF", uppercase=False, font="Impact")
    options.clips = 1
    source = tmp_path / "video.mp4"
    source.write_bytes(b"stub")

    process(source, options, transcript=sample_transcript)
    ass = (options.output_dir / "clip_01.ass").read_text()
    assert "Impact" in ass
    assert "&HFF00FF&" in ass


def test_voiceover_is_synthesised_once_and_mixed_into_every_clip(
    tmp_path, sample_transcript, options, monkeypatch
):
    import clipping.pipeline as pipeline
    import clipping.voiceover as vo

    calls: list[str] = []
    rendered: list[dict] = []

    def fake_synth(text, dest, **kwargs):
        calls.append(text)
        dest.write_bytes(b"wav")
        return dest

    monkeypatch.setattr(vo, "synthesize", fake_synth)
    monkeypatch.setattr(
        pipeline.render, "render_clip",
        lambda src, dest, **kw: (rendered.append(kw), Path(dest).write_bytes(b""), Path(dest))[-1],
    )
    monkeypatch.setattr(pipeline.media, "probe", lambda p: type(
        "I", (), {"duration": 999.0, "width": 1920, "height": 1080, "fps": 30.0, "has_audio": True}
    )())

    options.render_video = True
    options.clips = 2
    options.voiceover = "Isaiah Rashad sang his heart out."
    source = tmp_path / "video.mp4"
    source.write_bytes(b"stub")

    specs = pipeline.process(source, options, transcript=sample_transcript)

    assert len(calls) == 1, "narration should be synthesised once per run, not per clip"
    assert len(rendered) == len(specs) >= 2
    assert all(kw["voiceover"] is not None for kw in rendered)


def test_no_voiceover_means_no_synthesis(tmp_path, sample_transcript, options, monkeypatch):
    import clipping.voiceover as vo

    def boom(*a, **k):
        raise AssertionError("should not synthesise without --voiceover")

    monkeypatch.setattr(vo, "synthesize", boom)
    source = tmp_path / "video.mp4"
    source.write_bytes(b"stub")
    assert process(source, options, transcript=sample_transcript)
