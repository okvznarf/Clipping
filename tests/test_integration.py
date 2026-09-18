"""End-to-end test against real ffmpeg. Skipped when ffmpeg is not installed."""

import json
import shutil
import subprocess

import pytest

from clipping.captions import CaptionStyle
from clipping.media import probe
from clipping.pipeline import Options, process
from tests.conftest import speak

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="ffmpeg/ffprobe not installed",
)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    path = tmp_path_factory.mktemp("media") / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=14",
            "-f", "lavfi", "-i", "sine=frequency=220:duration=14",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(path),
        ],
        check=True,
    )
    return path


def test_probe_reads_the_source(source):
    info = probe(source)
    assert (info.width, info.height) == (640, 360)
    assert info.has_audio
    assert 13 < info.duration < 15


def test_renders_a_vertical_captioned_clip(tmp_path, source):
    options = Options(
        output_dir=tmp_path / "out",
        clips=1,
        min_duration=4,
        max_duration=10,
        ideal_duration=6,
        width=540,
        height=960,
        preset="ultrafast",
        crf=30,
        normalize_audio=False,
        style=CaptionStyle(font_size=48),
        verbose=False,
    )
    transcript = speak(
        "Here's the truth nobody tells you about shipping. Most people wait "
        "until they feel ready. Readiness arrives after you start."
    )

    specs = process(source, options, transcript=transcript)
    assert specs

    clip = specs[0]
    info = probe(clip.video_path)
    assert (info.width, info.height) == (540, 960)
    assert info.has_audio
    assert abs(info.duration - clip.duration) < 0.75

    manifest = json.loads((options.output_dir / "clips.json").read_text())
    assert manifest["clips"][0]["video_path"] == clip.video_path


@pytest.mark.parametrize("layout", ["blur", "fit"])
def test_alternate_layouts_encode(tmp_path, source, layout):
    from clipping.captions import build_ass
    from clipping.render import render_clip

    ass = tmp_path / "c.ass"
    ass.write_text(build_ass(speak("Ship it today.").words, width=360, height=640))
    dest = render_clip(
        source, tmp_path / f"{layout}.mp4",
        start=1.0, duration=2.0, layout=layout,
        width=360, height=640, subtitles=ass,
        preset="ultrafast", crf=32, normalize_audio=False,
    )
    assert probe(dest).width == 360


def test_subtitle_paths_with_awkward_characters_still_burn_in(tmp_path, source):
    from clipping.captions import build_ass
    from clipping.render import render_clip

    awkward = tmp_path / "a dir, with: odd chars"
    awkward.mkdir()
    ass = awkward / "clip 01.ass"
    ass.write_text(build_ass(speak("Ship it today.").words, width=360, height=640))

    dest = render_clip(
        source, tmp_path / "odd.mp4",
        start=0.5, duration=2.0, layout="crop",
        width=360, height=640, subtitles=ass,
        preset="ultrafast", crf=32, normalize_audio=False,
    )
    assert dest.exists() and probe(dest).duration > 1.0
