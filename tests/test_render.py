import pytest

from clipping.render import LAYOUTS, build_filter, escape_filter_path


@pytest.mark.parametrize("layout", LAYOUTS)
def test_every_layout_produces_one_labelled_output(layout):
    graph = build_filter(layout, 1080, 1920, subtitles="clip.ass")
    assert graph.endswith("[v]")
    assert graph.count("[v]") == 1
    assert "subtitles='clip.ass'" in graph


def test_crop_fills_the_frame_and_fit_pads_it():
    crop = build_filter("crop", 1080, 1920)
    assert "force_original_aspect_ratio=increase" in crop and "crop=1080:1920" in crop
    fit = build_filter("fit", 1080, 1920)
    assert "force_original_aspect_ratio=decrease" in fit and "pad=1080:1920" in fit


def test_blur_layout_splits_into_background_and_foreground():
    graph = build_filter("blur", 1080, 1920)
    assert "split=2" in graph and "gblur" in graph and "overlay=" in graph


def test_subtitles_are_optional():
    assert "subtitles" not in build_filter("crop", 1080, 1920)


def test_unknown_layout_is_rejected():
    with pytest.raises(ValueError):
        build_filter("spiral", 1080, 1920)


def test_filter_path_escaping():
    assert escape_filter_path("C:\\videos\\a.ass") == r"C\:/videos/a.ass"
    assert escape_filter_path("it's.ass") == r"it\'s.ass"


def test_audio_filter_ducks_for_the_length_of_the_narration():
    from clipping.render import build_audio_filter

    graph = build_audio_filter(has_audio=True, voiceover_duration=4.0, delay=0.5, duck=0.3)
    # Duck spans the narration plus a short tail, and restores afterwards.
    assert "between(t,0.500,4.850)" in graph
    assert ",0.3,1)" in graph
    assert "adelay=500|500" in graph
    assert "amix=inputs=2:duration=first:normalize=0" in graph
    assert graph.endswith("[aout]")


def test_audio_filter_without_source_audio_uses_narration_alone():
    from clipping.render import build_audio_filter

    graph = build_audio_filter(has_audio=False, voiceover_duration=3.0)
    assert "[0:a]" not in graph
    assert "amix" not in graph
    assert graph.endswith("[aout]")


def test_audio_filter_honours_the_normalise_toggle():
    from clipping.render import build_audio_filter

    assert "loudnorm" in build_audio_filter(has_audio=True, voiceover_duration=2.0)
    assert "loudnorm" not in build_audio_filter(
        has_audio=True, voiceover_duration=2.0, normalize=False
    )


def test_negative_delay_cannot_produce_an_invalid_adelay():
    from clipping.render import build_audio_filter

    assert "adelay=0|0" in build_audio_filter(
        has_audio=True, voiceover_duration=2.0, delay=-1.0
    )
