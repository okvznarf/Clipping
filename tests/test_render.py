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
