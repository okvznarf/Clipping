import pytest

from clipping.cli import PRESETS, build_parser, options_from_args, parse_size, style_from_args


def parse(argv: list[str]):
    return build_parser().parse_args(argv)


def test_defaults_are_vertical_and_sane():
    options = options_from_args(parse(["video.mp4"]))
    assert (options.width, options.height) == (1080, 1920)
    assert options.clips == 5
    assert options.layout == "crop"
    assert options.render_video is True


def test_parse_size():
    assert parse_size("720x1280") == (720, 1280)
    assert parse_size("1080X1920") == (1080, 1920)
    for bad in ("1080", "0x100", "axb"):
        with pytest.raises(ValueError):
            parse_size(bad)


def test_dry_run_and_quiet_flags():
    options = options_from_args(parse(["v.mp4", "--dry-run", "-q"]))
    assert options.render_video is False
    assert options.verbose is False


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_caption_preset_builds_a_valid_style(preset):
    style = style_from_args(parse(["v.mp4", "--caption-preset", preset]))
    assert style.font_size > 0
    assert 0 < style.position <= 1


def test_caption_flags_override_the_preset():
    style = style_from_args(
        parse(
            [
                "v.mp4", "--caption-preset", "hormozi",
                "--font", "Inter", "--font-size", "70",
                "--highlight", "#FF0000", "--words-per-line", "5",
                "--no-uppercase", "--caption-position", "0.5",
            ]
        )
    )
    assert style.font == "Inter"
    assert style.font_size == 70
    assert style.highlight == "#FF0000"
    assert style.max_words_per_line == 5
    assert style.uppercase is False
    assert style.position == 0.5


def test_invalid_caption_position_is_rejected():
    with pytest.raises(ValueError):
        style_from_args(parse(["v.mp4", "--caption-position", "1.5"]))


def test_bad_size_exits_with_a_message(capsys):
    from clipping.cli import main

    assert main(["v.mp4", "--size", "wide"]) == 2
    assert "error:" in capsys.readouterr().err


def test_missing_file_exits_nonzero(capsys):
    from clipping.cli import main

    assert main(["no-such-file.mp4", "--dry-run", "-q"]) == 1
    assert "error:" in capsys.readouterr().err


def test_unknown_layout_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        parse(["v.mp4", "--layout", "spiral"])
