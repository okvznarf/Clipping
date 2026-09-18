import re

import pytest

from clipping.captions import (
    CaptionStyle,
    build_ass,
    build_srt,
    fit_chars_per_line,
    fit_style,
    format_ass_time,
    format_srt_time,
    group_lines,
    hex_to_ass,
    hex_to_ass_inline,
)
from clipping.transcript import Word

WORDS = [
    Word("Here's", 0.0, 0.40),
    Word("the", 0.40, 0.60),
    Word("truth.", 0.60, 1.20),
    Word("Nobody", 3.00, 3.50),
    Word("tells", 3.50, 3.80),
    Word("you", 3.80, 4.00),
]


def dialogue_lines(ass: str) -> list[str]:
    return [line for line in ass.splitlines() if line.startswith("Dialogue:")]


def test_colour_conversion_is_bgr():
    assert hex_to_ass("#FF0000") == "&H000000FF"
    assert hex_to_ass("#000000", alpha=0x80) == "&H80000000"
    assert hex_to_ass_inline("#FFE23D") == "&H3DE2FF&"


def test_colour_conversion_rejects_junk():
    with pytest.raises(ValueError):
        hex_to_ass("nope")


def test_time_formats():
    assert format_ass_time(3661.5) == "1:01:01.50"
    assert format_ass_time(-1) == "0:00:00.00"
    assert format_srt_time(3661.5) == "01:01:01,500"


def test_lines_break_on_punctuation_and_pauses():
    lines = group_lines(WORDS, CaptionStyle(max_words_per_line=6, max_chars_per_line=99))
    assert [l.text for l in lines] == ["Here's the truth.", "Nobody tells you"]


def test_lines_respect_word_and_char_limits():
    style = CaptionStyle(max_words_per_line=2, max_chars_per_line=99)
    assert all(len(l.words) <= 2 for l in group_lines(WORDS, style))

    narrow = CaptionStyle(max_words_per_line=9, max_chars_per_line=10)
    for line in group_lines(WORDS, narrow):
        if len(line.words) > 1:
            assert len(line.text) <= 10 + len(line.words[-1].text)


def test_every_word_appears_exactly_once_in_lines():
    lines = group_lines(WORDS, CaptionStyle())
    assert [w.text for l in lines for w in l.words] == [w.text for w in WORDS]


def test_dialogue_lines_have_the_ten_standard_fields():
    """Regression: a missing MarginV field leaks the next field into the text."""
    ass = build_ass(WORDS)
    fmt = next(l for l in ass.splitlines() if l.startswith("Format: Layer"))
    field_count = len(fmt.split(":", 1)[1].split(","))
    assert field_count == 10
    for line in dialogue_lines(ass):
        head = line.split(",", 9)
        assert len(head) == 10
        assert not head[9].startswith(",")


def test_one_event_per_word_with_exactly_one_highlight():
    ass = build_ass(WORDS)
    events = dialogue_lines(ass)
    assert len(events) == len(WORDS)
    for event in events:
        assert event.count("\\t(0,110") == 1


def test_highlight_walks_through_the_line():
    ass = build_ass(WORDS, CaptionStyle(uppercase=True))
    highlighted = [
        re.search(r"\\t\([^)]*\)\}([^{]+)", event).group(1).strip()
        for event in dialogue_lines(ass)
    ]
    assert highlighted == ["HERE'S", "THE", "TRUTH.", "NOBODY", "TELLS", "YOU"]


def test_events_are_ordered_and_never_zero_length():
    for event in dialogue_lines(build_ass(WORDS)):
        _, start, end, *_ = event.split(",")
        assert start < end


def test_highlight_holds_until_the_next_word_starts():
    """No blink between words inside a line."""
    events = dialogue_lines(build_ass(WORDS))
    first_end = events[0].split(",")[2]
    second_start = events[1].split(",")[1]
    assert first_end == second_start == format_ass_time(WORDS[1].start)


def test_braces_and_backslashes_cannot_inject_overrides():
    words = [Word("{\\an8}evil", 0, 1)]
    ass = build_ass(words)
    text = dialogue_lines(ass)[0].split(",", 9)[9]
    assert "\\an8" not in text
    assert "(AN8)EVIL" in text


def test_resolution_reaches_the_header():
    ass = build_ass(WORDS, width=720, height=1280)
    assert "PlayResX: 720" in ass and "PlayResY: 1280" in ass


def test_position_sets_the_vertical_margin():
    ass = build_ass(WORDS, CaptionStyle(position=0.5), height=1000)
    style_line = next(l for l in ass.splitlines() if l.startswith("Style: Caption"))
    assert style_line.split(",")[-2] == "500"


def test_wrap_width_shrinks_for_large_fonts():
    assert fit_chars_per_line(1080, 120) < fit_chars_per_line(1080, 48)
    fitted = fit_style(CaptionStyle(font_size=120, max_chars_per_line=40), 1080)
    assert fitted.max_chars_per_line < 40
    # An already-narrow style is left alone.
    narrow = CaptionStyle(font_size=40, max_chars_per_line=8)
    assert fit_style(narrow, 1080) is narrow


def test_style_validation():
    with pytest.raises(ValueError):
        CaptionStyle(position=0)
    with pytest.raises(ValueError):
        CaptionStyle(max_words_per_line=0)


def test_srt_blocks_are_numbered_and_timed():
    srt = build_srt(WORDS)
    assert srt.startswith("1\n00:00:00,000 --> ")
    assert "Here's the" in srt and "Nobody tells" in srt
    # The sidecar is chunked exactly like the burned-in captions.
    expected = group_lines(WORDS, fit_style(CaptionStyle(), 1080))
    assert srt.count(" --> ") == len(expected)
    assert len(dialogue_lines(build_ass(WORDS))) == len(WORDS)


def test_srt_is_not_uppercased_or_styled():
    srt = build_srt(WORDS, CaptionStyle(uppercase=True))
    assert "Here's" in srt and "HERE'S" not in srt
    assert "{" not in srt
