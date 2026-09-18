"""Caption rendering: word-level ASS subtitles plus a plain SRT sidecar.

The ASS output is the animated, one-word-at-a-time style short-form clips use:
a short line of context is on screen, and the word currently being spoken is
recoloured and pops slightly. That is done by emitting one Dialogue event per
word, each showing the whole line with a different word highlighted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from .transcript import Word, merge_short_words

# Gap between words that forces a new caption line, in seconds.
LINE_BREAK_PAUSE = 0.7
# Rough average glyph width as a fraction of font size, for the heavy sans-serif
# fonts these captions use. Only needs to be good enough to keep lines inside the
# frame; libass still wraps anything that slips past it.
AVG_GLYPH_RATIO = 0.62
# Fraction of frame width kept clear on each side.
SIDE_MARGIN_RATIO = 0.06


def fit_chars_per_line(width: int, font_size: int, pop_scale: int = 100) -> int:
    """How many characters fit on one line at this width and font size."""
    usable = width * (1 - 2 * SIDE_MARGIN_RATIO)
    glyph = font_size * AVG_GLYPH_RATIO * max(1.0, pop_scale / 100)
    return max(6, int(usable / glyph))


def _bgr(colour: str) -> str:
    value = colour.strip().lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", value):
        raise ValueError(f"expected a #RRGGBB colour, got {colour!r}")
    r, g, b = value[0:2], value[2:4], value[4:6]
    return f"{b}{g}{r}".upper()


def hex_to_ass(colour: str, alpha: int = 0) -> str:
    """'#RRGGBB' -> '&HAABBGGRR', the form used in a [V4+ Styles] line."""
    return f"&H{alpha:02X}{_bgr(colour)}"


def hex_to_ass_inline(colour: str) -> str:
    """'#RRGGBB' -> '&HBBGGRR&', the terminated form used in a \\1c override."""
    return f"&H{_bgr(colour)}&"


@dataclass
class CaptionStyle:
    font: str = "Arial Black"
    font_size: int = 86
    primary: str = "#FFFFFF"
    highlight: str = "#FFE23D"
    outline: str = "#000000"
    outline_width: float = 5.0
    shadow: float = 1.5
    bold: bool = True
    uppercase: bool = True
    #: Vertical placement of the caption block, as a fraction of frame height.
    position: float = 0.76
    max_words_per_line: int = 4
    max_chars_per_line: int = 20
    #: Scale the spoken word grows to, in percent (100 = no pop).
    pop_scale: int = 112
    #: Fade-in for each new line, in milliseconds.
    fade_in_ms: int = 120

    def __post_init__(self) -> None:
        if not 0.0 < self.position <= 1.0:
            raise ValueError("position must be within (0, 1]")
        if self.max_words_per_line < 1:
            raise ValueError("max_words_per_line must be >= 1")


@dataclass
class CaptionLine:
    words: list[Word] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def group_lines(words: list[Word], style: CaptionStyle) -> list[CaptionLine]:
    """Chunk words into caption lines on word count, width, pauses and punctuation."""
    lines: list[CaptionLine] = []
    current: list[Word] = []
    chars = 0

    for word in words:
        clean = word.text.strip()
        if not clean:
            continue
        gap = word.start - current[-1].end if current else 0.0
        too_many = len(current) >= style.max_words_per_line
        too_wide = current and chars + 1 + len(clean) > style.max_chars_per_line
        if current and (too_many or too_wide or gap >= LINE_BREAK_PAUSE):
            lines.append(CaptionLine(current))
            current, chars = [], 0
        current.append(word)
        chars += len(clean) + (1 if chars else 0)
        if clean.rstrip('"\'")]').endswith((".", "?", "!", "…")):
            lines.append(CaptionLine(current))
            current, chars = [], 0

    if current:
        lines.append(CaptionLine(current))
    return lines


def _escape(text: str) -> str:
    """ASS treats braces and backslashes as markup; strip them from spoken text."""
    return text.replace("\\", "").replace("{", "(").replace("}", ")").strip()


def format_ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    centis = int(round(seconds * 100))
    hours, centis = divmod(centis, 360000)
    minutes, centis = divmod(centis, 6000)
    secs, centis = divmod(centis, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def format_srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3600000)
    minutes, millis = divmod(millis, 60000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _header(style: CaptionStyle, width: int, height: int) -> str:
    primary = hex_to_ass(style.primary)
    outline = hex_to_ass(style.outline)
    highlight = hex_to_ass(style.highlight)
    margin_v = max(10, int(height * (1.0 - style.position)))
    side_margin = int(width * SIDE_MARGIN_RATIO)
    bold = -1 if style.bold else 0
    return "\n".join(
        [
            "[Script Info]",
            "; Generated by clipping (https://github.com/okvznarf/clipping)",
            "ScriptType: v4.00+",
            f"PlayResX: {width}",
            f"PlayResY: {height}",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "YCbCr Matrix: TV.709",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Caption,{style.font},{style.font_size},{primary},{highlight},"
            f"{outline},{hex_to_ass('#000000', alpha=0x80)},{bold},0,0,0,"
            f"100,100,0,0,1,{style.outline_width},{style.shadow},2,"
            f"{side_margin},{side_margin},{margin_v},1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
            "Effect, Text",
        ]
    )


def fit_style(style: CaptionStyle, width: int) -> CaptionStyle:
    """Copy of ``style`` whose line width cannot overflow a ``width``-wide frame."""
    limit = fit_chars_per_line(width, style.font_size, style.pop_scale)
    if style.max_chars_per_line <= limit:
        return style
    fitted = replace(style, max_chars_per_line=limit)
    return fitted


def build_ass(
    words: list[Word],
    style: CaptionStyle | None = None,
    *,
    width: int = 1080,
    height: int = 1920,
) -> str:
    """Render word-synced captions as an ASS subtitle script."""
    style = style or CaptionStyle()
    words = merge_short_words(words)
    lines = group_lines(words, fit_style(style, width))

    primary = hex_to_ass_inline(style.primary)
    highlight = hex_to_ass_inline(style.highlight)
    events: list[str] = []

    for line in lines:
        rendered = [_escape(w.text) for w in line.words]
        if style.uppercase:
            rendered = [t.upper() for t in rendered]

        for index, word in enumerate(line.words):
            start = word.start
            if index + 1 < len(line.words):
                # Hold the highlight until the next word actually starts so the
                # line never blinks between words.
                end = max(start + 0.02, line.words[index + 1].start)
            else:
                end = max(start + 0.02, word.end)

            parts = []
            for i, token in enumerate(rendered):
                if i == index:
                    parts.append(
                        f"{{\\1c{highlight}\\fscx100\\fscy100"
                        f"\\t(0,110,\\fscx{style.pop_scale}\\fscy{style.pop_scale})}}"
                        f"{token}{{\\1c{primary}\\fscx100\\fscy100}}"
                    )
                else:
                    parts.append(token)
            text = " ".join(parts)
            if index == 0 and style.fade_in_ms > 0:
                text = f"{{\\fad({style.fade_in_ms},0)}}" + text
            events.append(
                f"Dialogue: 0,{format_ass_time(start)},{format_ass_time(end)},"
                f"Caption,,0,0,0,,{text}"
            )

    return _header(style, width, height) + "\n" + "\n".join(events) + "\n"


def build_srt(
    words: list[Word], style: CaptionStyle | None = None, *, width: int = 1080
) -> str:
    """Plain SRT of the same lines, for platforms that want an upload sidecar."""
    style = style or CaptionStyle()
    lines = group_lines(merge_short_words(words), fit_style(style, width))
    blocks = []
    for number, line in enumerate(lines, start=1):
        text = " ".join(_escape(w.text) for w in line.words)
        blocks.append(
            f"{number}\n"
            f"{format_srt_time(line.start)} --> {format_srt_time(line.end)}\n"
            f"{text}\n"
        )
    return "\n".join(blocks)
