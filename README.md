# clipping

Turn a long video — a podcast, an interview, a talk — into short vertical clips
with animated, word-synced captions burned in.

```
long video  →  transcribe  →  score every window  →  pick the best  →  cut, reframe, caption
```

One command:

```bash
python -m clipping talk.mp4 -n 5
```

Writes to `out/`:

```
out/
  clip_01.mp4     1080x1920, captions burned in, loudness-normalised
  clip_01.ass     the caption script that was burned in
  clip_01.srt     plain sidecar, for platforms that want an upload file
  transcript.json word-level transcript (reusable, see --transcript)
  clips.json      what was chosen, when, why, and with what title
```

## Install

```bash
pip install -e .            # the tool itself
pip install -e ".[download,llm,dev]"   # URL input, Claude ranking, tests
```

FFmpeg is required and must be on your `PATH` (`ffmpeg` and `ffprobe`), built
with libass — every mainstream build is. Check with `ffmpeg -filters | grep subtitles`.

Captions use whatever font you name, so it has to be installed locally. The
default is Arial Black; on a bare Linux box try `--font "DejaVu Sans"` or
install `fonts-liberation`.

## How clips get chosen

Candidates are every run of consecutive sentences that fits the duration window,
so a clip never opens or closes mid-sentence. Each one is scored on signals that
track how a clip plays on its own:

| signal | what it rewards |
| --- | --- |
| `hook` | the opening line promises something ("here's why…", "most people…") |
| `emphasis` | emphatic, high-energy vocabulary |
| `question` | a question posed to the viewer |
| `numbers` | concrete figures rather than vague claims |
| `duration` | length near `--ideal-duration` |
| `pace` | a natural ~2.8 words/sec — not dead air, not an auctioneer |
| `completeness` | starts a thought and finishes it |
| `cleanliness` | little um/uh/like filler |

The winners are taken highest-score-first, skipping anything that overlaps a
clip already chosen (`--max-overlap`). Every score and its breakdown lands in
`clips.json`, so you can see why a clip was picked.

Heuristics judge *form*, not whether an idea is interesting. With an Anthropic
API key you can add that second opinion:

```bash
export ANTHROPIC_API_KEY=...
python -m clipping talk.mp4 --ranker claude
```

The shortlist is re-scored by the model and blended with the heuristic score
(`--llm-weight`, 0–1), and the model also writes each clip's title. If the key
is missing or the call fails, the run continues on heuristics alone.

## Captions

Captions are word-synced: a short line sits on screen and the word currently
being spoken is recoloured and pops. That comes out of the word-level timestamps
Whisper produces, rendered as an ASS script with one event per word.

Four presets, then override anything:

```bash
python -m clipping talk.mp4 --caption-preset hormozi
python -m clipping talk.mp4 --font Inter --font-size 72 --highlight "#39FF14" \
                            --words-per-line 3 --caption-position 0.62 --no-uppercase
```

| preset | look |
| --- | --- |
| `bold` | white Arial Black, yellow highlight, uppercase (default) |
| `clean` | sentence case, lighter outline, cyan highlight |
| `minimal` | small, low, no pop — for talking-head footage |
| `hormozi` | huge Impact, green highlight, 3 words a line, mid-frame |

Line width is derived from the frame size and font size, so text stays inside
the frame when you scale the font up.

## Reframing

`--layout` decides how a 16:9 source becomes 9:16:

- `crop` — scale up and centre-crop (default; best for a centred speaker)
- `blur` — fit the whole frame over a blurred fill of itself (nothing is lost)
- `fit` — letterbox onto black
- `original` — keep the source framing, just cut and caption

## Iterating without re-transcribing

Transcription is the slow part, and it's cached in `out/transcript.json`. Reuse
it while you tune the look:

```bash
python -m clipping talk.mp4 -o out                       # once
python -m clipping talk.mp4 -o out --transcript out/transcript.json \
       --caption-preset hormozi --layout blur            # instant
```

`--dry-run` goes further: it picks clips and writes caption files but encodes no
video, so you can check the selection in a second.

## Common options

```
-n, --clips N            how many clips to produce (default 5)
--min/--max-duration     clip length bounds in seconds (15–60)
--ideal-duration         length the scorer treats as ideal (32)
--padding                breathing room added to each end (0.25s)
--model                  whisper model: tiny/base/small/medium/large-v3
--language               force a language code, skips detection
--device cpu|cuda        force a device (default: auto-detect, CPU fallback)
--size 1080x1920         output resolution
--crf / --preset         x264 quality and speed
--dry-run                pick clips, write captions, encode nothing
```

`python -m clipping --help` lists everything.

A URL works as the source if `yt-dlp` is installed:

```bash
python -m clipping "https://example.com/watch?v=..." -n 3
```

## Using it as a library

```python
from clipping.pipeline import Options, process
from clipping.captions import CaptionStyle

specs = process("talk.mp4", Options(
    clips=3,
    layout="blur",
    style=CaptionStyle(font="Impact", highlight="#39FF14"),
))
for spec in specs:
    print(spec.start, spec.duration, spec.title, spec.video_path)
```

The pieces are independent: `clipping.transcribe` produces a `Transcript`,
`clipping.segmenter` + `clipping.scoring` choose windows, `clipping.captions`
renders ASS/SRT from words, and `clipping.render` drives ffmpeg. Only the last
two touch the filesystem or shell out.

## Tests

```bash
python -m pytest
```

Everything but `tests/test_integration.py` runs without ffmpeg; the integration
tests encode real clips and skip themselves when ffmpeg is missing.

## Speed

Transcription dominates. `small` on CPU runs roughly 5–10× faster than realtime;
on a GPU, `--device cuda --compute-type float16` with `medium` or `large-v3` is
both faster and noticeably more accurate at word timings, which is what caption
sync depends on. Encoding five 30-second clips takes well under a minute.

GPU transcription needs more than an NVIDIA card: CTranslate2 loads the cuBLAS
and cuDNN runtime libraries, which are a separate install from the driver. The
default `--device auto` tries the GPU and falls back to CPU with a warning if
those libraries are missing, so a machine without them still works. Use
`--device cpu` to skip the attempt, or `--device cuda` to make a missing runtime
an error instead of a fallback.
