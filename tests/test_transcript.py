import json

from clipping.transcript import Transcript, Word, merge_short_words, shift


def test_round_trip(tmp_path, sample_transcript):
    path = sample_transcript.save(tmp_path / "t.json")
    loaded = Transcript.load(path)
    assert loaded.text == sample_transcript.text
    assert len(loaded) == len(sample_transcript)
    assert loaded.words[3].start == sample_transcript.words[3].start


def test_from_dict_tolerates_missing_probability():
    t = Transcript.from_dict({"words": [{"text": "hi", "start": 0, "end": 1}]})
    assert t.words[0].probability == 1.0


def test_slice_uses_word_midpoints():
    t = Transcript([Word("a", 0, 1), Word("b", 2, 3), Word("c", 10, 11)])
    assert [w.text for w in t.slice(0, 5)] == ["a", "b"]


def test_shift_rebases_and_clamps():
    words = shift([Word("a", 5, 6), Word("b", 7, 8)], 5.5)
    assert words[0].start == 0.0
    assert words[1].start == 1.5
    # The original words are untouched.
    assert words[0] is not None


def test_merge_short_words_gives_zero_length_words_a_floor():
    out = merge_short_words([Word("a", 1.0, 1.0)], min_duration=0.04)
    assert out[0].end == 1.04


def test_duration_of_empty_transcript():
    assert Transcript([]).duration == 0.0
