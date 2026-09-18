from clipping.segmenter import Candidate, build_candidates, select, split_sentences
from clipping.transcript import Transcript, Word


def test_sentences_split_on_punctuation(sample_transcript):
    sentences = split_sentences(sample_transcript)
    assert len(sentences) >= 6
    assert sentences[0].text.endswith("business.")


def test_sentences_split_on_long_pause():
    t = Transcript([Word("one", 0, 0.4), Word("two", 3.0, 3.4)])
    assert len(split_sentences(t, pause_split=0.65)) == 2


def test_every_word_survives_splitting(sample_transcript):
    sentences = split_sentences(sample_transcript)
    assert sum(len(s) for s in sentences) == len(sample_transcript)


def test_candidates_respect_duration_bounds(sample_transcript):
    sentences = split_sentences(sample_transcript)
    cands = build_candidates(sentences, min_duration=4, max_duration=12)
    assert cands
    assert all(4 <= c.duration <= 12 for c in cands)
    # Clips begin and end on sentence boundaries.
    starts = {s.start for s in sentences}
    assert all(c.start in starts for c in cands)


def test_oversized_single_sentence_is_truncated_not_dropped():
    words = [Word(f"w{i}", i * 1.0, i * 1.0 + 0.9) for i in range(40)]
    sentences = split_sentences(Transcript(words), pause_split=5.0, max_words=100)
    cands = build_candidates(sentences, min_duration=5, max_duration=20)
    assert cands and cands[0].duration == 20


def test_overlap_is_relative_to_the_shorter_window():
    a = Candidate(0, 100, [])
    b = Candidate(90, 100, [])
    assert a.overlap(b) == 1.0
    assert Candidate(0, 10, []).overlap(Candidate(20, 30, [])) == 0.0


def test_select_prefers_high_scores_and_rejects_overlap():
    cands = [Candidate(0, 30, []), Candidate(5, 35, []), Candidate(60, 90, [])]
    cands[0].score, cands[1].score, cands[2].score = 0.9, 0.95, 0.5
    chosen = select(cands, count=3, max_overlap=0.25)
    assert [c.start for c in chosen] == [5, 60]


def test_select_returns_clips_in_timeline_order():
    cands = [Candidate(t, t + 10, []) for t in (100, 0, 50)]
    for i, c in enumerate(cands):
        c.score = i
    assert [c.start for c in select(cands, count=3)] == [0, 50, 100]
