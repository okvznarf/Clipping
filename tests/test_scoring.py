from clipping import scoring
from clipping.segmenter import Candidate, build_candidates, split_sentences
from tests.conftest import speak


def make(text: str, duration: float = 32.0) -> Candidate:
    transcript = speak(text)
    sentences = split_sentences(transcript)
    cand = Candidate(0.0, duration, sentences)
    return scoring.score_candidate(cand)


def test_hook_only_counts_the_opening():
    assert scoring.hook_score("Here's why this matters.") > 0
    assert scoring.hook_score("word " * 40 + "here's why this matters") == 0


def test_duration_score_peaks_at_the_ideal():
    assert scoring.duration_score(32) > scoring.duration_score(20)
    assert scoring.duration_score(20) > scoring.duration_score(90)
    assert scoring.duration_score(32, ideal=32) == 1.0


def test_pace_rejects_dead_air_and_auctioneering():
    assert scoring.pace_score(90, 32) > 0.9
    assert scoring.pace_score(5, 60) == 0.0     # near silence
    assert scoring.pace_score(400, 30) == 0.0   # impossibly fast
    assert scoring.pace_score(10, 0) == 0.0


def test_completeness_penalises_dangling_starts_and_ends():
    assert scoring.completeness_score("And then it broke") == 0.0
    assert scoring.completeness_score("It broke.") == 1.0
    assert scoring.completeness_score("") == 0.0


def test_cleanliness_penalises_filler():
    clean = scoring.cleanliness_score("We shipped the feature on Friday morning.")
    filler = scoring.cleanliness_score("Um so like yeah okay um right so like yeah")
    assert clean > filler


def test_scores_stay_within_range_and_beat_filler():
    strong = make(
        "Here's the truth nobody tells you. Most people never ship anything. "
        "The biggest mistake is waiting three years for permission."
    )
    weak = make("Um so yeah like okay right so um yeah like you know I mean so")
    assert 0.0 <= weak.score <= strong.score <= 1.0
    assert strong.score > weak.score


def test_reasons_explain_the_score():
    cand = make("Here's the truth: most people never actually finish anything.")
    assert cand.reasons
    assert any(r.startswith("hook") for r in cand.reasons)


def test_score_all_covers_every_candidate(sample_transcript):
    cands = build_candidates(split_sentences(sample_transcript), min_duration=4, max_duration=15)
    scoring.score_all(cands)
    assert all(c.score > 0 for c in cands)


def test_suggest_title_is_short_and_capitalised():
    title = scoring.suggest_title(
        "here's the truth nobody tells you about starting a business. Other stuff."
    )
    assert title.startswith("Here's")
    assert len(title) <= 70
    assert scoring.suggest_title("") == "Clip"


def test_suggest_title_truncates_on_a_word_boundary():
    title = scoring.suggest_title("word " * 60, max_chars=30)
    assert len(title) <= 30
    assert title.endswith("…")
