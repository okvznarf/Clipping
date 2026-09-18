"""Loudness envelope maths. The ffmpeg decode is covered separately."""

import array
import math
import shutil
import subprocess

import pytest

from clipping.audio import Energy, decode_pcm, rms_envelope

SAMPLE_RATE = 8000


def tone(amplitude: float, seconds: float, rate: int = SAMPLE_RATE) -> bytes:
    samples = array.array(
        "h",
        [
            int(amplitude * 32767 * math.sin(2 * math.pi * 440 * t / rate))
            for t in range(int(rate * seconds))
        ],
    )
    return samples.tobytes()


def test_envelope_has_one_value_per_hop():
    pcm = tone(0.5, 3.0)
    assert len(rms_envelope(pcm, sample_rate=SAMPLE_RATE, hop=1.0)) == 3
    assert len(rms_envelope(pcm, sample_rate=SAMPLE_RATE, hop=0.5)) == 6


def test_envelope_tracks_loudness():
    envelope = rms_envelope(
        tone(0.05, 1) + tone(0.9, 1) + tone(0.05, 1), sample_rate=SAMPLE_RATE
    )
    quiet_a, loud, quiet_b = envelope
    assert loud > quiet_a * 5
    assert quiet_a == pytest.approx(quiet_b, rel=0.2)
    assert all(0.0 <= v <= 1.0 for v in envelope)


def test_silence_reads_as_zero():
    # 2 bytes per sample, so this is exactly two seconds of silence.
    assert rms_envelope(b"\x00" * (SAMPLE_RATE * 2 * 2)) == [0.0, 0.0]


def test_envelope_handles_empty_and_ragged_input():
    assert rms_envelope(b"") == []
    # A trailing odd byte cannot form a sample and must not raise.
    assert len(rms_envelope(tone(0.5, 1.0) + b"\x01")) == 1


def test_hop_must_be_positive():
    with pytest.raises(ValueError):
        rms_envelope(tone(0.5, 1.0), hop=0)


def test_energy_scores_loud_spans_above_quiet_ones():
    energy = Energy(rms_envelope(tone(0.05, 2) + tone(0.9, 2), sample_rate=SAMPLE_RATE))
    assert energy.mean(2, 4) > 0.9
    assert energy.mean(0, 2) < 0.2


def test_energy_is_bounded_and_clamps_out_of_range_spans():
    energy = Energy(rms_envelope(tone(0.6, 3), sample_rate=SAMPLE_RATE))
    assert 0.0 <= energy.mean(-5, 99) <= 1.0
    # A span past the end of the envelope still returns something usable.
    assert 0.0 <= energy.mean(60, 90) <= 1.0


def test_a_flat_envelope_is_neutral():
    """With no audio the signal must not tilt a ranking either way."""
    assert Energy.flat().mean(0, 30) == 0.5
    assert Energy([0.0, 0.0, 0.0]).mean(0, 3) == 0.5


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_decode_pcm_reads_real_media(tmp_path):
    source = tmp_path / "tone.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            str(source),
        ],
        check=True,
    )
    pcm = decode_pcm(source)
    # 3 seconds of mono 16-bit at 8 kHz, within a frame of rounding.
    assert abs(len(pcm) - 3 * SAMPLE_RATE * 2) < SAMPLE_RATE

    envelope = Energy.from_file(source)
    assert len(envelope.values) == 3
    assert envelope.mean(0, 3) > 0.5


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_decode_pcm_reports_a_bad_file(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not media")
    with pytest.raises(RuntimeError, match="could not decode"):
        decode_pcm(broken)
