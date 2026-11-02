#!/usr/bin/env python
"""Verify the noisy-clip augmentation patch.

Clips whose filename matches `noisy_clip_pattern` are ALREADY noisy (real
recordings made over music), so mixing another background over them at the
harsh training SNR buries the utterance past recovery. They must instead get
the mild `noisy_background_{min,max}_snr_db` range, while every other clip in
the same call keeps the harsh range.

Run with ./.venv/bin/python (no pytest).
"""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "openwakeword"))
import numpy as np
import scipy.io.wavfile
import torch
from openwakeword.data import augment_clips

SR, TOTAL = 16000, 32000
TMP = os.path.join(os.path.dirname(__file__), "_tmp_noisy_augment")
os.makedirs(TMP, exist_ok=True)

# Two deterministic quiet tones, each padded to exactly TOTAL samples so
# create_fixed_size_clip adds no random padding. The tone frequency is a label
# that survives augmentation (only AddBackgroundNoise is enabled), so each
# output clip can be traced back to its partition regardless of yield order.
NOISY_HZ, NORMAL_HZ, AMP = 220.0, 440.0, 0.01


def write_tone(path, freq):
    x = (AMP * np.sin(2 * np.pi * freq * np.arange(TOTAL) / SR)).astype(np.float32)
    scipy.io.wavfile.write(path, SR, (x * 32767).astype(np.int16))
    return x


CLEAN_RMS = AMP / np.sqrt(2.0)  # rms of a sine of amplitude AMP

# A loud background bank. torch_audiomentations rescales the background to hit
# the requested SNR, so its absolute level only has to be non-degenerate.
rng = np.random.default_rng(0)
BG_PATHS = []
for i in range(4):
    p = os.path.join(TMP, f"bg{i}.wav")
    scipy.io.wavfile.write(p, SR, (rng.normal(0, 0.2, TOTAL * 2) * 32767).astype(np.int16))
    BG_PATHS.append(p)

NO_AUG = {"SevenBandParametricEQ": 0, "TanhDistortion": 0, "PitchShift": 0,
          "BandStopFilter": 0, "AddColoredNoise": 0, "AddBackgroundNoise": 1.0,
          "Gain": 0, "RIR": 0}

PATTERN = "REAL_MUSIC_*"
HARSH, MILD = -20.0, 30.0


def make_clips(n_noisy, n_normal):
    """Interleaved matching/non-matching clip paths (so a naive filter that
    keeps only a contiguous run would be caught)."""
    paths = []
    for i in range(max(n_noisy, n_normal)):
        if i < n_noisy:
            p = os.path.join(TMP, f"REAL_MUSIC_speaker_b_{i:03d}.wav")
            write_tone(p, NOISY_HZ)
            paths.append(p)
        if i < n_normal:
            p = os.path.join(TMP, f"tts_sample_{i:03d}.wav")
            write_tone(p, NORMAL_HZ)
            paths.append(p)
    return paths


def seed_all(s=0):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)


def run(clip_paths, batch_size, **kwargs):
    """Return every yielded clip as one (n_clips, TOTAL) float array."""
    gen = augment_clips(clip_paths, total_length=TOTAL, batch_size=batch_size,
                        augmentation_probabilities=NO_AUG,
                        background_clip_paths=BG_PATHS, RIR_paths=[],
                        background_mode="per_example", **kwargs)
    batches = [np.asarray(b, dtype=np.float32) for b in gen]
    for b in batches:
        assert b.ndim == 2 and b.shape[0] > 0 and b.shape[1] == TOTAL, b.shape
    return np.concatenate(batches, axis=0)


def tone_of(clip):
    """Which tone dominates this output clip: NOISY_HZ or NORMAL_HZ."""
    spec = np.abs(np.fft.rfft(clip))
    freqs = np.fft.rfftfreq(clip.shape[0], 1.0 / SR)
    band = lambda f: spec[np.abs(freqs - f) <= 5.0].max()
    return NOISY_HZ if band(NOISY_HZ) > band(NORMAL_HZ) else NORMAL_HZ


def rms_ratio(clip):
    """Output RMS as a multiple of the clean tone's RMS. augment_clips yields
    int16-scaled audio, so divide back to the float scale of the source tone."""
    return float(np.sqrt(((clip / 32767.0) ** 2).mean())) / CLEAN_RMS


# ---------------------------------------------------------------- 1. exemption
# 3 matching + 5 non-matching in one call: matching clips must come out with
# markedly LESS added background energy than the non-matching ones.
clips = make_clips(3, 5)
seed_all(0)
out = run(clips, batch_size=4,
          background_min_snr_db=HARSH, background_max_snr_db=HARSH,
          noisy_clip_pattern=PATTERN,
          noisy_background_min_snr_db=MILD, noisy_background_max_snr_db=MILD)
assert out.shape == (8, TOTAL), out.shape
assert np.isfinite(out).all()

noisy_out = [c for c in out if tone_of(c) == NOISY_HZ]
normal_out = [c for c in out if tone_of(c) == NORMAL_HZ]
assert len(noisy_out) == 3, f"expected 3 matching clips out, got {len(noisy_out)}"
assert len(normal_out) == 5, f"expected 5 non-matching clips out, got {len(normal_out)}"

noisy_ratios = [rms_ratio(c) for c in noisy_out]
normal_ratios = [rms_ratio(c) for c in normal_out]
assert max(noisy_ratios) < 1.5, f"matching clips got harsh background: {noisy_ratios}"
assert min(normal_ratios) > 3.0, f"non-matching clips lost harsh background: {normal_ratios}"
assert min(normal_ratios) > 3 * max(noisy_ratios), (normal_ratios, noisy_ratios)
print("PASS noisy clips get the mild background range, others stay harsh")

# ------------------------------------------------------- 2. nothing is dropped
# Uneven partition vs batch_size: 5 matching, 7 non-matching, batch_size 4
# (5 -> 4+1, 7 -> 4+3). Every input clip must still appear in the output.
clips = make_clips(5, 7)
seed_all(1)
out = run(clips, batch_size=4,
          background_min_snr_db=HARSH, background_max_snr_db=HARSH,
          noisy_clip_pattern=PATTERN,
          noisy_background_min_snr_db=MILD, noisy_background_max_snr_db=MILD)
assert out.shape[0] == len(clips) == 12, (out.shape[0], len(clips))
tones = [tone_of(c) for c in out]
assert tones.count(NOISY_HZ) == 5, tones.count(NOISY_HZ)
assert tones.count(NORMAL_HZ) == 7, tones.count(NORMAL_HZ)
print("PASS uneven partition yields every input clip exactly once")

# ----------------------------------------------------- 3. default is unchanged
# 3a: with no pattern, the REAL_MUSIC clips get the harsh range like everything
# else -- the exemption is opt-in, not implicit in the filename.
clips = make_clips(3, 5)
seed_all(2)
out = run(clips, batch_size=4,
          background_min_snr_db=HARSH, background_max_snr_db=HARSH,
          noisy_clip_pattern="",
          noisy_background_min_snr_db=MILD, noisy_background_max_snr_db=MILD)
assert out.shape == (8, TOTAL), out.shape
ratios = [rms_ratio(c) for c in out]
assert min(ratios) > 3.0, f"empty pattern exempted something: {ratios}"

# 3b: passing the new kwargs with an empty pattern is bit-for-bit identical to
# not passing them at all (same clips, same seed, same batching).
seed_all(3)
baseline = run(clips, batch_size=4,
               background_min_snr_db=HARSH, background_max_snr_db=HARSH)
seed_all(3)
with_defaults = run(clips, batch_size=4,
                    background_min_snr_db=HARSH, background_max_snr_db=HARSH,
                    noisy_clip_pattern="",
                    noisy_background_min_snr_db=MILD, noisy_background_max_snr_db=MILD)
assert baseline.shape == with_defaults.shape, (baseline.shape, with_defaults.shape)
assert np.array_equal(baseline, with_defaults), \
    f"default path diverged, max abs diff {np.abs(baseline - with_defaults).max()}"
print("PASS empty noisy_clip_pattern preserves existing behaviour exactly")

print("PASS test_noisy_clip_augmentation")
