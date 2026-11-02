#!/usr/bin/env python
"""Verify the augment_clips patch: configurable SNR + per-example background."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "openwakeword"))
import numpy as np
import scipy.io.wavfile
import torch
from openwakeword.data import augment_clips

SR, TOTAL = 16000, 32000
TMP = os.path.join(os.path.dirname(__file__), "_tmp_augment")
os.makedirs(TMP, exist_ok=True)

# One deterministic "speech" clip padded to exactly TOTAL samples so
# create_fixed_size_clip adds no random padding; identical copies isolate
# the background-noise randomness.
rng = np.random.default_rng(0)
clip = (0.1 * np.sin(2 * np.pi * 220 * np.arange(TOTAL) / SR)).astype(np.float32)
clip_path = os.path.join(TMP, "clip.wav")
scipy.io.wavfile.write(clip_path, SR, (clip * 32767).astype(np.int16))
bg_paths = []
for i in range(4):
    p = os.path.join(TMP, f"bg{i}.wav")
    scipy.io.wavfile.write(p, SR, (rng.normal(0, 0.05, TOTAL * 2) * 32767).astype(np.int16))
    bg_paths.append(p)

NO_AUG = {"SevenBandParametricEQ": 0, "TanhDistortion": 0, "PitchShift": 0,
          "BandStopFilter": 0, "AddColoredNoise": 0, "AddBackgroundNoise": 1.0,
          "Gain": 0, "RIR": 0}

def run(mode, lo, hi, n=8):
    gen = augment_clips([clip_path] * n, total_length=TOTAL, batch_size=n,
                        augmentation_probabilities=NO_AUG,
                        background_clip_paths=bg_paths, RIR_paths=[],
                        background_min_snr_db=lo, background_max_snr_db=hi,
                        background_mode=mode)
    out = next(iter(gen))
    return np.asarray(out, dtype=np.float32)

torch.manual_seed(0)
per_ex = run("per_example", -10, 15)
assert per_ex.shape == (8, TOTAL), per_ex.shape
assert np.isfinite(per_ex).all()
# per_example: identical inputs must yield different outputs (independent noise draws)
diffs = [np.abs(per_ex[i] - per_ex[j]).max() for i in range(8) for j in range(i + 1, 8)]
assert min(diffs) > 1e-4, f"per_example outputs identical: {min(diffs)}"

torch.manual_seed(0)
per_batch = run("per_batch", -10, 15)
# per_batch: one draw for the whole batch of identical inputs -> identical outputs
assert np.abs(per_batch - per_batch[0]).max() < 1e-6, "per_batch outputs differ"

# SNR plumbing: fixed -20 dB SNR buries the tone in noise; +30 dB barely touches it
torch.manual_seed(1)
loud = run("per_example", -20, -20)
torch.manual_seed(1)
quiet = run("per_example", 30, 30)
# augment_clips returns int16-scaled audio (see the `*32767).astype(np.int16)` yield in
# data.py); `clip` above is the raw float array written to the wav file. Divide back down
# to the same float scale before comparing RMS, or the ~32767x unit mismatch swamps the
# actual (correct) SNR effect being tested.
clean_rms = float(np.sqrt((clip ** 2).mean()))
loud_rms = float(np.sqrt(((loud / 32767.0) ** 2).mean()))
quiet_rms = float(np.sqrt(((quiet / 32767.0) ** 2).mean()))
assert loud_rms > 2.5 * clean_rms, (loud_rms, clean_rms)
assert quiet_rms < 1.5 * clean_rms, (quiet_rms, clean_rms)
print("PASS test_augment_patch")
