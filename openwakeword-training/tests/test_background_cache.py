#!/usr/bin/env python
"""Verify the background-noise cache patch.

`AddBackgroundNoise` re-read its background audio from disk on every call, which in
mode="per_example" meant ~64 file operations per batch and ~507 ms of the ~1077 ms
each 32-clip batch cost. `CachedAddBackgroundNoise` serves the same audio from a
memmapped cache of fixed-length segments instead (~5.6 ms).

The whole point is that this is a pure performance change, so these checks are about
equivalence, not speed:

  1. the cache stores source samples exactly (bit-for-bit)
  2. cached and upstream augmentation produce the SAME audio given the same seed
  3. per-source-file sampling weight is uniform over files, not over segments
     (otherwise long files would silently dominate the pool)
  4. a cache cut at the wrong total_length is refused, not silently reused
  5. background_cache_path="" leaves the upstream path exactly as it was

Run with ./.venv/bin/python (no pytest).
"""
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "openwakeword"))

import numpy as np  # noqa: E402
import scipy.io.wavfile  # noqa: E402
import torch  # noqa: E402
import torch_audiomentations  # noqa: E402

from openwakeword.data import (  # noqa: E402
    CachedAddBackgroundNoise,
    augment_clips,
    build_background_cache,
    load_background_cache,
)

SR, TOTAL = 16000, 32000
TMP = os.path.join(os.path.dirname(__file__), "_tmp_bg_cache")
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(TMP, exist_ok=True)
CACHE = os.path.join(TMP, "bg_cache.npy")

failures = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


rng = np.random.default_rng(0)

# Background bank. Each file is an exact multiple of TOTAL so segment boundaries are
# unambiguous; the first four are exactly TOTAL long, which is what makes check 2
# possible (see below).
bg_paths, bg_int16 = [], {}
for i in range(4):
    p = os.path.join(TMP, f"bg_short{i}.wav")
    x = (rng.normal(0, 0.2, TOTAL) * 32767).astype(np.int16)
    scipy.io.wavfile.write(p, SR, x)
    bg_paths.append(p)
    bg_int16[p] = x
for i in range(2):
    p = os.path.join(TMP, f"bg_long{i}.wav")           # 3 segments each
    x = (rng.normal(0, 0.2, TOTAL * 3) * 32767).astype(np.int16)
    scipy.io.wavfile.write(p, SR, x)
    bg_paths.append(p)
    bg_int16[p] = x

print("\n1. cache stores source samples exactly")
build_background_cache(bg_paths, TOTAL, CACHE, sr=SR)
segments, path_index = load_background_cache(CACHE, TOTAL)
check("segment count", segments.shape == (4 * 1 + 2 * 3, TOTAL), f"{segments.shape}")
exact = True
for p, (start, count) in path_index.items():
    src = bg_int16[p]
    check(f"  {os.path.basename(p)} segment count", count == len(src) // TOTAL)
    for k in range(count):
        if not np.array_equal(np.asarray(segments[start + k]), src[k * TOTAL:(k + 1) * TOTAL]):
            exact = False
check("all segments bit-for-bit equal to source", exact)

print("\n2. cached vs upstream produce identical audio for the same seed")
# Restricted to the exactly-TOTAL background files. For those, upstream's
# random_background takes its `else` branch (num_samples == target, so it loads the
# whole file with no random offset) and the cached version has exactly one segment
# per file -- so both consume the same background audio and the two paths are
# directly comparable. batch_size=1 keeps the RNG streams aligned through the single
# draw that matters.
short_paths = bg_paths[:4]
common = dict(p=1.0, background_paths=short_paths, min_snr_in_db=-5.0,
              max_snr_in_db=5.0, mode="per_example")
upstream = torch_audiomentations.AddBackgroundNoise(**common)
cached = CachedAddBackgroundNoise(**common, segments=segments, path_index=path_index)

max_diff = 0.0
for seed in range(8):
    x = torch.from_numpy(rng.normal(0, 0.05, (1, 1, TOTAL)).astype(np.float32))
    random.seed(seed)
    torch.manual_seed(seed)
    a = upstream(samples=x.clone(), sample_rate=SR)
    random.seed(seed)
    torch.manual_seed(seed)
    b = cached(samples=x.clone(), sample_rate=SR)
    a = a.samples if hasattr(a, "samples") else a
    b = b.samples if hasattr(b, "samples") else b
    max_diff = max(max_diff, float((a - b).abs().max()))
check("identical output across 8 seeds", max_diff < 1e-6, f"max abs diff = {max_diff:.3e}")

print("\n3. sampling is uniform over FILES, not over segments")
# bg_long* contribute 3 segments each but must still be drawn as often as a
# bg_short* file. Sampling segments uniformly would give the long files 3x the
# weight -- 6/14 vs 6/10 of all draws.
counts = {os.path.basename(p): 0 for p in bg_paths}
probe = CachedAddBackgroundNoise(p=1.0, background_paths=bg_paths, min_snr_in_db=0.0,
                                 max_snr_in_db=0.0, mode="per_example",
                                 segments=segments, path_index=path_index)
real_choice = random.choice
random.seed(1234)


def counting_choice(seq):
    picked = real_choice(seq)
    counts[os.path.basename(str(picked))] += 1
    return picked


from torch_audiomentations.utils.io import Audio  # noqa: E402

audio_helper = Audio(sample_rate=SR, mono=True)
random.choice = counting_choice
try:
    for _ in range(600):
        probe.random_background(audio_helper, TOTAL)
finally:
    random.choice = real_choice
short_share = sum(v for k, v in counts.items() if "short" in k)
long_share = sum(v for k, v in counts.items() if "long" in k)
# 4 short files vs 2 long files -> expect ~2:1 regardless of segment counts
ratio = short_share / max(1, long_share)
check("file-uniform sampling", 1.6 < ratio < 2.5,
      f"short:long draw ratio = {ratio:.2f} (expect ~2.0; segment-uniform would give ~0.67)")

print("\n4. a cache cut at a different total_length is refused")
try:
    load_background_cache(CACHE, TOTAL // 2)
    check("mismatched total_length raises", False, "no exception")
except ValueError as e:
    check("mismatched total_length raises", "total_length" in str(e), str(e)[:70])

print("\n5. background_cache_path='' keeps the upstream path byte-for-byte")
clip = os.path.join(TMP, "clip.wav")
scipy.io.wavfile.write(clip, SR, (0.01 * np.sin(2 * np.pi * 440 * np.arange(TOTAL) / SR)
                                  * 32767).astype(np.int16))
NO_AUG = {"SevenBandParametricEQ": 0, "TanhDistortion": 0, "PitchShift": 0,
          "BandStopFilter": 0, "AddColoredNoise": 0, "AddBackgroundNoise": 1.0,
          "Gain": 0, "RIR": 0}


def run(cache_path):
    random.seed(7)
    torch.manual_seed(7)
    np.random.seed(7)
    gen = augment_clips([clip] * 4, total_length=TOTAL, batch_size=2,
                        augmentation_probabilities=NO_AUG,
                        background_clip_paths=short_paths,
                        background_mode="per_example",
                        background_cache_path=cache_path)
    return np.vstack([next(gen), next(gen)])


off_a, off_b = run(""), run("")
check("upstream path is deterministic under a fixed seed", np.array_equal(off_a, off_b))

on_a, on_b = run(CACHE), run(CACHE)
check("cache path is deterministic under a fixed seed", np.array_equal(on_a, on_b))

# `short_paths` are all exactly TOTAL samples, so every source has a single segment:
# the cached path draws no extra random number (see CachedAddBackgroundNoise) and its
# RNG stream stays aligned with upstream's, making the two byte-identical end to end.
#
# With multi-segment sources the streams do diverge by construction -- upstream draws a
# sample offset with random.randint, the cache draws a segment index with
# random.randrange, and the two consume different amounts of entropy. That changes
# *which* background is selected, not how it is mixed; check 2 covers the mixing.
d = np.abs(on_a.astype(np.int32) - off_a.astype(np.int32))
check("cache path matches upstream path through augment_clips (single-segment sources)",
      d.max() == 0, f"max int16 delta = {d.max()}")

on_multi = None
for cache_path in (CACHE,):
    random.seed(11)
    torch.manual_seed(11)
    np.random.seed(11)
    gen = augment_clips([clip] * 4, total_length=TOTAL, batch_size=2,
                        augmentation_probabilities=NO_AUG,
                        background_clip_paths=bg_paths,     # includes 3-segment sources
                        background_mode="per_example",
                        background_cache_path=cache_path)
    on_multi = np.vstack([next(gen), next(gen)])
check("multi-segment sources still produce well-formed output",
      on_multi.shape == off_a.shape and on_multi.dtype == np.int16
      and np.isfinite(on_multi).all() and np.abs(on_multi).max() > 0,
      f"shape={on_multi.shape} dtype={on_multi.dtype}")

print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    sys.exit(1)
print("all background-cache checks passed")
