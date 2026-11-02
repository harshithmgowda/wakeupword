#!/usr/bin/env python
"""openwakeword.data._load_audio must be bit-identical to torchaudio.load.

This is the guard for the augment-stage clip loader (plan item 4). The whole
justification for the swap is that libsndfile is what torchaudio is already
using underneath -- so anything less than `torch.equal` invalidates it, and
with it every feature array built after the change.

It is deliberately run over EVERY corpus shape the augment stage actually
reads -- positives, tv_mix, the FMA / audioset background pools and the RIRs --
not just the positives, because shape/dtype handling differs for multichannel,
non-PCM_16 and non-16 kHz files and those pools are where such files live.

Run:  ./.venv/bin/python tests/test_clip_loader_equivalence.py
"""
import os
import random
import sys
import time
from pathlib import Path

import torch
import torchaudio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "openwakeword"))
sys.path.insert(0, str(ROOT))          # for repo_paths and the shared modules
from repo_paths import DATA_ROOT  # noqa: E402
os.chdir(ROOT)

from openwakeword.data import _load_audio  # noqa: E402

# (label, directory, how many files to sample)
POOLS = [
    ("positives (v7 train)", ROOT / "my_custom_model_v7" / "ouh_ke-i_nahbu" / "positive_train", 60),
    ("positives_es", ROOT / "positives_es", 40),
    ("positives_lombard", ROOT / "positives_lombard", 40),
    ("tv_mix", ROOT / "tv_mix", 40),
    ("tv_mix_speech", ROOT / "tv_mix_speech", 40),
    ("fma", ROOT / "fma", 40),
    ("audioset_16k", ROOT / "audioset_16k", 40),
    ("mit_rirs", ROOT / "mit_rirs", 40),
    ("eval recordings", DATA_ROOT / "eval", 27),
]

AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3"}
rng = random.Random(0)

checked = 0
pools_seen = 0
t_ta = t_sf = 0.0
for label, d, n in POOLS:
    if not d.is_dir():
        print(f"  skip {label}: {d} not present")
        continue
    files = sorted(p for p in d.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    if not files:
        print(f"  skip {label}: no audio files")
        continue
    rng.shuffle(files)
    files = files[:n]

    for f in files:
        t = time.perf_counter()
        ref, ref_sr = torchaudio.load(str(f))
        t_ta += time.perf_counter() - t
        t = time.perf_counter()
        got, got_sr = _load_audio(str(f))
        t_sf += time.perf_counter() - t

        assert ref_sr == got_sr, (f, ref_sr, got_sr)
        assert ref.dtype == got.dtype == torch.float32, (f, ref.dtype, got.dtype)
        assert ref.shape == got.shape, (f, tuple(ref.shape), tuple(got.shape))
        assert torch.equal(ref, got), (
            f"{f}: loaders disagree, max_abs_diff="
            f"{float((ref - got).abs().max())}")
    checked += len(files)
    pools_seen += 1
    print(f"  PASS {label:<22} {len(files):>3} files torch.equal")

assert pools_seen >= 4, (
    f"only {pools_seen} corpus pools available -- this test's value is that it "
    "covers heterogeneous shapes; refusing to certify on fewer")
assert checked >= 150, f"only {checked} files checked"
print(f"PASS _load_audio bit-identical on {checked} real files across "
      f"{pools_seen} corpus pools "
      f"(torchaudio {t_ta/checked*1000:.3f} ms/clip, "
      f"soundfile {t_sf/checked*1000:.3f} ms/clip, "
      f"{t_ta/t_sf:.2f}x)")


# The fallback must exist and must actually work on something libsndfile
# cannot open, otherwise a future corpus format silently breaks the stage.
import inspect  # noqa: E402
src = inspect.getsource(_load_audio)
assert "except" in src and "torchaudio.load" in src, (
    "_load_audio lost its torchaudio fallback")

import tempfile  # noqa: E402
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
    tf.write(b"not an audio file at all")
    bogus = tf.name
try:
    raised = False
    try:
        _load_audio(bogus)
    except Exception:
        raised = True  # torchaudio also failed -- fine, it propagated
    assert raised, "a corrupt file should not silently return data"
finally:
    os.unlink(bogus)
print("PASS _load_audio keeps a working torchaudio fallback")
print("PASS test_clip_loader_equivalence")
