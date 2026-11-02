#!/usr/bin/env python
"""compute_features_from_generator must not depend on FEATURE_BATCH_SIZE.

Plan item 6 decouples the feature-extraction batch from the augmentation batch
by buffering already-augmented batches before calling embed_clips. That is only
legitimate if the memmap it writes is BIT-IDENTICAL to what upstream's
one-call-per-augmentation-batch loop produced -- same values, same row order,
same row count.

The augmentation itself is unseeded, so this feeds BOTH runs the SAME
pre-augmented buffers (captured once from the real augment_clips on real
clips). That removes augmentation randomness as a confound: any difference
observed is the batching change and nothing else.

Run:  ./.venv/bin/python tests/test_feature_batching.py
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "openwakeword"))
os.chdir(ROOT)

import openwakeword.data as D  # noqa: E402
import openwakeword.utils as U  # noqa: E402

assert hasattr(U, "FEATURE_BATCH_SIZE"), "the buffering knob went missing"
assert U.FEATURE_BATCH_SIZE > 1, "FEATURE_BATCH_SIZE=1 is the un-optimised path"

AUG_BATCH = 32
TOTAL_LENGTH = 32000
N_BATCHES = 11          # deliberately not a multiple of 256/32, to exercise the
                        # partial trailing flush


def real_clip_paths(n):
    """Real positive clips from whichever model directory is present."""
    for d in sorted(ROOT.glob("my_custom_model_v*/*/positive_train")):
        files = sorted(d.glob("*.wav"))[:n]
        if len(files) >= n:
            return [str(p) for p in files]
    raise SystemExit("no real positive clip pool found")


clips = real_clip_paths(AUG_BATCH * N_BATCHES)
rirs = [str(p) for p in (ROOT / "mit_rirs").rglob("*.wav")]
bg = [str(p) for p in sorted((ROOT / "fma").glob("*.wav"))[:200]]

# Capture real augmented audio ONCE. Both runs replay these exact buffers.
gen = D.augment_clips(clips, total_length=TOTAL_LENGTH, batch_size=AUG_BATCH,
                      RIR_paths=rirs, background_clip_paths=bg,
                      background_cache_path=str(ROOT / "background_cache.npy"))
buffers = [next(gen) for _ in range(N_BATCHES)]
n_rows = sum(b.shape[0] for b in buffers)
assert n_rows == AUG_BATCH * N_BATCHES, n_rows
print(f"captured {N_BATCHES} real augmented batches = {n_rows} clips "
      f"({buffers[0].dtype}, {buffers[0].shape[1]} samples)")

tmp = Path(tempfile.mkdtemp(prefix="oww_featbatch_"))
try:
    def run(feature_batch_size, n_total, name):
        old = U.FEATURE_BATCH_SIZE
        U.FEATURE_BATCH_SIZE = feature_batch_size
        try:
            out = tmp / f"{name}.npy"
            U.compute_features_from_generator(
                iter([b.copy() for b in buffers]), n_total=n_total,
                clip_duration=TOTAL_LENGTH, output_file=str(out),
                device="cpu", ncpu=4)
            return np.load(out)
        finally:
            U.FEATURE_BATCH_SIZE = old

    # FEATURE_BATCH_SIZE=1 -> max(batch_size, 1) == batch_size == one embed_clips
    # call per augmentation batch, i.e. upstream's exact loop.
    for n_total, label in ((n_rows, "exact"),
                           (n_rows - 50, "n_total below the row count"),
                           (n_rows + 200, "n_total above the row count")):
        ref = run(1, n_total, f"ref_{n_total}")
        for fbs in (64, 256, 1024):
            got = run(fbs, n_total, f"got_{n_total}_{fbs}")
            assert ref.shape == got.shape, (label, fbs, ref.shape, got.shape)
            assert np.array_equal(ref, got), (
                f"{label}: FEATURE_BATCH_SIZE={fbs} changed the features, "
                f"max_abs_diff={float(np.abs(ref - got).max())}")
        print(f"  PASS {label:<28} rows={ref.shape[0]:<5} "
              f"bit-identical at FEATURE_BATCH_SIZE 1/64/256/1024")

    # Row ORDER, not just row content: the memmap must stay in generator order,
    # because trim_mmap and every downstream split assume front-to-back fill.
    ref = run(1, n_rows, "order_ref")
    got = run(256, n_rows, "order_got")
    assert np.array_equal(ref[0], got[0]) and np.array_equal(ref[-1], got[-1])
    # and a shuffled version must NOT compare equal -- proves the check has teeth
    shuf = ref.copy()
    shuf[[0, -1]] = shuf[[-1, 0]]
    assert not np.array_equal(shuf, got), "row-order check is vacuous"
    print("  PASS row order preserved (and the order check is not vacuous)")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("PASS test_feature_batching")
