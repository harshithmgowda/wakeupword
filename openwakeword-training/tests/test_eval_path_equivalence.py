#!/usr/bin/env python
"""Equivalence guard for the two eval-path speedups (plan items 1 and 2).

Both changes are pure scheduling changes and MUST NOT move a single bit of any
committed eval number.  This test proves that on REAL audio from this repo, not
on synthetic arrays:

  1. media_utils.embed(..., ncpu=K) -- embed_clips' ThreadPool width -- is
     bit-identical for every K.  (The dangerous same-named knob,
     AudioFeatures(ncpu=K), is asserted NOT to be used by embed().)
  2. compare_models.load()'s num_threads choice does not change head scores:
     the previous default (os.cpu_count()) and the new one produce
     np.array_equal outputs over thousands of real windows.

Run:  ./.venv/bin/python tests/test_eval_path_equivalence.py

Part 1 runs in-process (.venv).  Part 2 needs ai_edge_litert, which only exists
in .venv-onnx2tf, so it is executed there via a subprocess -- the two venvs are
never mixed.  If .venv-onnx2tf is missing, part 2 is skipped
loudly rather than silently passing.
"""
import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from repo_paths import DATA_ROOT  # noqa: E402
os.chdir(ROOT)

from media_utils import EMBED_NCPU, embed, read_wav16k, windows16  # noqa: E402

# Real clips, in preference order; the first directory that exists is used.
# eval_media/music comes first because its clips are 30 s each, so a handful of
# them yield thousands of sliding windows for the head-equivalence half -- the
# ~3 s recordings in eval/ yield only ~3 windows apiece, which is too thin a
# sample to call a thread-count change "bit-identical".
CLIP_DIRS = [
    ROOT / "eval_media" / "music",
    DATA_ROOT / "eval",
]


def pick_clips(n=8):
    for d in CLIP_DIRS:
        if d.is_dir():
            files = sorted(p for p in d.iterdir()
                           if p.suffix.lower() in {".wav", ".flac"})[:n]
            if files:
                return files
    raise SystemExit("no real clip corpus found -- cannot run an honest "
                     "equivalence test on synthetic data")


# ---------------------------------------------------------------- part 0
# embed() must never route its ncpu into the AudioFeatures constructor.
src = inspect.getsource(embed)
assert "embed_clips(" in src, src
assert "AudioFeatures(" not in src, (
    "media_utils.embed must not construct AudioFeatures with ncpu -- that knob "
    "sets onnxruntime intra_op/inter_op threads and CHANGES feature values")
print("PASS embed() uses the ThreadPool ncpu, not the onnxruntime one")


# ---------------------------------------------------------------- part 1
import openwakeword.utils as oww_utils  # noqa: E402

files = pick_clips()
audio = [read_wav16k(f) for f in files]
secs = sum(len(a) for a in audio) / 16000.0

# device="cpu" so this test is reproducible on a box without a GPU; the ncpu
# invariance being proved is a property of the ThreadPool, not the provider.
feat = oww_utils.AudioFeatures(device="cpu")

ref = [embed(feat, a, ncpu=1) for a in audio]
for k in (2, 4, EMBED_NCPU, 24):
    got = [embed(feat, a, ncpu=k) for a in audio]
    for f, r, g in zip(files, ref, got):
        assert r.shape == g.shape, (f.name, r.shape, g.shape)
        assert np.array_equal(r, g), (
            f"embed_clips ncpu={k} changed {f.name}: "
            f"max_abs_diff={np.abs(r - g).max()}")
print(f"PASS embed_clips ncpu invariance: {len(files)} real clips "
      f"({secs:.1f}s), bit-identical at ncpu in 1/2/4/{EMBED_NCPU}/24")

# The default really is the value we measured, not something else.
assert inspect.signature(embed).parameters["ncpu"].default == EMBED_NCPU
assert EMBED_NCPU > 1, "the whole point of item 2 is that the default is >1"

emb_ref = ref  # reuse for part 2 so the subprocess scores real windows


# ---------------------------------------------------------------- part 2
def run_tflite_half():
    py = ROOT / ".venv-onnx2tf" / "bin" / "python"
    if not py.exists():
        print("SKIP tflite half: .venv-onnx2tf not present")
        return
    heads = [p for p in [ROOT / "my_custom_model_v7" / "ouh_ke-i_nahbu.tflite",
                         ROOT / "stock_models" / "ok_nabu_v0.1.tflite"]
             if p.exists()]
    if not heads:
        print("SKIP tflite half: no head found")
        return

    wins = np.concatenate([windows16(e) for e in emb_ref if len(e) >= 16])
    npy = ROOT / "tests" / "_tmp_eval_path_windows.npy"
    np.save(npy, np.ascontiguousarray(wins, dtype=np.float32))

    # Plain (non-f) template: the child takes root, windows file and heads on
    # argv, so no interpolation happens inside the source text.
    code = textwrap.dedent("""
        import os, sys
        import numpy as np
        root, npy = sys.argv[1], sys.argv[2]
        heads = sys.argv[3:]
        os.chdir(root)
        sys.path.insert(0, root)
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import compare_models as cm
        from ai_edge_litert.interpreter import Interpreter

        w = np.load(npy)
        assert w.ndim == 3 and w.shape[1:] == (16, 96), w.shape
        for head in heads:
            new = cm.predict(cm.load(head), w)
            for nt in (os.cpu_count(), 8, 1):
                it = Interpreter(model_path=head, num_threads=nt)
                it.allocate_tensors()
                old = cm.predict(it, w)
                assert np.array_equal(old, new), (
                    head, nt, float(np.abs(old - new).max()))
            print("  PASS", os.path.basename(head), len(w),
                  "real windows: cm.load == num_threads in",
                  (os.cpu_count(), 8, 1))
    """)
    r = subprocess.run([str(py), "-c", code, str(ROOT), str(npy)]
                       + [str(h) for h in heads],
                       capture_output=True, text=True)
    npy.unlink(missing_ok=True)
    sys.stdout.write("\n".join(l for l in r.stdout.splitlines()
                               if not l.startswith("INFO:")) + "\n")
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit("tflite thread-count equivalence FAILED")
    print("PASS compare_models.load thread count does not change head scores")


run_tflite_half()
print("PASS test_eval_path_equivalence")
