#!/usr/bin/env python
"""Guards for the three piper-sample-generator patches.

  item 3 -- the dead `np.trim_zeros` at the top of the write loop is gone and
            must not come back (it was 60-77% of TTS wall clock)
  item 8 -- remove_silence() is vectorised and BIT-IDENTICAL to the original
            list-building implementation on real generated clips
  item 9 -- auto_reduce_batch_size actually works: forcing an OOM used to die
            with `IndexError: index N is out of bounds` because the per-item
            bookkeeping was not truncated with the batch

Run:  ./.venv/bin/python tests/test_piper_generate_samples.py
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "piper-sample-generator"))
os.chdir(ROOT / "piper-sample-generator")

import generate_samples as gs  # noqa: E402

SRC = (ROOT / "piper-sample-generator" / "generate_samples.py").read_text()


# ------------------------------------------------------------------ item 3
# The dead value: `audio_data = np.trim_zeros(...)` whose result was
# immediately overwritten from the untrimmed source array.
code_lines = [ln for ln in SRC.splitlines()
              if not ln.lstrip().startswith("#")]
assert not any("np.trim_zeros" in ln for ln in code_lines), (
    "np.trim_zeros is back in generate_samples.py. It was dead code (the very "
    "next statement reassigns audio_data from the untrimmed array) and cost "
    "151.3 ms/clip at batch 100. If it is ever needed again its result must "
    "actually be used, and it must not be a Python element loop.")
print("PASS item 3: dead np.trim_zeros stays deleted")


# ------------------------------------------------------------------ item 8
def remove_silence_original(x, frame_duration=0.030, sample_rate=16000,
                            min_start=2000):
    """Verbatim upstream implementation, kept here as the equivalence oracle."""
    import webrtcvad
    vad = webrtcvad.Vad(0)
    if x.dtype in (np.float32, np.float64):
        x = (x * 32767).astype(np.int16)
    x_new = x[0:min_start].tolist()
    step_size = int(sample_rate * frame_duration)
    for i in range(min_start, x.shape[0] - step_size, step_size):
        vad_res = vad.is_speech(x[i:i + step_size].tobytes(), sample_rate)
        if vad_res:
            x_new.extend(x[i:i + step_size].tolist())
    return np.array(x_new).astype(np.int16)


import soundfile as sf  # noqa: E402

CLIP_DIRS = [ROOT / "positives_es", ROOT / "tv_mix_speech",
             ROOT / "positives_lombard", ROOT / "tv_mix_speech_eval"]
clips = []
for d in CLIP_DIRS:
    if d.is_dir():
        for f in sorted(d.glob("*.wav"))[:60]:
            a, _ = sf.read(f, dtype="int16")
            if a.ndim > 1:
                a = a[:, 0]
            clips.append((f.name, np.ascontiguousarray(a)))
assert len(clips) >= 100, (
    f"only {len(clips)} real clips found -- refusing to certify bit-identity "
    "on a thin sample")

for name, x in clips:
    ref = remove_silence_original(x)
    got = gs.remove_silence(x)
    assert ref.dtype == got.dtype == np.int16, (name, ref.dtype, got.dtype)
    assert ref.shape == got.shape, (name, ref.shape, got.shape)
    assert np.array_equal(ref, got), name

# degenerate inputs the vectorised concatenate path must still handle
for x in (np.zeros(0, dtype=np.int16),
          np.zeros(500, dtype=np.int16),
          np.zeros(2000, dtype=np.int16),
          np.arange(2100, dtype=np.int16)):
    ref, got = remove_silence_original(x), gs.remove_silence(x)
    assert ref.shape == got.shape and np.array_equal(ref, got), x.shape
    assert got.dtype == np.int16
# float input takes the (x * 32767) branch in both
xf = (np.arange(5000, dtype=np.float32) % 1000) / 32768.0
assert np.array_equal(remove_silence_original(xf), gs.remove_silence(xf))
print(f"PASS item 8: remove_silence bit-identical on {len(clips)} real clips "
      f"+ 5 edge cases")


# ------------------------------------------------------------------ item 9
def test_oom_retry():
    """Force the reduce-batch path and assert the run COMPLETES.

    Before the patch this raised
        IndexError: index 33 is out of bounds for dimension 0 with size 33
    from the `for i, clip_phoneme_index in enumerate(clip_indexes_by_batch)`
    loop, because the retry shrank the audio batch but left that list at its
    original length. auto_reduce_batch_size=True (which gen_tts_speech.py
    passes) therefore had never been a working safety net.
    """
    try:
        import torch
    except ImportError:
        print("SKIP item 9: no torch")
        return
    if not torch.cuda.is_available():
        print("SKIP item 9: no CUDA")
        return
    model = ROOT / "piper-sample-generator" / "models" / "en_US-libritts_r-medium.pt"
    if not model.exists():
        print("SKIP item 9: piper model not present")
        return

    out = Path(tempfile.mkdtemp(prefix="oww_piper_oom_"))
    try:
        n, batch = 24, 24
        texts = ["hola que tal esto es una prueba de sintesis de voz larga"] * n
        names = [f"o_{i:03d}.wav" for i in range(n)]
        # A fraction small enough that batch=24 cannot fit, forcing >=1 halving.
        # Without cuda_memory_fraction this path is UNREACHABLE on WSL2: the
        # driver satisfies the allocation from host memory and no OOM is raised.
        gs.generate_samples(text=texts, output_dir=str(out), max_samples=n,
                            batch_size=batch, auto_reduce_batch_size=True,
                            file_names=names, cuda_memory_fraction=0.02)
        wrote = sorted(out.glob("*.wav"))
        assert len(wrote) >= 1, "retry path produced no clips"
        for w in wrote:
            a, sr = sf.read(w, dtype="int16")
            assert sr == 16000 and a.size > 0, (w, sr, a.shape)
        print(f"PASS item 9: forced OOM retry completed, {len(wrote)} valid "
              f"clips written (previously IndexError)")
    finally:
        shutil.rmtree(out, ignore_errors=True)
        torch.cuda.set_per_process_memory_fraction(1.0)
        torch.cuda.empty_cache()


test_oom_retry()
print("PASS test_piper_generate_samples")
