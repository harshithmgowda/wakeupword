#!/usr/bin/env python
"""Regression test: make_tv_mix.py resume must be RNG-safe.

Reproduces the reviewer's finding: generate a few clips from scratch,
delete a couple, regenerate (resume), and assert the regenerated files
are byte-for-byte identical to what the from-scratch run produced at that
same index -- not merely "some valid clip".

With the buggy shared-rng implementation (one np.random.default_rng(seed)
walked across a for-loop that `continue`s on existing files), a skipped
index consumes no draws, so on resume the RNG stream position is behind
where a from-scratch run would be. The reviewer's exact repro: deleting
indices 3 and 4 of a 5-clip run and regenerating produced index 3
byte-identical to original index 0, and index 4 identical to original
index 1. This test asserts both that resumed content matches its own
original (primary check) and that it does not match an unrelated earlier
index (the specific failure signature).
"""
import hashlib, shutil, subprocess, sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent
SCRIPT = ROOT / "tools" / "data" / "make_tv_mix.py"
TMP = HERE / "_tmp_tv_mix"
SR = 16000
N_CLIPS = 5
DELETE_IDX = [3, 4]


def _write_tone(path, seconds, freq, amp):
    t = np.arange(int(seconds * SR)) / SR
    x = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, (x * 32767).astype(np.int16), SR, subtype="PCM_16")


def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _run(out_dir):
    p = subprocess.run(
        [sys.executable, str(SCRIPT), "--music", str(TMP / "music"),
         "--speech", str(TMP / "speech"), "--out", str(out_dir),
         "--clips", str(N_CLIPS), "--seed", "42"],
        cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    return p.stdout


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    (TMP / "music").mkdir(parents=True)
    (TMP / "speech").mkdir(parents=True)
    # Content doesn't matter (loud enough to clear MUSIC_MIN_RMS, short so
    # the test is fast) -- this exercises compose() through the real
    # CLI/resume path, not a hand-rolled unit call.
    for i in range(4):
        _write_tone(TMP / "music" / f"m{i}.wav", 6.0, 220 + i * 30, 0.2)
    for i in range(6):
        _write_tone(TMP / "speech" / f"s{i}.wav", 1.5, 440 + i * 50, 0.3)

    out = TMP / "out"
    _run(out)
    files = sorted(out.glob("*.wav"))
    assert len(files) == N_CLIPS, files
    original_md5 = {p.name: _md5(p) for p in files}

    for i in DELETE_IDX:
        (out / f"tv_42_{i:05d}.wav").unlink()
    assert len(list(out.glob("*.wav"))) == N_CLIPS - len(DELETE_IDX)

    _run(out)  # resume
    files = sorted(out.glob("*.wav"))
    assert len(files) == N_CLIPS, files
    resumed_md5 = {p.name: _md5(p) for p in files}

    for name, digest in original_md5.items():
        assert resumed_md5[name] == digest, (
            f"{name}: resumed content does not match a from-scratch run "
            f"at this index (RNG desync on resume) -- "
            f"got {resumed_md5[name]}, want {digest}")

    # The reviewer's specific failure signature: a regenerated clip must
    # not silently become a copy of an *earlier*, unrelated index.
    for i in DELETE_IDX:
        name = f"tv_42_{i:05d}.wav"
        for j in range(N_CLIPS):
            if j == i:
                continue
            other = f"tv_42_{j:05d}.wav"
            assert resumed_md5[name] != original_md5[other], (
                f"{name} content matches unrelated clip {other} "
                f"-- RNG stream desync on resume")

    shutil.rmtree(TMP)
    print("PASS test_make_tv_mix")


if __name__ == "__main__":
    main()
