#!/usr/bin/env python
"""Guard for tools/data/inject_real_clips.py.

Three of this project's model generations carried real recordings duplicated into
positive_train, but no script ever performed that step -- so re-running
`04_train.sh` against one of those configs silently built a 50,000-positive model
that was not the model the config named. The injector makes it declarative; this
checks the three things that would make it silently wrong again:

  * the `only` / `only_split` filter selects the right subset,
  * REAL_MUSIC_* is not double-counted as REAL_* (the prefixes overlap),
  * re-running is a no-op rather than a second round of duplication.

Synthetic clips only, so this needs no recording corpus. Run with
./.venv/bin/python.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import yaml

ROOT = Path(__file__).resolve().parent.parent
INJECT = ROOT / "tools" / "data" / "inject_real_clips.py"
SR = 16000


def utterance(path, rng):
    """Quiet room tone with a tone burst in the middle, so the trimmer has
    something to find."""
    y = rng.normal(0, 0.001, int(2.5 * SR))
    t = np.arange(int(0.6 * SR)) / SR
    y[SR:SR + len(t)] += 0.3 * np.sin(2 * np.pi * 300 * t)
    scipy.io.wavfile.write(str(path), SR, (y * 32767).astype(np.int16))


def run(cfg):
    p = subprocess.run([sys.executable, str(INJECT), "--config", str(cfg)],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert p.returncode == 0, f"injector failed:\n{p.stdout}\n{p.stderr}"
    return p.stdout


def counts(pt):
    real_music = sorted(pt.glob("REAL_MUSIC_*.wav"))
    real = [p for p in pt.glob("REAL_*.wav") if not p.name.startswith("REAL_MUSIC_")]
    other = [p for p in pt.glob("*.wav")
             if not p.name.startswith("REAL_")]
    return len(other), len(real), len(real_music)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="oww_inject_"))
    try:
        rng = np.random.default_rng(0)
        plain, music = tmp / "src_plain", tmp / "src_music"
        for d, n in ((plain, 5), (music, 6)):
            d.mkdir(parents=True)
            for i in range(n):
                utterance(d / f"clip_{i:02d}.wav", rng)

        # only 2 of the 6 music clips are on the "train" side of the split
        split = tmp / "split.tsv"
        split.write_text(
            "# comment line that must be ignored\n"
            "clip_00.wav\ttrain\nclip_01.wav\theldout\nclip_02.wav\ttrain\n"
            "clip_03.wav\theldout\nclip_04.wav\theldout\nclip_05.wav\theldout\n"
        )

        pt = tmp / "out" / "wakeword" / "positive_train"
        pt.mkdir(parents=True)
        for i in range(4):                      # stand-ins for synthetic positives
            utterance(pt / f"synth_{i}.wav", rng)

        cfg = tmp / "cfg.yaml"
        cfg.write_text(yaml.safe_dump({
            "model_name": "wakeword",
            "output_dir": str(tmp / "out"),
            "real_clip_sources": [
                {"dir": str(plain), "prefix": "REAL", "copies": 3},
                {"dir": str(music), "prefix": "REAL_MUSIC", "copies": 3,
                 "only": str(split), "only_split": "train"},
            ],
            "expect_positive_train_total": 4 + 5 * 3 + 2 * 3,
        }))

        run(cfg)
        synth, real, real_music = counts(pt)
        assert (synth, real, real_music) == (4, 15, 6), \
            f"expected (4, 15, 6), got ({synth}, {real}, {real_music})"
        print("PASS only_split selects the right subset, prefixes do not overlap")

        out = run(cfg)
        assert "already present, skipping" in out, "second run was not idempotent"
        assert counts(pt) == (4, 15, 6), f"re-run changed the set: {counts(pt)}"
        print("PASS re-running injects nothing")

        # a wrong expected total must fail loudly rather than train on it
        bad = yaml.safe_load(cfg.read_text())
        bad["expect_positive_train_total"] = 999
        cfg.write_text(yaml.safe_dump(bad))
        p = subprocess.run([sys.executable, str(INJECT), "--config", str(cfg), "--check"],
                           capture_output=True, text=True, cwd=str(ROOT))
        assert p.returncode != 0 and "FAILED" in p.stdout + p.stderr, \
            "a mismatched expect_positive_train_total did not fail the check"
        print("PASS a wrong expected total fails the check")
        print("PASS test_inject_real_clips")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
