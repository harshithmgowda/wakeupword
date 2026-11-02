#!/usr/bin/env python
"""The augment stage must not treat a PARTIAL feature set as finished.

The stage writes four feature files (positive/negative x train/test), largest
first. Upstream gated the whole stage on `positive_features_train.npy` alone, so
an augment run interrupted after the first file landed was considered complete
forever: every later run logged "Openwakeword features already exist, skipping"
and did nothing, and the three missing files only surfaced much later in the
train stage as `FileNotFoundError: positive_features_test.npy`.

That is not a theoretical failure -- it happened during the v7 reproduction on
2026-07-30 and cost a full augment run. It is also invisible to a size check: the
memmaps are sized upfront from `n_total`, so a half-written file has exactly the
size a complete one would.

This drives the real `train.py --augment_clips` in a subprocess:

  1. run on an empty feature dir            -> all four files appear
  2. delete three of them (the interrupted-run state) and run again
                                            -> all four are regenerated

Step 2 is the regression: unpatched, it logs "already exist, skipping", exits 0,
and leaves three files missing.

Run with ./.venv/bin/python (needs torch, openwakeword and the ONNX feature
extractors that setup_env.sh downloads).
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import yaml

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = ROOT / "openwakeword" / "openwakeword" / "train.py"

FEATURE_NAMES = ("positive_features_train.npy", "positive_features_test.npy",
                 "negative_features_train.npy", "negative_features_test.npy")
SR = 16000
N_CLIPS = 8


def _wavs(d, n, seconds, rng):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        y = (rng.normal(0, 0.1, int(seconds * SR)) * 32767).astype(np.int16)
        scipy.io.wavfile.write(str(d / f"{i:03d}.wav"), SR, y)


def build_workspace(tmp):
    """A minimal but REAL training workspace: four clip dirs, backgrounds, RIRs."""
    rng = np.random.default_rng(0)
    out = tmp / "model_out"
    md = out / "wakeword"
    for name in ("positive_train", "positive_test", "negative_train", "negative_test"):
        _wavs(md / name, N_CLIPS, 2.0, rng)
    _wavs(tmp / "backgrounds", 3, 10.0, rng)
    _wavs(tmp / "rirs", 3, 0.5, rng)

    config = {
        "model_name": "wakeword",
        "output_dir": str(out),
        "piper_sample_generator_path": str(ROOT / "piper-sample-generator"),
        "total_length": 32000,
        "augmentation_batch_size": 4,
        "augmentation_rounds": 1,
        "background_paths": [str(tmp / "backgrounds")],
        "background_paths_duplication_rate": [1],
        "rir_paths": [str(tmp / "rirs")],
        "target_phrase": ["test phrase"],
        "n_samples": N_CLIPS,
        "n_samples_val": N_CLIPS,
        "tts_batch_size": 2,
    }
    cfg = tmp / "config.yaml"
    cfg.write_text(yaml.safe_dump(config))
    return cfg, md


def run_augment(cfg):
    p = subprocess.run(
        [sys.executable, str(TRAIN_PY), "--training_config", str(cfg), "--augment_clips"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    return p


def main():
    tmp = Path(tempfile.mkdtemp(prefix="oww_resume_gate_"))
    try:
        cfg, md = build_workspace(tmp)

        # --- 1. a clean run produces all four -------------------------------
        p = run_augment(cfg)
        assert p.returncode == 0, f"first augment failed:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}"
        present = [n for n in FEATURE_NAMES if (md / n).exists()]
        assert len(present) == 4, f"clean run produced only {present}"
        print("PASS clean augment run writes all four feature files")

        # --- 2. simulate an interrupted run ---------------------------------
        # Keep the file upstream checked for, drop the three written after it.
        for n in FEATURE_NAMES[1:]:
            (md / n).unlink()
        kept_mtime = (md / FEATURE_NAMES[0]).stat().st_mtime

        p = run_augment(cfg)
        assert p.returncode == 0, f"resume augment failed:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}"
        log = p.stdout + p.stderr

        missing = [n for n in FEATURE_NAMES if not (md / n).exists()]
        assert not missing, (
            "REGRESSION: the augment stage skipped on a partial feature set and left "
            f"{missing} missing. This is the upstream bug: the gate must require all "
            "four feature files, not just positive_features_train.npy.\n"
            f"--- log tail ---\n{log[-2000:]}"
        )
        assert "already exist, skipping" not in log, (
            "REGRESSION: stage reported 'already exist, skipping' with three of four "
            f"feature files missing.\n--- log tail ---\n{log[-2000:]}"
        )
        print("PASS partial feature set is regenerated rather than skipped")

        assert "incomplete feature set" in log.lower(), (
            "the incomplete-set warning should tell the user why work is being "
            f"redone.\n--- log tail ---\n{log[-2000:]}"
        )
        print("PASS the incomplete-set warning is emitted")

        assert (md / FEATURE_NAMES[0]).stat().st_mtime >= kept_mtime, \
            "the surviving file should have been rewritten, not left stale"
        print("PASS test_augment_resume_gate")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
