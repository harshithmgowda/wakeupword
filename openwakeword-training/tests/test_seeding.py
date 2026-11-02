#!/usr/bin/env python
"""The optional `seed:` config key must actually pin the augment stage.

Nothing in this pipeline was seeded: `train.py` set no seed at all, so clip
generation, augmentation draws and the training loop varied run to run and two
runs of one config were two samples rather than one result. That made this
project's own reproduction attempt judgeable only through the eval harness, never
by comparing artifacts (write-up on the experiments branch).

`seed: <int>` in the training config now seeds python, numpy and torch. This test
holds it to two claims, both end-to-end through the real
`train.py --augment_clips`:

  1. WITH a seed, two runs produce byte-identical feature files.
  2. WITHOUT one, they do not -- i.e. the default is genuinely still unseeded,
     so every model shipped before this patch is still described correctly as
     the output of an unseeded pipeline.

Claim 2 matters as much as claim 1: silently seeding by default would change
what an existing config produces.

Note this covers the augment stage. Bit-identical *training* additionally depends
on cuDNN/cuBLAS kernel selection, which seeding alone does not pin down.

Run with ./.venv/bin/python.
"""
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_augment_resume_gate import (  # noqa: E402
    FEATURE_NAMES, build_workspace, run_augment,
)


def digest(model_dir):
    h = hashlib.sha256()
    for name in FEATURE_NAMES:
        h.update(name.encode())
        h.update((model_dir / name).read_bytes())
    return h.hexdigest()


def augment_once(seed):
    """One full augment run in a throwaway workspace; returns the feature digest."""
    tmp = Path(tempfile.mkdtemp(prefix="oww_seed_"))
    try:
        cfg, md = build_workspace(tmp)
        if seed is not None:
            config = yaml.safe_load(cfg.read_text())
            config["seed"] = seed
            cfg.write_text(yaml.safe_dump(config))
        p = run_augment(cfg)
        assert p.returncode == 0, f"augment failed:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}"
        missing = [n for n in FEATURE_NAMES if not (md / n).exists()]
        assert not missing, f"augment did not write {missing}"
        return digest(md)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    a = augment_once(seed=1234)
    b = augment_once(seed=1234)
    assert a == b, (
        "seed: 1234 did not pin the augment stage -- two runs produced different "
        f"feature files ({a[:16]} vs {b[:16]})"
    )
    print("PASS same seed -> byte-identical feature files")

    c = augment_once(seed=9999)
    assert c != a, (
        "a different seed produced identical output, which means the seed is not "
        "reaching the augmentation draws at all"
    )
    print("PASS different seed -> different feature files")

    d = augment_once(seed=None)
    e = augment_once(seed=None)
    assert d != e, (
        "two UNSEEDED runs were identical. Either the default silently became "
        "seeded -- which would change what every existing config produces -- or "
        "this workspace is too small to expose any randomness."
    )
    print("PASS no seed -> still unseeded, as before the patch")
    print("PASS test_seeding")


if __name__ == "__main__":
    main()
