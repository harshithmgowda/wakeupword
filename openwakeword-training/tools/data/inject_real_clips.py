#!/usr/bin/env python
"""Add real recordings to a config's positive_train set, between clip generation
and augmentation.

WHY THIS EXISTS. Three of this project's model generations were not `04_train.sh`
runs of their own configs. Each also carried real recordings duplicated into the
positive set -- 21,412 copies of 106 recordings, plus 2,828 copies of 14
over-music recordings for two of them. Those entered the pipeline by inheritance
(each generation was built on a copy of the previous one's clip directories) and
no script ever performed the step, so re-running `04_train.sh` against such a
config produced a 50,000-positive model that looked like a successful build and
was not one -- silently, with no error.

This makes the step declarative. A config that lists `real_clip_sources` gets
them, one that does not is unaffected:

    real_clip_sources:
      - dir: "${OWW_DATA_ROOT}/train"      # every recording in the directory
        prefix: REAL
        copies: 202
      - dir: "${OWW_DATA_ROOT}/unused/excluded/music"
        prefix: REAL_MUSIC
        copies: 202
        only: data/v7_music_split.tsv      # restrict to a subset...
        only_split: train                  # ...the rows whose 2nd column is this
    expect_positive_train_total: 74240     # optional; asserted if present

(`data/v7_music_split.tsv` is kept as a worked example of the `only` format.)

Each source is trimmed with the same tight, variable-length cut `trim_clips.py
--training` performs (augment_clips does its own padding), then copied `copies`
times as `<prefix>_<stem>_<NNN>.wav`. The duplication is what makes the
augmentation meaningful: every copy draws its own background and RIR, so with
`augmentation_rounds: 2` one recording yields 404 distinct variants.

Idempotent -- an already-populated prefix is left alone, so this is safe to run
on every `04_train.sh` invocation and safe to re-run after an interruption.

    ./.venv/bin/python tools/data/inject_real_clips.py --config my_model.yaml
    ./.venv/bin/python tools/data/inject_real_clips.py --config my_model.yaml --check
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from repo_paths import DATA_ROOT, REPO_ROOT  # noqa: E402

TRIM = REPO_ROOT / "tools" / "recordings" / "trim_clips.py"


def resolve(value):
    """Expand ${OWW_DATA_ROOT}, ~ and repo-relative paths in a config path."""
    os.environ.setdefault("OWW_DATA_ROOT", str(DATA_ROOT))
    p = Path(os.path.expandvars(str(value))).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


def stems_for(source):
    """The subset of `dir` this source selects, or None for 'everything'."""
    only = source.get("only")
    if not only:
        return None
    path = resolve(only)
    if not path.is_file():
        raise SystemExit(f"real_clip_sources: 'only' file not found: {path}")

    split = source.get("only_split")
    stems = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if split is not None:
            if len(fields) < 2 or fields[1].strip() != split:
                continue
        stems.append(Path(fields[0].strip()).stem)
    if not stems:
        raise SystemExit(
            f"real_clip_sources: {path} selected no clips"
            + (f" for only_split={split!r}" if split else "")
        )
    return stems


def trim_source(src_dir, stems, dest):
    """Tight variable-length trim into `dest` (skipped if already populated)."""
    if dest.is_dir() and any(dest.glob("*.wav")):
        return len(list(dest.glob("*.wav")))
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(TRIM), "--training", str(src_dir), str(dest)]
    tmp = None
    if stems is not None:
        tmp = Path(tempfile.mkstemp(suffix=".txt")[1])
        tmp.write_text("\n".join(stems) + "\n")
        cmd += ["--only", str(tmp)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise SystemExit(f"trim_clips.py failed for {src_dir}:\n{p.stdout}\n{p.stderr}")
        print(p.stdout.strip())
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return len(list(dest.glob("*.wav")))


def duplicate(trimmed, positive_train, prefix, copies, all_prefixes):
    existing = count_prefix(positive_train, prefix, all_prefixes)
    sources = sorted(trimmed.glob("*.wav"))
    want = len(sources) * copies
    if existing == want:
        print(f"  {prefix}_*: {existing} already present, skipping")
        return existing
    if existing:
        raise SystemExit(
            f"  {prefix}_*: found {existing} files but expected {want}. This is a "
            f"partially-injected set from an interrupted run -- delete them from "
            f"{positive_train} and re-run rather than guessing which are missing."
        )
    for f in sources:
        for i in range(1, copies + 1):
            shutil.copyfile(f, positive_train / f"{prefix}_{f.stem}_{i:03d}.wav")
    print(f"  {prefix}_*: {len(sources)} clips x {copies} copies = {want} files")
    return want


def count_prefix(positive_train, prefix, all_prefixes):
    """Count files belonging to `prefix`, not to a longer prefix that extends it.

    REAL_MUSIC_x.wav starts with "REAL_" too, so a naive glob would count it
    twice and make the composition check pass on a wrong set.
    """
    longer = [q for q in all_prefixes if q != prefix and q.startswith(prefix + "_")]
    return sum(
        1 for p in positive_train.glob(f"{prefix}_*.wav")
        if not any(p.name.startswith(q + "_") for q in longer)
    )


def compose(positive_train, prefixes):
    total = len(list(positive_train.glob("*.wav")))
    counts = {p: count_prefix(positive_train, p, prefixes) for p in prefixes}
    counts["synthetic"] = total - sum(counts.values())
    return total, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.environ.get("CONFIG", "my_model.yaml"))
    ap.add_argument("--check", action="store_true",
                    help="report the composition and verify it, injecting nothing")
    args = ap.parse_args()

    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text())
    sources = config.get("real_clip_sources") or []

    model_dir = resolve(config["output_dir"]) / config["model_name"]
    positive_train = model_dir / "positive_train"
    prefixes = [s["prefix"] for s in sources]

    if not sources:
        print(f"{config_path.name}: no real_clip_sources, nothing to inject")
        return

    if not positive_train.is_dir():
        raise SystemExit(
            f"{positive_train} does not exist -- run './04_train.sh clips' first."
        )

    if not args.check:
        print(f"{config_path.name}: injecting {len(sources)} real-clip source(s)")
        for source in sources:
            src_dir = resolve(source["dir"])
            if not src_dir.is_dir():
                raise SystemExit(
                    f"real_clip_sources: {src_dir} not found. These are real "
                    f"recordings, which live outside the repo -- set OWW_DATA_ROOT "
                    f"(currently {DATA_ROOT})."
                )
            trimmed = model_dir / "_real_trimmed" / source["prefix"]
            n = trim_source(src_dir, stems_for(source), trimmed)
            print(f"  {source['prefix']}: {n} source clips from {src_dir}")
            duplicate(trimmed, positive_train, source["prefix"],
                      int(source["copies"]), prefixes)

    total, counts = compose(positive_train, prefixes)
    print(f"\npositive_train composition ({positive_train}):")
    for k in ["synthetic"] + prefixes:
        print(f"  {k+'_*' if k != 'synthetic' else k:<14s} {counts[k]:>7,}")
    print(f"  {'TOTAL':<14s} {total:>7,}")

    expected = config.get("expect_positive_train_total")
    if expected is not None and total != int(expected):
        raise SystemExit(
            f"\nFAILED: positive_train holds {total:,} files, config expects "
            f"{int(expected):,}. Training on this would silently produce a "
            f"different model. Do not proceed."
        )
    if expected is not None:
        print(f"\nOK: matches expect_positive_train_total ({int(expected):,})")


if __name__ == "__main__":
    main()
