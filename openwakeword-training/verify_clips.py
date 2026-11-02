#!/usr/bin/env python
"""Delete unreadable or empty generated clips.

If the machine dies partway through clip generation, piper leaves behind files
that exist but hold no audio. That matters because train.py decides how many
more clips to generate by counting directory entries -- so the empty files are
counted as done, and then blow up the augmentation stage instead.

Removing them makes the count honest, and the next --generate_clips run tops
the directory back up.

    ./.venv/bin/python verify_clips.py            # clean my_model.yaml's output
    ./.venv/bin/python verify_clips.py --dry-run  # just report
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf
import yaml

SPLITS = ("positive_train", "positive_test", "negative_train", "negative_test")


def check(path):
    try:
        return path, sf.info(path).frames > 0
    except Exception:
        return path, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="my_model.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    root = Path(cfg["output_dir"]) / cfg["model_name"]
    if not root.exists():
        print(f"{root} does not exist yet -- nothing to verify")
        return

    total_bad = 0
    for split in SPLITS:
        d = root / split
        if not d.exists():
            continue
        files = list(d.glob("*.wav"))
        with ThreadPoolExecutor(max_workers=16) as ex:
            bad = [f for f, ok in ex.map(check, files) if not ok]
        total_bad += len(bad)

        if bad and not args.dry_run:
            for f in bad:
                f.unlink()
        state = "would remove" if args.dry_run else "removed"
        note = f", {state} {len(bad)}" if bad else ""
        print(f"  {split:<16} {len(files) - len(bad):>6} valid{note}")

    if total_bad and not args.dry_run:
        print(f"\nRemoved {total_bad} unusable clips -- re-run the clips stage to regenerate them.")
    elif not total_bad:
        print("\nAll clips are readable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
