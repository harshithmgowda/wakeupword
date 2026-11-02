#!/usr/bin/env python
"""Rename recording files by speaker prefix, so audio filenames match the repo.

The committed metadata (`data/labels.tsv`, `data/split_*.txt`,
`data/real_music_ground_truth.tsv`, and every evaluation result on the
experiments branch) refers to recordings by neutral speaker IDs --
`speaker_a_07.wav`, `speaker_b_34.wav`. The
audio itself lives outside the repo (see `repo_paths.DATA_ROOT`) and may still
carry whatever prefixes it was recorded with.

This script bridges the two. The mapping is supplied on the command line, so no
real name is ever written into the repo:

    python tools/recordings/rename_speakers.py --map oldname=speaker_a           # reports only
    python tools/recordings/rename_speakers.py --map oldname=speaker_a --apply   # renames

It renames `<old>_*` to `<new>_*` under every subdirectory of DATA_ROOT, and
rewrites any `manifest.tsv` alongside them. Default is a dry run in all but
name: nothing is touched until you pass `--apply`.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from repo_paths import DATA_ROOT  # noqa: E402

AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


def parse_map(pairs):
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--map expects old=new, got {pair!r}")
        old, new = pair.split("=", 1)
        old, new = old.strip(), new.strip()
        if not old or not new:
            raise SystemExit(f"--map expects non-empty names, got {pair!r}")
        out[old] = new
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", action="append", required=True, metavar="OLD=NEW",
                    help="prefix rename, repeatable (e.g. --map alice=speaker_a)")
    ap.add_argument("--root", default=None,
                    help=f"corpus root (default: {DATA_ROOT})")
    ap.add_argument("--apply", action="store_true",
                    help="actually rename; without this the script only reports")
    args = ap.parse_args()

    mapping = parse_map(args.map)
    root = Path(args.root).expanduser() if args.root else DATA_ROOT
    if not root.is_dir():
        raise SystemExit(f"No such directory: {root}")

    renames = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in AUDIO_EXT:
            continue
        for old, new in mapping.items():
            if p.name.startswith(old + "_"):
                renames.append((p, p.with_name(new + p.name[len(old):])))
                break

    manifests = sorted(root.rglob("manifest.tsv"))

    print(f"root: {root}")
    print(f"{len(renames)} audio files match, {len(manifests)} manifest.tsv found")
    for src, dst in renames[:10]:
        print(f"  {src.relative_to(root)}  ->  {dst.name}")
    if len(renames) > 10:
        print(f"  ... and {len(renames) - 10} more")

    if not args.apply:
        print("\nDry run. Re-run with --apply to perform the renames.")
        return

    clashes = [d for _, d in renames if d.exists()]
    if clashes:
        raise SystemExit(f"Refusing to overwrite {len(clashes)} existing file(s), "
                         f"e.g. {clashes[0]}")

    for src, dst in renames:
        src.rename(dst)
    print(f"renamed {len(renames)} files")

    for m in manifests:
        text = m.read_text(encoding="utf-8")
        updated = text
        for old, new in mapping.items():
            updated = updated.replace(old + "_", new + "_")
        if updated != text:
            m.write_text(updated, encoding="utf-8")
            print(f"rewrote {m.relative_to(root)}")


if __name__ == "__main__":
    main()
