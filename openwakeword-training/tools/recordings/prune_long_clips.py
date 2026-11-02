#!/usr/bin/env python
"""Enforce a maximum speech duration on a generated clip set + its manifest.

`gen_piper_onnx.py` now applies `--max-duration` at generation time, but a set
produced before that guard existed (or one wanted at a tighter cap) can be
brought into line here.  Deletes the offending WAVs and rewrites manifest.tsv,
leaving a `.bak` copy.

The cap applies to *speech* duration, i.e. `duration_s - pause_s`: a deliberate
internal pause is not a runaway.  VITS's stochastic duration predictor
occasionally rambles on a short phrase (one 6.9 s clip in 11k from carlfm),
while the slow `es_MX-ald` voice plus a ~1 s pause legitimately reaches 2.6-3.1 s.

    ./.venv/bin/python tools/recordings/prune_long_clips.py --dir positives_es --max-duration 3.5
    ./.venv/bin/python tools/recordings/prune_long_clips.py --dir positives_es --dry-run
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--manifest", default=None, help="default: <dir>/manifest.tsv")
    ap.add_argument("--max-duration", type=float, default=3.5,
                    help="max speech seconds (duration_s minus pause_s)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    d = Path(args.dir)
    man = Path(args.manifest) if args.manifest else d / "manifest.tsv"
    lines = man.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    i_file = header.index("file")
    i_dur = header.index("duration_s")
    i_pause = header.index("pause_s") if "pause_s" in header else None

    keep, drop = [], []
    for line in lines[1:]:
        if not line.strip():
            continue
        p = line.split("\t")
        speech = float(p[i_dur]) - (float(p[i_pause]) if i_pause is not None else 0.0)
        (drop if speech > args.max_duration else keep).append((p[i_file], speech, line))

    print(f"{len(keep) + len(drop)} clips in manifest; "
          f"{len(drop)} exceed {args.max_duration}s of speech")
    for name, speech, _ in sorted(drop, key=lambda r: -r[1])[:20]:
        print(f"  drop {name}  speech {speech:.3f}s")
    if args.dry_run:
        print("dry run, nothing changed")
        return

    removed = 0
    for name, _, _ in drop:
        f = d / name
        if f.exists():
            f.unlink()
            removed += 1
    if drop:
        shutil.copyfile(man, str(man) + ".bak")
        man.write_text("\t".join(header) + "\n" + "".join(l + "\n" for _, _, l in keep),
                       encoding="utf-8")
    on_disk = len(list(d.glob("*.wav")))
    print(f"removed {removed} files; manifest now {len(keep)} rows; "
          f"{on_disk} .wav on disk")
    if on_disk != len(keep):
        print(f"  WARNING: {on_disk} files on disk vs {len(keep)} manifest rows")


if __name__ == "__main__":
    main()
