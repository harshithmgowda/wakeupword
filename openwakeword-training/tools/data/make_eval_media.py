#!/usr/bin/env python
"""Extract FMA tracks NOT used in training into eval_media/{music,_tv_beds}.

    ./.venv/bin/python tools/data/make_eval_media.py --n-music 240 --n-beds 240
"""
import sys
import argparse, subprocess, zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# python puts the *script's* directory on sys.path, not the cwd, so the
# repo-root modules below have to be found explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from repo_paths import REPO_ROOT  # noqa: E402

HERE = REPO_ROOT
ZIP = HERE / "downloads" / "fma_small.zip"
TRAIN_DIR = HERE / "fma"


def _convert(args):
    data, dest = args
    p = subprocess.run(
        ["ffmpeg", "-loglevel", "quiet", "-i", "pipe:0", "-ac", "1",
         "-ar", "16000", "-acodec", "pcm_s16le", "-f", "wav", "-y", dest],
        input=data, capture_output=True)
    return p.returncode == 0


def extract(z, names, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(z.read(n), str(out_dir / (Path(n).stem + ".wav")))
            for n in names if not (out_dir / (Path(n).stem + ".wav")).exists()]
    with ProcessPoolExecutor(max_workers=8) as ex:
        ok = sum(ex.map(_convert, jobs))
    print(f"  {out_dir.name}: {ok} new, {len(list(out_dir.glob('*.wav')))} total")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-music", type=int, default=240)
    ap.add_argument("--n-beds", type=int, default=240)
    args = ap.parse_args()

    train_stems = {p.stem for p in TRAIN_DIR.glob("*.wav")}
    with zipfile.ZipFile(ZIP) as z:
        fresh = [n for n in sorted(z.namelist())
                 if n.endswith(".mp3") and Path(n).stem not in train_stems]
        # same spread-selection idea as 02_download_data.py stage_fma
        want = args.n_music + args.n_beds
        step = max(1, len(fresh) // want)
        picked = fresh[::step][:want]
        extract(z, picked[:args.n_music], HERE / "eval_media" / "music")
        extract(z, picked[args.n_music:], HERE / "eval_media" / "_tv_beds")

    # paranoia: verify disjointness from training music
    for d in ["music", "_tv_beds"]:
        overlap = {p.stem for p in (HERE / "eval_media" / d).glob("*.wav")} & train_stems
        assert not overlap, f"eval media overlaps training music: {sorted(overlap)[:5]}"
    print("eval media disjoint from ./fma: OK")


if __name__ == "__main__":
    main()
