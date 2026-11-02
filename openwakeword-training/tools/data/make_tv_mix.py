#!/usr/bin/env python
"""Compose 30 s TV-like clips: TTS speech segments over a music bed.

    ./.venv/bin/python tools/data/make_tv_mix.py --music fma --speech tv_mix_speech \
        --out tv_mix --clips 1200 --seed 2
"""
import sys
import argparse
from pathlib import Path

import numpy as np

# python puts the *script's* directory on sys.path, not the cwd, so the
# repo-root modules below have to be found explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from media_utils import read_wav16k, write_wav16k, rms, pick_segment, SR

CLIP_SECONDS = 30
# A handful of source tracks are near-silent (encoding artifacts / blank
# leader); as a bed they make the whole 30 s clip near-silent too (mixed
# rms < 0.005, no audible music or speech). Drop them from the pool.
MUSIC_MIN_RMS = 0.01


def compose(music_files, speech_files, rng, db_lo, db_hi):
    n = CLIP_SECONDS * SR
    bed = pick_segment(read_wav16k(str(rng.choice(music_files))), n, rng)
    speech = np.zeros(n, dtype=np.float32)
    pos = int(rng.integers(0, SR))
    while pos < n - SR:
        clip = read_wav16k(str(rng.choice(speech_files)))
        end = min(pos + len(clip), n)
        speech[pos:end] += clip[:end - pos]
        pos = end + int(rng.uniform(0.3, 1.5) * SR)
    # speech sits db above the bed, like TV dialog over theme music
    db = rng.uniform(db_lo, db_hi)
    if rms(speech) > 1e-6:
        speech *= (rms(bed) / rms(speech)) * (10.0 ** (db / 20.0))
    mixed = bed + speech
    peak = np.abs(mixed).max()
    return mixed * (0.9 / peak) if peak > 0.9 else mixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--music", required=True)
    ap.add_argument("--speech", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--speech-db-lo", type=float, default=4.0)
    ap.add_argument("--speech-db-hi", type=float, default=10.0)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    # Resume by exact filename, not a percentage threshold: a run that is
    # 99% done still has missing/broken files that must be (re)generated.
    dests = [out / f"tv_{args.seed}_{i:05d}.wav" for i in range(args.clips)]
    missing = [i for i, d in enumerate(dests) if not d.exists()]
    if not missing:
        print(f"[tv_mix] {args.clips} clips present, skipping")
        return

    # Only pay for the file-list scan (and the RMS decode of every music
    # file below) when there is actually work to do.
    music_all = sorted(Path(args.music).glob("*.wav"))
    speech = sorted(Path(args.speech).glob("*.wav"))
    assert music_all and speech, "empty music or speech dir"
    music = [p for p in music_all if rms(read_wav16k(str(p))) >= MUSIC_MIN_RMS]
    if len(music) < len(music_all):
        print(f"[tv_mix] dropped {len(music_all) - len(music)} near-silent "
              f"music beds (rms < {MUSIC_MIN_RMS})")
    assert music, "no music files left after silence filtering"

    print(f"[tv_mix] {args.clips - len(missing)}/{args.clips} clips present, "
          f"generating {len(missing)} more")
    for done, i in enumerate(missing, 1):
        # A fresh, index-derived generator -- not one shared stream walked
        # in a for-loop -- so clip i's content depends only on (seed, i)
        # and is identical whether generated from scratch or on resume.
        # A shared rng would draw index i's random numbers at whatever
        # position the loop happened to reach it, which shifts on every
        # resume (skipped indices consume no draws) and desyncs clip
        # content from its filename.
        rng = np.random.default_rng([args.seed, i])
        write_wav16k(dests[i], compose(music, speech, rng, args.speech_db_lo,
                                       args.speech_db_hi))
        if done % 100 == 0:
            print(f"  {done}/{len(missing)}", flush=True)
    print(f"[tv_mix] wrote {len(list(out.glob('*.wav')))} clips -> {out}")


if __name__ == "__main__":
    main()
