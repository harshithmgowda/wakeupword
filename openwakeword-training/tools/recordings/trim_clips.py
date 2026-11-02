#!/usr/bin/env python
"""Trim recordings down to the utterance, centered in a fixed-length window.

Writes to a new directory -- originals are never modified.

Two constraints decide the output length, and they pull in opposite directions:

  * The embedding model needs 76 melspectrogram frames before it emits
    anything, so a clip shorter than 2.0s yields ZERO predictions. Trimming to
    the bare utterance (~0.65s here) would make every clip unusable.
  * Training pads every clip to exactly 2.0s (train.py's `total_length`,
    minimum 32000 samples), which is why the model takes 16 frames. Matching
    that is what makes these clips comparable to the training distribution.

So the default is 2.0s: the tightest window that still produces a prediction,
and the one training actually uses. Padding is taken from the surrounding
original audio wherever possible, so the context stays real room tone rather
than digital silence.

    R=~/openwakeword-train-data
    ./.venv-analysis/bin/python tools/recordings/trim_clips.py $R/train $R/unused/rec_trimmed
    ./.venv-analysis/bin/python tools/recordings/trim_clips.py $R/train $R/unused/rec_trimmed_25 --duration 2.5

The two sets already sitting in $R/unused/ were cut from the 49 clips of the
original ~/rec, which are now only part of train/ -- so re-running the above
produces 106 clips, not 49. Those existing sets are also not usable as v5
evaluation data: every recording behind them is in train/.
"""
import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 16000


def find_utterance(y, hop=160):
    """Return (start, end) sample indices of the spoken part.

    Uses a high threshold relative to the clip's own peak: these recordings
    have room tone and breath that a low threshold happily includes.
    """
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=hop))
    rms = librosa.feature.rms(S=S, frame_length=512, hop_length=hop)[0]
    peak = np.percentile(rms, 95)
    idx = np.where(rms > 0.30 * peak)[0]
    if len(idx) < 2:
        return 0, len(y)
    return int(idx[0] * hop), int(min(len(y), (idx[-1] + 1) * hop))


def trim_for_training(path, margin=0.25, cap=1.9):
    """Tight trim around the utterance, for use as training clips.

    augment_clips left-pads each clip to total_length (2.0s) and truncates
    anything longer, so training clips must stay under that. The synthetic
    clips it is used to sit at ~0.82s, so we aim for the same ballpark: the
    energy core (5th-95th percentile of cumulative energy) plus a margin for
    quiet onsets and final consonants the energy measure misses.
    """
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)

    hop = 160
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=hop))
    rms = librosa.feature.rms(S=S, frame_length=512, hop_length=hop)[0]
    energy = rms ** 2
    cum = np.cumsum(energy) / energy.sum()
    lo_f, hi_f = int(np.searchsorted(cum, 0.05)), int(np.searchsorted(cum, 0.95))

    lo = max(0, int(lo_f * hop - margin * SR))
    hi = min(len(y), int(hi_f * hop + margin * SR))
    if (hi - lo) / SR > cap:                    # keep the centre if too long
        centre = (lo + hi) // 2
        half = int(cap * SR / 2)
        lo, hi = max(0, centre - half), min(len(y), centre + half)
    return y[lo:hi], (hi - lo) / SR


def trim(path, duration):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)

    want = int(duration * SR)
    start, end = find_utterance(y)
    centre = (start + end) // 2
    lo = centre - want // 2
    hi = lo + want

    # Prefer real surrounding audio over silence; only zero-pad if the source
    # genuinely runs out.
    if lo < 0:
        hi = min(len(y), hi - lo)
        lo = 0
    if hi > len(y):
        lo = max(0, lo - (hi - len(y)))
        hi = len(y)
    out = y[lo:hi]
    if len(out) < want:
        pad = want - len(out)
        out = np.pad(out, (pad // 2, pad - pad // 2))

    return out, (end - start) / SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--duration", type=float, default=2.0,
                    help="output length in seconds (default 2.0 = training's "
                         "total_length; below 2.0 the model cannot predict)")
    ap.add_argument("--training", action="store_true",
                    help="tight variable-length trim for use as TRAINING clips "
                         "(augment_clips does its own padding). Not usable for "
                         "evaluation -- see the note in this file's docstring.")
    ap.add_argument("--only", help="file listing clip stems to include (one per line)")
    args = ap.parse_args()

    if args.duration < 2.0:
        print(f"WARNING: {args.duration}s is below the 2.0s minimum -- the "
              f"embedding model will produce no predictions from these clips.")

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.wav"))
    if args.only:
        keep = set(Path(args.only).read_text().split())
        files = [f for f in files if f.stem in keep]
    if not files:
        raise SystemExit(f"no matching wav files in {src}")

    spans = []
    for f in files:
        if args.training:
            out, span = trim_for_training(f)
        else:
            out, span = trim(f, args.duration)
        sf.write(dst / f.name, out, SR)
        spans.append(span)

    spans = np.array(spans)
    print(f"Trimmed {len(files)} clips -> {dst}")
    if args.training:
        print(f"  clip length: median {np.median(spans):.2f}s  "
              f"range {spans.min():.2f}-{spans.max():.2f}s  (synthetic median 0.82s)")
        over = (spans >= 2.0).sum()
        print(f"  clips >= 2.0s (would be truncated by augment_clips): {over}")
    else:
        print(f"  output length : {args.duration:.2f}s")
        print(f"  utterance span: median {np.median(spans):.2f}s  "
              f"range {spans.min():.2f}-{spans.max():.2f}s")
        long = [f.name for f, s in zip(files, spans) if s > args.duration]
        if long:
            print(f"  WARNING: {len(long)} utterance(s) longer than the output "
                  f"window, so they are clipped: {', '.join(long[:5])}")


if __name__ == "__main__":
    main()
