#!/usr/bin/env python
"""Classify recordings by acoustic condition: normal / whisper / music / far.

Written for a set where the conditions are known to exist but not labelled.
Each condition has a distinct acoustic signature that survives without
listening to the audio:

  whisper -- whispered speech has no vocal-fold vibration, so there is no
             fundamental frequency. Voiced-frame fraction collapses toward
             zero and energy shifts upward in the spectrum.
  music   -- music runs continuously underneath the speech, so the quiet
             frames between words never reach the noise floor a speech-only
             recording drops to. Measured as the energy floor (low percentile
             of frame RMS) relative to the speech peak.
  far     -- still normally phonated (voiced), but attenuated and reverberant:
             low absolute level with a voiced fraction in the normal range.

    ./.venv-analysis/bin/python tools/recordings/classify_conditions.py            # train/ + eval/
    ./.venv-analysis/bin/python tools/recordings/classify_conditions.py <some-dir>

NOTE: the default pool no longer contains what this was written to find. The
music and whisper clips it originally separated out now live in
<root>/unused/excluded/{music,whisper} -- already classified, which is why they
were pulled out. --reference-max-index also assumes a set numbered so that the
low indices are the normal-condition reference and the high ones are the hard
cases; that holds for the speaker_a_NN/speaker_b_NN clips, not for the tagged
speaker_a_<condition>_NN ones. Point it at an unclassified directory to use it in
anger.
"""
import argparse
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# python puts the *script's* directory on sys.path, not the cwd, so the
# repo-root modules below have to be found explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from repo_paths import DATA_ROOT, REPO_ROOT  # noqa: E402

SR = 16000


def features(path):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)

    n_fft, hop = 512, 160  # 32 ms window, 10 ms hop
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    rms = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop)[0]
    peak = np.percentile(rms, 95) + 1e-9

    # Energy floor relative to the loud part: high when something (music)
    # sounds continuously underneath, low when pauses are true silence.
    floor_ratio = np.percentile(rms, 20) / peak

    # Voicing: pitch tracking over the frames that actually carry energy.
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=60, fmax=400, sr=SR, frame_length=1024, hop_length=hop)
    loud = rms[:len(voiced_flag)] > 0.25 * peak
    voiced_frac = float(np.nansum(voiced_flag[loud]) / max(loud.sum(), 1))

    centroid = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=SR)[0][loud[:S.shape[1]]])) \
        if loud.any() else float("nan")
    flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)[0]))

    # Fraction of the clip carrying meaningful energy. Music keeps this high
    # across the whole clip; isolated speech does not.
    active_frac = float((rms > 0.25 * peak).mean())

    # Look at what is happening *between* the words. A speech-only recording
    # has near-silence there; music keeps playing, and what it leaves behind is
    # tonal (low spectral flatness) rather than broadband hiss.
    quiet = rms < 0.25 * peak
    if quiet.sum() >= 3:
        Sq = S[:, quiet[:S.shape[1]]] if quiet[:S.shape[1]].any() else S
        bg_flatness = float(np.mean(librosa.feature.spectral_flatness(S=Sq)[0]))
        bg_level = float(np.mean(rms[quiet]) / peak)
    else:
        bg_flatness, bg_level = float("nan"), float("nan")

    # Rhythmic pulse: music has periodic onsets, speech does not.
    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop)
    ac = librosa.autocorrelate(onset - onset.mean(), max_size=len(onset) // 2)
    pulse = float(np.max(ac[10:] / (ac[0] + 1e-9))) if len(ac) > 12 else float("nan")

    return {
        "rms": float(np.sqrt((y ** 2).mean())),
        "peak": float(peak),
        "floor_ratio": float(floor_ratio),
        "voiced_frac": voiced_frac,
        "centroid": centroid,
        "flatness": flatness,
        "active_frac": active_frac,
        "bg_flatness": bg_flatness,
        "bg_level": bg_level,
        "pulse": pulse,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directories", nargs="*",
                    help="directories of recordings (default: the 133 originals "
                         "in <root>/train and <root>/eval)")
    ap.add_argument("--reference-max-index", type=int, default=25,
                    help="clips at or below this index are treated as the "
                         "normal-condition reference set")
    ap.add_argument("--per-speaker-music", type=int, default=5)
    ap.add_argument("--per-speaker-whisper", type=int, default=5)
    args = ap.parse_args()

    # unused/ stays out of the default pool: rec_trimmed/ and rec_trimmed_25/
    # are re-cuts of clips already in train/ and share their stems, so the same
    # recording would be measured three times over.
    dirs = ([Path(d) for d in args.directories] if args.directories
            else [DATA_ROOT / "train", DATA_ROOT / "eval"])
    files = sorted((f for d in dirs if d.is_dir() for f in d.glob("*.wav")),
                   key=lambda p: p.name)
    if not files:
        sys.exit(f"no wav files in {', '.join(str(d) for d in dirs)}")

    def speaker(p):
        return p.stem.rsplit("_", 1)[0]

    def index(p):
        return int(p.stem.rsplit("_", 1)[1])

    print(f"Analysing {len(files)} files...", file=sys.stderr)
    feats = {p: features(p) for p in files}

    labels = {}
    for spk in sorted({speaker(p) for p in files}):
        ref = [p for p in files if speaker(p) == spk and index(p) <= args.reference_max_index]
        hard = [p for p in files if speaker(p) == spk and index(p) > args.reference_max_index]

        # Reference statistics for this speaker's normal recordings.
        def refmed(k):
            return float(np.nanmedian([feats[p][k] for p in ref]))

        print(f"\n=== {spk} — reference (clips <= {args.reference_max_index}, n={len(ref)}) ===")
        print("  " + " | ".join(f"{k} {refmed(k):.3f}" for k in
                                ("voiced_frac", "floor_ratio", "active_frac",
                                 "bg_level", "bg_flatness", "pulse")))

        # Music first: the between-words signature is the least ambiguous of
        # the three, and claiming it first stops a music clip whose vocal is
        # partly masked from being mistaken for a whisper.
        def music_score(p):
            f = feats[p]
            return (f["bg_level"] / (refmed("bg_level") + 1e-9)
                    + f["active_frac"] / (refmed("active_frac") + 1e-9)
                    - 2 * f["bg_flatness"] / (refmed("bg_flatness") + 1e-9))

        music = set(sorted(hard, key=music_score, reverse=True)[:args.per_speaker_music])

        # Whisper: no vocal-fold vibration, judged among what music didn't take.
        rest = [p for p in hard if p not in music]
        whisper = set(sorted(rest, key=lambda p: feats[p]["voiced_frac"])[:args.per_speaker_whisper])

        for p in hard:
            labels[p] = "music" if p in music else "whisper" if p in whisper else "far"

        print(f"  {'clip':<16}{'voiced':>8}{'bglvl':>8}{'bgflat':>8}{'active':>8}"
              f"{'pulse':>8}{'peak':>9}  label")
        for p in sorted(hard, key=index):
            f = feats[p]
            print(f"  {p.name:<16}{f['voiced_frac']:>8.3f}{f['bg_level']:>8.3f}"
                  f"{f['bg_flatness']:>8.4f}{f['active_frac']:>8.2f}{f['pulse']:>8.3f}"
                  f"{f['peak']:>9.4f}  {labels[p]}")

    print("\n" + "=" * 60)
    for cond in ("music", "whisper", "far"):
        got = sorted((p.name for p, c in labels.items() if c == cond))
        print(f"{cond:<8} ({len(got):>2}): {', '.join(got)}")

    out = REPO_ROOT / "data" / "conditions.txt"
    out.write_text("\n".join(f"{p.name}\t{c}" for p, c in sorted(labels.items())) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
