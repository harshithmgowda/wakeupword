#!/usr/bin/env python
"""Music/TV robustness eval: real clips, SNR overlay sweep, FP streaming.

    ./.venv-onnx2tf/bin/python tools/eval/eval_music.py \
        --model mine=my_custom_model/ouh_ke-i_nahbu.tflite \
        --model stock=stock_models/ok_nabu_v0.1.tflite \
        --out baseline.json

Performance note (deviation from the task-7 brief, see task-7-report.md):
openWakeWord embeddings come from a shared feature extractor and do not
depend on the classifier head, so for both the SNR sweep and the streaming
false-positive pass, each audio source is embedded ONCE here and the
resulting (frames, 96) embedding is windowed + scored by every requested
model. The brief's original code recomputed embeddings inside each model's
loop; with ~4h of streaming audio, that embedding step is the dominant cost,
so caching it turns an O(models x audio) embedding cost into O(audio).
Scoring itself (compare_models.predict) is still called once per model per
window batch -- only the expensive feature extraction is cached. The output
JSON schema is unchanged.

Determinism note (deviation): the brief's sweep RNG seed used Python's
built-in hash() on a tuple containing a string. CPython randomizes str
hashing per-process by default (PYTHONHASHSEED), so that seed -- and the
resulting bg-file pick / mix -- would differ on every run of this script,
violating the "deterministic given the seeds in the code" requirement. This
version derives the per-(clip, snr, base) seed with zlib.crc32 over a fixed
string encoding instead, which is stable across process runs while keeping
the same "one seed per (clip, snr, base)" scheme.

CAVEAT -- stock ok_nabu is not a fair recall comparison: stock_models/
ok_nabu_v0.1.tflite is trained on the wake phrase "ok nabu", a DIFFERENT
phrase from this project's target "ouh ke-i nahbu". They only share a
"nabu"/"nahbu" suffix. Any non-zero score stock_models/*.tflite gets here
(e.g. 3/22 real_music, sweep rates up to .333) most plausibly reflects
incidental partial-phrase phonetic overlap on that shared suffix, not
genuine recognition of the target phrase -- stock was never trained to
recognize "ouh ke-i nahbu" and has no way to. It is included only as a
rough external reference point (how a shipped, unrelated model reacts to
this audio), NOT as a baseline this project's model is meant to beat.
Do not read the v5-vs-stock columns as apples-to-apples recall numbers.
"""
import sys
import argparse
import json
import os
import zlib
from pathlib import Path

import numpy as np

# python puts the *script's* directory on sys.path, not the cwd, so the
# repo-root modules below have to be found explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from repo_paths import DATA_ROOT  # noqa: E402
import compare_models as cm
from media_utils import read_wav16k, rms, pick_segment, embed, windows16, SR

# Honor OWW_DATA_ROOT like classify_conditions.py / make_split.py /
# record_clips.sh / label_clips.sh, defaulting to the same
# ~/openwakeword-train-data used everywhere else.
REAL_MUSIC = DATA_ROOT / "unused" / "excluded" / "music"
HELD_OUT = DATA_ROOT / "eval"
SNRS = [10, 5, 0, -5]
LEAD_IN_S = 1.0
GAP_WINDOWS = 12  # ~1 s between distinct activations when streaming
# Backgrounds quieter than this are near-silent (encoding artifacts / blank
# leader). RMS-relative SNR mixing scales such a background up enormously to
# hit the target SNR, which turns dither/quantization noise into the
# "music" -- not a valid music condition. Excluded from the SWEEP background
# pool only (see task-7-report.md); same constant make_tv_mix.py uses for
# the same reason. The streaming FP pool is left intact.
SWEEP_BG_MIN_RMS = 0.01

# See module docstring CAVEAT. Kept as one string so the console print, the
# JSON "notes" field, and the docstring all say the same thing.
STOCK_CAVEAT = (
    "CAVEAT: stock ok_nabu targets a DIFFERENT wake phrase (\"ok nabu\") "
    "than this project's model (\"ouh ke-i nahbu\") -- they only share a "
    "'nabu'/'nahbu' suffix. Its non-zero scores here most plausibly reflect "
    "incidental partial-phrase phonetic overlap on that suffix, not genuine "
    "recognition of the target phrase. Treat it as a rough external "
    "reference point only, NOT as a baseline this project's model is meant "
    "to beat -- the v5-vs-stock columns are not an apples-to-apples recall "
    "comparison."
)


def _is_stock_reference(name, path):
    """True if this --model spec looks like an off-target reference model
    (named "stock" or living under stock_models/) that needs STOCK_CAVEAT
    attached to its results, rather than a model trained on this project's
    wake phrase."""
    return name.lower() == "stock" or "stock_models" in Path(path).parts


def _seed(*parts):
    """Deterministic (process-stable) seed from arbitrary hashable parts.

    Python's built-in hash() randomizes str hashing per process by default,
    which would make the sweep's bg pick / mix -- and therefore its
    detection rate -- different on every run. crc32 over a fixed string
    encoding is stable across runs and interpreters.
    """
    return zlib.crc32(repr(parts).encode()) % 2**32


def peak_from_windows(interp, w):
    return float(cm.predict(interp, w).max()) if len(w) else float("nan")


def build_sweep_mix(clip_path, bg_files, snr, rng_seed_base):
    """Build one lead-in + utterance clip mixed with a background at snr dB.

    Model-independent (depends only on clip/bg/snr/seed), so it is computed
    once per (clip, snr) and the resulting audio is embedded once and scored
    by every model -- see module docstring.
    """
    clip = read_wav16k(str(clip_path))
    n_lead = int(LEAD_IN_S * SR)
    tail = clip[-n_lead:] if len(clip) >= n_lead else clip
    lead = np.tile(tail, int(np.ceil(n_lead / max(1, len(tail)))) + 1)[:n_lead]
    full = np.concatenate([lead, clip]).astype(np.float32)
    rng = np.random.default_rng(_seed(clip_path.stem, snr, rng_seed_base))
    bg = pick_segment(read_wav16k(str(rng.choice(bg_files))), len(full), rng)
    # SNR is speech-vs-background: reference the utterance clip, not the lead-in
    scale = (rms(clip) / rms(bg)) * (10.0 ** (-snr / 20.0))
    mixed = full + bg * scale
    peak = np.abs(mixed).max()
    if peak > 0.95:
        mixed *= 0.95 / peak
    return mixed


def sweep_rate(interp, embs, thr):
    detected = 0
    for emb in embs:
        w = windows16(emb, stride=1)
        if peak_from_windows(interp, w) >= thr:
            detected += 1
    return detected / len(embs)


def stream_fp_from_cache(interp, cached, hours, thr):
    activations = 0
    for emb in cached:
        w = windows16(emb, stride=1)
        if not len(w):
            continue
        hot = cm.predict(interp, w) >= thr
        idx = np.flatnonzero(hot)
        if len(idx):
            # Grouping is per-file (idx is recomputed fresh each iteration,
            # never carried across files), and each streaming file is a
            # single ~30 s clip, so GAP_WINDOWS (~1 s) can only merge hot
            # windows that are already within the same short clip -- it
            # cannot merge activations "minutes apart" across files.
            activations += 1 + int((np.diff(idx) > GAP_WINDOWS).sum())
    return {"hours": round(hours, 2), "activations": activations,
            "per_hour": round(activations / max(hours, 1e-9), 3)}


def main():
    import openwakeword.utils
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="name=path/to/model.tflite (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="real,sweep,stream")
    ap.add_argument("--threshold", type=float, default=cm.THRESHOLD)
    args = ap.parse_args()
    thr = args.threshold

    feat = openwakeword.utils.AudioFeatures(device="cpu")
    parts = set(args.parts.split(","))
    music_bg = sorted(Path("eval_media/music").glob("*.wav"))
    tv_bg = sorted(Path("eval_media/tv").glob("*.wav"))
    held_out = sorted(HELD_OUT.glob("*.wav"))

    # ---- Model-independent precompute: embed each audio source ONCE ----
    # (see module docstring "Performance note").
    sweep_cache = {}
    if "sweep" in parts:
        sweep_music_bg = [p for p in music_bg
                           if rms(read_wav16k(str(p))) >= SWEEP_BG_MIN_RMS]
        sweep_tv_bg = [p for p in tv_bg
                       if rms(read_wav16k(str(p))) >= SWEEP_BG_MIN_RMS]
        dropped_music = len(music_bg) - len(sweep_music_bg)
        dropped_tv = len(tv_bg) - len(sweep_tv_bg)
        if dropped_music or dropped_tv:
            print(f"[sweep] excluded {dropped_music} near-silent music / "
                  f"{dropped_tv} near-silent tv backgrounds (rms < "
                  f"{SWEEP_BG_MIN_RMS}) from the SNR sweep pool")
        for cond, bgs in [("music", sweep_music_bg), ("tv", sweep_tv_bg)]:
            for snr in SNRS:
                mixes = [build_sweep_mix(cp, bgs, snr, 100) for cp in held_out]
                sweep_cache[(cond, snr)] = [embed(feat, m) for m in mixes]

    stream_cache = {}
    if "stream" in parts:
        # An unbuilt/empty eval_media/{music,tv} would otherwise fail
        # silently: hours accumulates to 0.0, activations stays 0, and
        # stream_fp_from_cache's per_hour = activations / max(hours, 1e-9)
        # comes out 0.0 -- bit-identical to a genuine zero-false-activation
        # result instead of an error. Same precedent as make_tv_mix.py's
        # `assert music_all and speech` for its background pools.
        assert music_bg, ("no files in eval_media/music -- run "
                           "make_eval_media.py first")
        assert tv_bg, ("no files in eval_media/tv -- run "
                        "make_eval_media.py first")
        for cond, files in [("music", music_bg), ("tv", tv_bg)]:
            embs, hours = [], 0.0
            for f in files:
                audio = read_wav16k(str(f))
                hours += len(audio) / SR / 3600
                embs.append(embed(feat, audio))
            stream_cache[cond] = (embs, hours)
            print(f"[stream] {cond}: embedded {len(files)} files, "
                  f"{hours:.2f} h")

    results = {"threshold": thr, "models": {}}
    has_stock_reference = False
    for spec in args.model:
        name, path = spec.split("=", 1)
        interp = cm.load(path)
        r = {}
        if "real" in parts:
            c = cm.eval_clips(interp, REAL_MUSIC, lead_in=LEAD_IN_S)
            r["real_music"] = {"n": c["n_clips"],
                               "detected": sum(1 for _, s in c["per_file"] if s >= thr),
                               "per_file": dict(c["per_file"])}
        if "sweep" in parts:
            r["sweep"] = {
                cond: {str(snr): round(sweep_rate(interp, sweep_cache[(cond, snr)], thr), 3)
                       for snr in SNRS}
                for cond in ["music", "tv"]}
        if "stream" in parts:
            r["stream_fp"] = {
                cond: stream_fp_from_cache(interp, *stream_cache[cond], thr)
                for cond in ["music", "tv"]}
        results["models"][name] = r
        print(f"== {name}: {json.dumps(r, indent=1, default=str)[:600]}")
        if _is_stock_reference(name, path):
            has_stock_reference = True
            print(f"[{name}] {STOCK_CAVEAT}")

    if has_stock_reference:
        results["notes"] = STOCK_CAVEAT

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
