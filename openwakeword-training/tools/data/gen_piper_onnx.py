#!/usr/bin/env python
"""Generate wake-word positives from ordinary Piper ONNX voices.

WHY THIS EXISTS.  Every one of v7's 50,000 synthetic positives comes from a
single TTS model, `en_US-libritts_r-medium.pt`, phonemised with espeak's
`en-us` voice.  The wake phrase is a Spanish-accented rendering of "ok nabu"
and both household speakers are Spanish-accented, so nothing in training
covers the target accent.  piper-sample-generator only ever shipped .pt
"generator" checkpoints for en_US/de_DE/fr_FR/nl_NL (release v2.0.0), so
there is no Spanish .pt to drop in.  Spanish *is* available as normal Piper
ONNX voices, which upstream piper-sample-generator v3.x consumes directly
(`--model voices/xx.onnx`); this script is that inference path implemented
against the already-installed onnxruntime, so no new dependency is needed.

Deliberately reuses `get_phonemes`, `remove_silence` and
`audio_float_to_int16` from piper-sample-generator's `generate_samples.py`, so
phoneme-id construction, VAD silence trimming and int16 peak normalisation are
bit-for-bit the same conditioning the existing 50k English positives got.  Only
the acoustic model and the espeak voice change.

    ./.venv/bin/python tools/data/gen_piper_onnx.py \
        --out-dir positives_es --count 20000 --device cuda

Outputs 16 kHz mono int16 WAVs plus a manifest TSV recording, per clip, the
voice / speaker / spelling / scales / warp actually used, so a later phase can
subset or ablate any of those without regenerating.
"""
import argparse
import json
import random
import sys
import uuid
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

# python puts the *script's* directory on sys.path, not the cwd, so the
# repo-root modules below have to be found explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from repo_paths import REPO_ROOT  # noqa: E402

HERE = REPO_ROOT
sys.path.insert(0, str(HERE / "piper-sample-generator"))
from generate_samples import (  # noqa: E402
    audio_float_to_int16,
    get_phonemes,
    remove_silence,
)

SR = 16000

# Spanish spellings of the wake phrase.  "ouh_ke-i_nahbu" is the phonetic hack
# the project has always used (and what v7 was trained on, but phonemised as
# en-us); under espeak `es` it reads [ˈow kˈe ˈi nˈaβu], which is close to how a
# Spanish speaker actually renders "ok nabu".  The rest are natural Spanish
# orthography.  Which of these is kept is decided by measurement, not taste --
# the duration + MFCC-DTW probe that did it is on the experiments branch.
DEFAULT_TEXTS = [
    "ouh_ke-i_nahbu",
    "okey nabu",
    "ok nabu",
    "oquei nabu",
    "oc nabu",
]


def load_voice(onnx_path, device="cpu", intra_threads=1):
    """Open one Piper ONNX voice; returns (session, config, meta)."""
    onnx_path = Path(onnx_path)
    config = json.loads((onnx_path.parent / (onnx_path.name + ".json")).read_text(encoding="utf-8"))

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = 1
    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)

    meta = {
        "name": onnx_path.name[: -len(".onnx")],
        "sr": int(config["audio"]["sample_rate"]),
        "espeak_voice": config["espeak"]["voice"],
        "num_speakers": int(config.get("num_speakers", 1)),
        "takes_sid": any(i.name == "sid" for i in sess.get_inputs()),
    }
    return sess, config, meta


def synthesize(sess, meta, phoneme_ids, length_scale, noise_scale, noise_w, sid=0):
    """One utterance -> float32 waveform at the voice's own sample rate."""
    feeds = {
        "input": np.asarray([phoneme_ids], dtype=np.int64),
        "input_lengths": np.asarray([len(phoneme_ids)], dtype=np.int64),
        # Piper's ONNX scales vector is ordered [noise_scale, length_scale, noise_w].
        "scales": np.asarray([noise_scale, length_scale, noise_w], dtype=np.float32),
    }
    if meta["takes_sid"]:
        feeds["sid"] = np.asarray([sid], dtype=np.int64)
    audio = sess.run(None, feeds)[0]
    return np.asarray(audio, dtype=np.float32).squeeze()


def make_resampler(from_sr):
    """Same kaiser-window resampler generate_samples.py uses, so the Spanish
    clips get the identical anti-aliasing treatment as the English 50k."""
    if from_sr == SR:
        return lambda x: x
    import torch
    import torchaudio

    rs = torchaudio.transforms.Resample(
        from_sr,
        SR,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )

    def _apply(x):
        with torch.no_grad():
            return rs(torch.from_numpy(np.ascontiguousarray(x))[None, :]).numpy()[0]

    return _apply


def insert_internal_pause(audio_int16, gap_s, rng, sr=SR):
    """Split the clip at its quietest interior point and insert a pause.

    WHY.  Measured with webrtcvad on the real recordings: speaker_a's utterances span
    1.56-1.65 s while containing only ~0.85 s of voiced audio, and speaker_b's span
    2.34 s containing ~1.05 s -- i.e. the household leaves 0.7-1.3 s of pause
    *inside* the phrase.  Every synthetic positive has span == voiced duration,
    because `remove_silence` (which generate_samples applies, and so does this
    script, for consistency) concatenates the speech frames and deletes the gaps.
    So no synthetic positive has ever shown the model the phrase's real pause
    structure -- and speaker_b's 2.34 s span does not even fit inside the 1.96 s
    detection window, so for her no single window ever contains the whole phrase.

    The gap is dithered rather than exact zeros: digital silence is not a room,
    and the augment stage will mix background over it anyway.
    """
    n = len(audio_int16)
    win = int(0.03 * sr)
    lo, hi = int(0.25 * n), int(0.75 * n) - win
    if hi <= lo or gap_s <= 0:
        return audio_int16, 0.0
    step = max(1, int(0.005 * sr))
    y = np.abs(audio_int16.astype(np.float32))
    starts = np.arange(lo, hi, step)
    energy = np.array([y[s: s + win].mean() for s in starts])
    split = int(starts[int(np.argmin(energy))] + win // 2)
    gap = rng.integers(-2, 3, size=int(gap_s * sr)).astype(np.int16)
    return np.concatenate([audio_int16[:split], gap, audio_int16[split:]]), gap_s


def write_wav16k_int16(path, audio_int16):
    with wave.open(str(path), "wb") as wf:
        wf.setframerate(SR)
        wf.setsampwidth(2)
        wf.setnchannels(1)
        wf.writeframes(audio_int16.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices-dir", default="piper_voices_es")
    ap.add_argument("--voices", nargs="*", default=None,
                    help="voice basenames to use (default: every .onnx in --voices-dir)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--texts", nargs="*", default=DEFAULT_TEXTS)
    ap.add_argument("--text-weights", nargs="*", type=float, default=None,
                    help="relative share per text (default: uniform)")
    ap.add_argument("--length-scales", nargs="*", type=float,
                    default=[0.85, 1.0, 1.15, 1.3])
    ap.add_argument("--noise-scales", nargs="*", type=float, default=[0.667, 0.85, 1.0])
    ap.add_argument("--noise-scale-ws", nargs="*", type=float, default=[0.6, 0.8, 1.0])
    ap.add_argument("--pause-prob", type=float, default=0.0,
                    help="probability of inserting an internal pause (see "
                         "insert_internal_pause); recorded per clip in the manifest")
    ap.add_argument("--pause-range", nargs=2, type=float, default=[0.25, 1.10])
    ap.add_argument("--warp-prob", type=float, default=0.0,
                    help="probability of applying a VTLP formant warp (widens the "
                         "speaker space beyond the handful of Spanish voices)")
    ap.add_argument("--warp-range", nargs=2, type=float, default=[0.90, 1.10])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--manifest", default=None, help="default: <out-dir>/manifest.tsv")
    ap.add_argument("--file-prefix", default="ES")
    ap.add_argument("--min-duration", type=float, default=0.30,
                    help="drop clips shorter than this many seconds (dead generation)")
    ap.add_argument("--max-duration", type=float, default=3.5,
                    help="drop clips longer than this many seconds. VITS's stochastic "
                         "duration predictor occasionally runs away on a short phrase "
                         "(one 6.9 s clip in 11k from carlfm). Note the legitimate long "
                         "tail: the slow es_MX-ald voice plus a ~1 s internal pause "
                         "reaches 2.6-3.1 s by design, so do not set this below ~3 s.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else out_dir / "manifest.tsv"

    voices_dir = Path(args.voices_dir)
    paths = sorted(voices_dir.glob("*.onnx"))
    if args.voices:
        wanted = set(args.voices)
        available = {p.name[: -len(".onnx")] for p in paths}
        missing = wanted - available
        if missing:
            # Loudly, not silently: a typo'd voice name would otherwise just
            # shrink the voice set and skew the whole run's speaker balance.
            raise SystemExit(f"--voices names {sorted(missing)}, which are not in "
                             f"{voices_dir} (have: {sorted(available)})")
        paths = [p for p in paths if p.name[: -len(".onnx")] in wanted]
    if not paths:
        raise SystemExit(f"no matching voices in {voices_dir}")

    # (voice, speaker id) is the unit of vocal identity: sharvard is 2-speaker
    # (M/F), everything else is 1-speaker.
    voices = []
    for p in paths:
        sess, config, meta = load_voice(p, device=args.device)
        resampler = make_resampler(meta["sr"])
        for sid in range(meta["num_speakers"]):
            voices.append((sess, config, meta, resampler, sid))
    print(f"{len(paths)} voice models -> {len(voices)} (voice, speaker) identities", flush=True)

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    warp = None
    if args.warp_prob > 0:
        from speech_fx import vtlp_warp
        warp = vtlp_warp

    texts = list(args.texts)
    weights = args.text_weights or [1.0] * len(texts)
    if len(weights) != len(texts):
        raise SystemExit("--text-weights must match --texts in length")

    # Phonemise once per (voice, text): espeak output is deterministic, and the
    # id map is per-voice.
    phon_cache = {}
    for vi, (_, config, meta, _, _) in enumerate(voices):
        for text in texts:
            ids, clip_idx = get_phonemes(meta["espeak_voice"], config, text, False, None)
            # min_phoneme_count is unusable here: the repeat-and-clip trick needs
            # per-phoneme sample counts, which the ONNX graph does not expose.
            assert clip_idx is None, "unexpected clip index without min_phoneme_count"
            phon_cache[(vi, text)] = ids

    rows = []
    written = kept = dropped = 0
    with open(manifest_path, "w", encoding="utf-8") as mf:
        mf.write("file\tvoice\tsid\tespeak_voice\ttext\tlength_scale\tnoise_scale"
                 "\tnoise_w\twarp\tpause_s\tduration_s\n")
        # Bounded rather than `while kept < count`: if the duration filters reject
        # everything (a mis-set --max-duration, a broken voice) an unbounded loop
        # would spin forever writing nothing.
        max_attempts = args.count * 4 + 100
        while kept < args.count and written < max_attempts:
            vi = written % len(voices)
            sess, config, meta, resampler, sid = voices[vi]
            text = rng.choices(texts, weights=weights, k=1)[0]
            ls = rng.choice(args.length_scales)
            ns = rng.choice(args.noise_scales)
            nw = rng.choice(args.noise_scale_ws)
            written += 1

            audio = synthesize(sess, meta, phon_cache[(vi, text)], ls, ns, nw, sid=sid)
            audio = resampler(audio)

            warp_factor = 1.0
            if warp is not None and rng.random() < args.warp_prob:
                warp_factor = float(np_rng.uniform(*args.warp_range))
                audio = warp(audio, warp_factor)

            audio_int16 = audio_float_to_int16(audio)
            audio_int16 = remove_silence(audio_int16.flatten())
            if len(audio_int16) / SR < args.min_duration:
                dropped += 1
                continue

            pause_s = 0.0
            if args.pause_prob > 0 and rng.random() < args.pause_prob:
                audio_int16, pause_s = insert_internal_pause(
                    audio_int16, float(np_rng.uniform(*args.pause_range)), np_rng)
            dur = len(audio_int16) / SR
            # runaway check applies to the speech, not the deliberate pause
            if (dur - pause_s) > args.max_duration:
                dropped += 1
                continue

            name = f"{args.file_prefix}_{uuid.uuid4().hex}.wav"
            write_wav16k_int16(out_dir / name, audio_int16)
            mf.write(f"{name}\t{meta['name']}\t{sid}\t{meta['espeak_voice']}\t{text}"
                     f"\t{ls}\t{ns}\t{nw}\t{warp_factor:.4f}\t{pause_s:.4f}\t{dur:.4f}\n")
            rows.append(dur)
            kept += 1
            if kept % 500 == 0:
                print(f"  {kept}/{args.count} kept ({dropped} dropped)", flush=True)

    if kept < args.count:
        print(f"\nWARNING: only {kept} of {args.count} clips kept after "
              f"{written} attempts ({dropped} dropped by the duration filters). "
              "Check --min-duration / --max-duration.")
    if not rows:
        raise SystemExit("no clips were written -- every generation was dropped")

    durs = np.asarray(rows)
    print(f"\nWrote {kept} clips to {out_dir} "
          f"({dropped} dropped as too short or too long)")
    print(f"duration s: mean {durs.mean():.3f}  median {np.median(durs):.3f} "
          f"min {durs.min():.3f}  max {durs.max():.3f}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
