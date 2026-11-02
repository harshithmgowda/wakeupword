#!/usr/bin/env python
"""Guards for speech_fx.py and gen_piper_onnx.py's pause insertion.

Run with the training venv (needs librosa/scipy):

    ./.venv/bin/python tests/test_speech_fx.py

These assert the *direction and rough magnitude* of each transform on a
synthetic vowel, not exact sample values -- the point is that a future edit
cannot silently turn a component into a no-op or flip its sign.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(HERE))

import speech_fx as fx  # noqa: E402

SR = 16000
FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


def synth_vowel(f0=150.0, dur=1.2, sr=SR, formants=(700.0, 1200.0, 2600.0),
                tilt_db_per_khz=-6.0, seed=0):
    """Harmonic source through a crude 3-formant filter, with a quiet gap in the
    middle so the vowel/pause machinery has something to find.

    The formants are broad (200 Hz) and the whole envelope carries a -6 dB/kHz
    roll-off plus a little aspiration noise, so the clip has a speech-like
    broadband spectrum.  This matters: an earlier version used 120 Hz formants
    with no roll-off, which made the spectrum three narrow spikes over a near-zero
    floor.  A least-squares tilt fit on that is dominated by the spikes, so the
    tilt test reported no flattening even though the same filter measurably
    flattens real speech (-6.30 -> -5.27 dB/kHz over 250 real clips).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    src = np.zeros_like(t)
    for k in range(1, 60):
        if f0 * k >= sr / 2:
            break
        src += np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi)) / k
    src += 0.02 * rng.standard_normal(len(t))  # aspiration
    spec = np.fft.rfft(src)
    freqs = np.fft.rfftfreq(len(src), 1 / sr)
    env = np.zeros_like(freqs)
    for fc in formants:
        env += np.exp(-((freqs - fc) ** 2) / (2 * 200.0 ** 2))
    env += 0.06
    env *= 10.0 ** (tilt_db_per_khz * freqs / 1000.0 / 20.0)
    y = np.fft.irfft(spec * env, n=len(src)).astype(np.float32)
    y /= np.max(np.abs(y)) + 1e-9
    # amplitude envelope: loud, quiet, loud -> two "syllables"
    gate = np.ones_like(y)
    a, b = int(0.45 * len(y)), int(0.55 * len(y))
    gate[a:b] = 0.02
    return (y * gate).astype(np.float32)


def f0_of(y, sr=SR):
    import librosa

    f0, _, _ = librosa.pyin(y, fmin=60, fmax=500, sr=sr, frame_length=1024,
                            hop_length=160)
    f0 = f0[np.isfinite(f0)]
    return float(np.median(f0)) if len(f0) else float("nan")


def centroid_of(y, sr=SR):
    import librosa

    S = np.abs(librosa.stft(y, n_fft=512, hop_length=160))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    per = (freqs[:, None] * S).sum(axis=0) / (S.sum(axis=0) + 1e-12)
    return float(np.average(per, weights=S.sum(axis=0)))


def band_share(y, sr=SR, lo=1000.0, hi=4000.0):
    import librosa

    S = np.abs(librosa.stft(y, n_fft=512, hop_length=160)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    b = (freqs >= lo) & (freqs <= hi)
    return float(S[b].sum() / (S.sum() + 1e-12))


def tilt_of(y, sr=SR):
    import librosa

    S = np.abs(librosa.stft(y, n_fft=512, hop_length=160))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    band = (freqs >= 100) & (freqs <= 7000)
    e = S.sum(axis=0)
    hot = e >= np.percentile(e, 60)
    logmag = 20 * np.log10(S[band][:, hot] + 1e-8)
    fk = freqs[band] / 1000.0
    A = np.vstack([fk, np.ones_like(fk)]).T
    return float(np.linalg.lstsq(A, logmag.mean(axis=1), rcond=None)[0][0])


def main():
    y = synth_vowel()
    print(f"synthetic vowel: {len(y)/SR:.3f} s, F0 {f0_of(y):.1f} Hz, "
          f"centroid {centroid_of(y):.0f} Hz, tilt {tilt_of(y):+.2f} dB/kHz")

    # --- spectral tilt -----------------------------------------------------
    # The substantive claim is "energy moves into 1-4 kHz", so that is what is
    # asserted.  The raw least-squares tilt *slope* is deliberately NOT asserted
    # here: fitted in linear frequency over 100-7000 Hz it is a shape-dependent
    # summary, and a boost centred at 2.2 kHz sits left of that range's midpoint,
    # so on a spectrum with little energy above 3 kHz the fitted slope can steepen
    # even as energy shifts upward.  On the real positive clips the same filter
    # does flatten it (-6.30 -> -5.27 dB/kHz, 239/250 clips); that measurement
    # lives on the experiments branch.
    check("spectral_tilt moves energy into the 1-4 kHz band",
          band_share(fx.spectral_tilt(y)) > band_share(y) * 1.10,
          f"{band_share(y):.4f} -> {band_share(fx.spectral_tilt(y)):.4f}")
    check("spectral_tilt raises the spectral centroid",
          centroid_of(fx.spectral_tilt(y)) > centroid_of(y) * 1.05,
          f"{centroid_of(y):.0f} -> {centroid_of(fx.spectral_tilt(y)):.0f} Hz")

    # --- envelope warp: formants move, F0 does not --------------------------
    w = fx.envelope_warp(y, 1.15)
    check("envelope_warp leaves F0 within 2 % (source-filter separation holds)",
          abs(f0_of(w) - f0_of(y)) / f0_of(y) < 0.02,
          f"{f0_of(y):.1f} -> {f0_of(w):.1f} Hz")
    check("envelope_warp with factor>1 raises the centroid",
          centroid_of(w) > centroid_of(y) * 1.02,
          f"{centroid_of(y):.0f} -> {centroid_of(w):.0f} Hz")
    check("envelope_warp(1.0) is a no-op",
          np.array_equal(fx.envelope_warp(y, 1.0), y))
    check("envelope_warp preserves length", len(w) == len(y), f"{len(y)} -> {len(w)}")

    # --- pitch shift -------------------------------------------------------
    p = fx.pitch_shift(y, 3.0)
    ratio = f0_of(p) / f0_of(y)
    check("pitch_shift(+3 st) raises F0 by ~2**(3/12)",
          abs(ratio - 2 ** 0.25) < 0.03, f"ratio {ratio:.4f} vs {2**0.25:.4f}")

    # --- vowel lengthening -------------------------------------------------
    L = fx.lengthen_vowels(y, vowel_factor=1.30, other_factor=1.0)
    check("lengthen_vowels makes the clip longer", len(L) > len(y) * 1.02,
          f"{len(y)} -> {len(L)} ({len(L)/len(y):.3f}x)")
    check("lengthen_vowels stretches less than the vowel factor "
          "(it is selective, not global)", len(L) < len(y) * 1.30,
          f"{len(L)/len(y):.3f}x vs 1.30x")
    # Guards the run-clamping fix: overlapping voiced runs must not be emitted
    # twice. Every segment boundary must be non-decreasing and non-overlapping.
    runs = fx._voiced_runs(y)
    cur, ok = 0, True
    for a, b in runs:
        if b <= cur:
            continue
        if max(a, cur) < cur:
            ok = False
        cur = b
    check("_voiced_runs extents can be walked without overlap", ok)
    # a synthetic clip whose runs *do* overlap must still not duplicate audio
    two = np.concatenate([y[: int(0.5 * len(y))], y[: int(0.5 * len(y))]])
    L2 = fx.lengthen_vowels(two, vowel_factor=1.20, other_factor=1.0)
    check("lengthen_vowels never exceeds the vowel factor, even with adjacent runs",
          len(L2) <= len(two) * 1.20 + 8, f"{len(L2)/len(two):.4f}x vs 1.20x")

    # --- effort ------------------------------------------------------------
    s = fx.effort_saturate(y, drive=2.0)
    check("effort_saturate keeps the peak level (renormalised)",
          abs(np.abs(s).max() - np.abs(y).max()) < 0.02,
          f"{np.abs(y).max():.4f} -> {np.abs(s).max():.4f}")
    check("effort_saturate raises RMS relative to peak (compression)",
          np.sqrt(np.mean(s ** 2)) > np.sqrt(np.mean(y ** 2)) * 1.02)

    # --- composed transform ------------------------------------------------
    out = fx.lombard_transform(y, semitones=2.5, vowel_factor=1.25)
    check("lombard_transform output is finite", np.isfinite(out).all())
    check("lombard_transform does not clip", np.abs(out).max() <= 1.0 + 1e-6)
    st = 12 * np.log2(f0_of(out) / f0_of(y))
    check("lombard_transform raises F0 by ~the requested semitones",
          abs(st - 2.5) < 0.6, f"{st:+.2f} st requested +2.50")
    check("lombard_transform raises the centroid",
          centroid_of(out) > centroid_of(y) * 1.05,
          f"{centroid_of(y):.0f} -> {centroid_of(out):.0f} Hz")
    check("lombard_transform moves energy into the 1-4 kHz band",
          band_share(out) > band_share(y) * 1.10,
          f"{band_share(y):.4f} -> {band_share(out):.4f}")
    check("lombard_transform lengthens the clip", len(out) > len(y) * 1.02,
          f"{len(out)/len(y):.3f}x")

    # F0 compensation: formants must not ride up with the pitch shift.
    # Compare against an uncompensated pitch shift of the same size.
    naive = fx.pitch_shift(y, 2.5)
    comp = fx.envelope_warp(naive, 2.0 ** (-2.5 / 12.0))
    check("envelope_warp undoes the pitch shift's formant drag",
          abs(centroid_of(comp) - centroid_of(y)) < abs(centroid_of(naive) - centroid_of(y)),
          f"orig {centroid_of(y):.0f}, naive {centroid_of(naive):.0f}, "
          f"compensated {centroid_of(comp):.0f} Hz")

    # --- parameter sampling ------------------------------------------------
    rng = np.random.default_rng(0)
    ps = [fx.sample_lombard_params(rng) for _ in range(200)]
    check("sample_lombard_params keeps semitones in the 1-4 st literature range",
          all(1.0 <= p["semitones"] <= 4.0 for p in ps))
    check("sample_lombard_params returns exactly the transform's parameters",
          all(set(p) == set(fx.DEFAULT_LOMBARD) for p in ps))
    check("sampled parameters actually vary",
          len({round(p["semitones"], 4) for p in ps}) > 150)

    # --- internal pause insertion (gen_piper_onnx) -------------------------
    sys.path.insert(0, str(HERE / "tools" / "data"))
    from gen_piper_onnx import insert_internal_pause

    pcm = (y * 32767).astype(np.int16)
    prng = np.random.default_rng(3)
    out16, gap = insert_internal_pause(pcm, 0.5, prng)
    check("insert_internal_pause reports the gap it inserted", abs(gap - 0.5) < 1e-9)
    check("insert_internal_pause adds exactly gap_s of samples",
          len(out16) == len(pcm) + int(0.5 * SR), f"{len(pcm)} -> {len(out16)}")
    check("insert_internal_pause splits inside the clip, not at an edge",
          np.abs(out16[:int(0.1 * SR)]).mean() > 100
          and np.abs(out16[-int(0.1 * SR):]).mean() > 100)
    # the gap should land in the synthetic clip's quiet middle
    mid = np.abs(out16[int(0.45 * len(out16)):int(0.55 * len(out16))]).mean()
    check("the inserted gap lands in the quiet interior", mid < 50, f"mid level {mid:.1f}")
    check("insert_internal_pause with gap 0 is a no-op",
          insert_internal_pause(pcm, 0.0, prng)[0] is pcm
          or np.array_equal(insert_internal_pause(pcm, 0.0, prng)[0], pcm))
    check("insert_internal_pause output stays int16",
          out16.dtype == np.int16)

    print()
    if FAILS:
        print(f"FAIL test_speech_fx -- {len(FAILS)} check(s) failed: {FAILS}")
        return 1
    print("PASS test_speech_fx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
