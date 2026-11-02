"""Offline speech transforms for wake-word positive augmentation.

Two things live here:

1. `vtlp_warp` / `envelope_warp` -- source-filter-aware formant shifting.  Warps
   only the *cepstrally smoothed spectral envelope* and multiplies the original
   harmonic fine structure back in, so F0 and the harmonic comb survive exactly
   while formants move.  A plain phase-vocoder pitch shift cannot do this: it
   scales F0 and formants together.

2. `lombard_transform` -- a synthetic Lombard effect.  Speaking over loud
   music/TV is not neutral speech turned up: talkers raise F0, flatten spectral
   tilt (energy shifts toward 1-4 kHz), lengthen vowels and increase vocal
   effort.  Every one of v7's synthetic positives is neutral-effort TTS, so the
   positives do not resemble how the household actually speaks over a loud TV.
   That is simulable offline, which is what this does.

Only numpy / scipy / librosa are used -- all three are already in `.venv`.  No
new pip dependency.

Nothing here is random by itself: callers pass explicit parameters (see
`sample_lombard_params` for the sampling distribution actually used).
"""
import numpy as np
import scipy.signal

SR = 16000


# --------------------------------------------------------------------------
# biquads (RBJ audio-EQ cookbook), used for the spectral-tilt stage
# --------------------------------------------------------------------------
def _peaking(fs, f0, q, gain_db):
    a_ = 10.0 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * a_, -2 * np.cos(w0), 1 - alpha * a_]
    a = [1 + alpha / a_, -2 * np.cos(w0), 1 - alpha / a_]
    return np.asarray(b) / a[0], np.asarray(a) / a[0]


def _low_shelf(fs, f0, gain_db, s=0.9):
    a_ = 10.0 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / 2 * np.sqrt((a_ + 1 / a_) * (1 / s - 1) + 2)
    cw, sa = np.cos(w0), 2 * np.sqrt(a_) * alpha
    b = [a_ * ((a_ + 1) - (a_ - 1) * cw + sa),
         2 * a_ * ((a_ - 1) - (a_ + 1) * cw),
         a_ * ((a_ + 1) - (a_ - 1) * cw - sa)]
    a = [(a_ + 1) + (a_ - 1) * cw + sa,
         -2 * ((a_ - 1) + (a_ + 1) * cw),
         (a_ + 1) + (a_ - 1) * cw - sa]
    return np.asarray(b) / a[0], np.asarray(a) / a[0]


def spectral_tilt(audio, sr=SR, boost_db=7.0, boost_hz=2200.0, boost_q=0.8,
                  cut_db=3.5, cut_hz=260.0):
    """Flatten spectral tilt: attenuate below ~`cut_hz`, boost around `boost_hz`.

    Lombard speech redistributes energy from the F0/F1 region toward 1-4 kHz.
    A low shelf cut plus a broad peaking boost reproduces that redistribution
    with a fixed, inspectable magnitude response (no learned filter, no
    per-clip fitting).
    """
    y = np.asarray(audio, dtype=np.float64)
    for b, a in (_low_shelf(sr, cut_hz, -abs(cut_db)),
                 _peaking(sr, boost_hz, boost_q, abs(boost_db))):
        y = scipy.signal.lfilter(b, a, y)
    return y.astype(np.float32)


# --------------------------------------------------------------------------
# formant / envelope warping
# --------------------------------------------------------------------------
def _piecewise_warp_map(n_bins, factor, sr=SR, boundary_hz=4800.0):
    """VTLP's piecewise-linear frequency map, in bin units.

    Below `boundary_hz` frequencies scale by `factor`; above it the map bends
    back so the Nyquist bin maps to itself and no energy is pushed off the top
    of the band. Returns, for each *output* bin, the *input* bin to read from.
    """
    nyq = sr / 2.0
    f_out = np.linspace(0.0, nyq, n_bins)
    f_bnd = min(boundary_hz, boundary_hz / max(factor, 1e-6))
    warped_bnd = f_bnd * factor
    src = np.empty_like(f_out)
    lo = f_out <= warped_bnd
    src[lo] = f_out[lo] / factor
    if warped_bnd < nyq:
        # linear from (warped_bnd -> f_bnd) up to (nyq -> nyq)
        t = (f_out[~lo] - warped_bnd) / (nyq - warped_bnd)
        src[~lo] = f_bnd + t * (nyq - f_bnd)
    return np.clip(src / nyq * (n_bins - 1), 0, n_bins - 1)


def envelope_warp(audio, factor, sr=SR, n_fft=512, hop=128, n_ceps=32):
    """Shift formants by `factor` while leaving F0 and the harmonic comb intact.

    Splits each STFT frame's log-magnitude into a cepstrally smoothed envelope
    (first `n_ceps` real-cepstrum coefficients) and the residual fine structure,
    warps only the envelope along frequency, then recombines and inverts with
    the original phase.
    """
    if abs(factor - 1.0) < 1e-6:
        return np.asarray(audio, dtype=np.float32)
    y = np.asarray(audio, dtype=np.float32)
    if len(y) < n_fft:
        return y
    f, t, spec = scipy.signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
                                   window="hann", boundary="zeros", padded=True)
    mag = np.abs(spec)
    phase = np.angle(spec)
    log_mag = np.log(mag + 1e-8)

    # cepstral smoothing along the frequency axis (per frame)
    ceps = np.fft.irfft(log_mag, axis=0)
    ceps[n_ceps:-n_ceps if n_ceps else None, :] = 0.0
    env = np.fft.rfft(ceps, n=(log_mag.shape[0] - 1) * 2, axis=0)[: log_mag.shape[0]].real
    fine = log_mag - env

    src = _piecewise_warp_map(log_mag.shape[0], factor, sr=sr)
    idx0 = np.floor(src).astype(int)
    idx1 = np.minimum(idx0 + 1, log_mag.shape[0] - 1)
    frac = (src - idx0)[:, None]
    env_w = env[idx0] * (1 - frac) + env[idx1] * frac

    new_mag = np.exp(env_w + fine)
    _, out = scipy.signal.istft(new_mag * np.exp(1j * phase), fs=sr, nperseg=n_fft,
                                noverlap=n_fft - hop, window="hann", boundary=True)
    out = out[: len(y)]
    if len(out) < len(y):
        out = np.pad(out, (0, len(y) - len(out)))
    return out.astype(np.float32)


def vtlp_warp(audio, factor, sr=SR):
    """Public alias: vocal-tract-length perturbation via `envelope_warp`."""
    return envelope_warp(audio, factor, sr=sr)


# --------------------------------------------------------------------------
# pitch and duration
# --------------------------------------------------------------------------
# 512/128 rather than librosa's 2048/512 default: these clips are ~1.5 s of
# 16 kHz speech and individual stretch segments can be a few hundred samples, so
# a 2048-point window is longer than the signal it analyses (librosa warns and
# zero-pads, which smears transients).
STFT_N_FFT = 512
STFT_HOP = 128


def pitch_shift(audio, semitones, sr=SR):
    """Phase-vocoder pitch shift. Shifts F0 *and* formants by 2**(n/12);
    `lombard_transform` compensates the formant part with `envelope_warp`."""
    if abs(semitones) < 1e-6:
        return np.asarray(audio, dtype=np.float32)
    import librosa

    return librosa.effects.pitch_shift(
        np.asarray(audio, dtype=np.float32), sr=sr, n_steps=float(semitones),
        bins_per_octave=12, n_fft=STFT_N_FFT, hop_length=STFT_HOP,
    ).astype(np.float32)


def _voiced_runs(audio, sr=SR, frame=512, hop=128, top_db=12.0, min_run=3):
    """Contiguous runs of near-peak-energy frames -- a cheap vowel proxy.

    Deliberately energy-based, not a pitch tracker: it only has to find the
    syllable nuclei to stretch, and it must be fast enough for 20k clips.

    The threshold is `top_db` below the clip's loudest frame, NOT a fixed
    percentile.  A percentile marks a constant fraction of frames hot whatever
    the clip's dynamics are, so on audio with a flat amplitude envelope it
    slices arbitrarily through near-identical values and produces fragmented
    runs shorter than one analysis window -- which are then skipped, and almost
    nothing gets stretched.  A peak-relative threshold adapts: uniformly loud
    audio is treated as all nucleus, and speech with real dynamics gets only its
    nuclei.
    """
    n = len(audio)
    starts = np.arange(0, max(1, n - frame + 1), hop)
    if len(starts) == 0:
        return []
    rms = np.sqrt(np.array([np.mean(audio[s:s + frame] ** 2) for s in starts]) + 1e-12)
    rms_db = 20.0 * np.log10(rms)
    hot = rms_db >= (rms_db.max() - top_db)
    runs, i = [], 0
    while i < len(hot):
        if hot[i]:
            j = i
            while j + 1 < len(hot) and hot[j + 1]:
                j += 1
            if j - i + 1 >= min_run:
                runs.append((int(starts[i]), int(min(n, starts[j] + frame))))
            i = j + 1
        else:
            i += 1
    return runs


def lengthen_vowels(audio, sr=SR, vowel_factor=1.25, other_factor=1.02,
                    crossfade_ms=4.0):
    """Lengthen syllable nuclei more than the rest of the clip.

    Lombard speech lengthens vowels specifically, so a flat global time-stretch
    would be the wrong shape. Segments are stretched independently and
    cross-faded at the (low-energy) joins.
    """
    import librosa

    def _stretch(x, f):
        out = librosa.effects.time_stretch(
            x, rate=1.0 / f, n_fft=STFT_N_FFT, hop_length=STFT_HOP,
        ).astype(np.float32)
        # Resize to the exact requested length. The phase vocoder's centred STFT
        # padding adds up to ~n_fft samples of overhead per call, so with several
        # segments the clip ends up longer than `vowel_factor` asked for (measured
        # 1.347x for a nominal 1.30x on a two-segment clip) and each join gains a
        # sliver of near-silence. Trimming makes the factor mean what it says.
        want = int(round(len(x) * f))
        if len(out) > want:
            return out[:want]
        if len(out) < want:
            return np.pad(out, (0, want - len(out)))
        return out

    y = np.asarray(audio, dtype=np.float32)
    runs = _voiced_runs(y, sr=sr)
    if not runs:
        if abs(other_factor - 1.0) < 1e-6:
            return y
        return _stretch(y, other_factor)

    # Build an alternating segment list: [gap, run, gap, run, ..., gap].
    # Runs must be clamped to `cur` before use. `_voiced_runs` reports frame
    # extents, and the analysis frame (512) is longer than the hop (128), so two
    # runs separated by a single quiet frame still overlap by up to 384 samples.
    # Without clamping, the overlapping stretch is emitted twice -- measured on 300
    # real positives: 14 % of clips affected, ~11 ms duplicated on each.
    segs, cur = [], 0
    for a, b in runs:
        if b <= cur:
            continue
        if a > cur:
            segs.append((cur, a, other_factor))
        segs.append((max(a, cur), b, vowel_factor))
        cur = b
    if cur < len(y):
        segs.append((cur, len(y), other_factor))

    xf = max(1, int(crossfade_ms * sr / 1000))
    out = np.zeros(0, dtype=np.float32)
    for a, b, f in segs:
        piece = y[a:b]
        if len(piece) < STFT_N_FFT or abs(f - 1.0) < 1e-6:
            # too short to phase-vocode meaningfully; leave it alone rather than
            # let librosa zero-pad it up to a full analysis window
            stretched = piece
        else:
            # librosa rate>1 speeds up, so invert to lengthen
            stretched = _stretch(piece, f)
        if len(out) == 0:
            out = stretched
            continue
        k = min(xf, len(out), len(stretched))
        if k <= 1:
            out = np.concatenate([out, stretched])
            continue
        ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
        out[-k:] = out[-k:] * (1 - ramp) + stretched[:k] * ramp
        out = np.concatenate([out, stretched[k:]])
    return out.astype(np.float32)


# --------------------------------------------------------------------------
# effort
# --------------------------------------------------------------------------
def effort_saturate(audio, drive=1.6):
    """Mild soft-knee saturation standing in for raised vocal effort.

    Raising effort adds harmonic energy and compresses the amplitude envelope.
    A tanh with modest drive does both. Level itself is not meaningful here --
    the augment stage randomises gain anyway -- so the output is peak
    re-normalised to the input's peak.
    """
    y = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(y))) or 1.0
    out = np.tanh(y / peak * float(drive)) / np.tanh(float(drive))
    return (out * peak).astype(np.float32)


# --------------------------------------------------------------------------
# the composed transform
# --------------------------------------------------------------------------
DEFAULT_LOMBARD = dict(
    semitones=2.5,       # F0 raise, semitones (+1..+4 is the literature range)
    formant_factor=1.03,  # residual formant raise on top of F0 compensation
    vowel_factor=1.22,   # syllable-nucleus lengthening
    other_factor=1.03,   # everything else barely moves
    tilt_boost_db=7.0,
    tilt_boost_hz=2200.0,
    tilt_cut_db=3.5,
    drive=1.6,
)


def lombard_transform(audio, sr=SR, semitones=2.5, formant_factor=1.03,
                      vowel_factor=1.22, other_factor=1.03, tilt_boost_db=7.0,
                      tilt_boost_hz=2200.0, tilt_cut_db=3.5, drive=1.6):
    """Neutral-effort speech -> synthetic Lombard speech.

    Stage order follows the production chain: duration (prosody), then source
    F0, then the vocal-tract filter, then effort.

    The formant stage warps by `formant_factor * 2**(-semitones/12)`: the pitch
    shift already dragged formants up by `2**(semitones/12)`, so this undoes
    that and leaves exactly the intended `formant_factor` raise. Without it a
    +2.5 st shift would move formants 16 %, which is a different speaker, not a
    louder one.
    """
    y = np.asarray(audio, dtype=np.float32)
    if len(y) < 256:
        return y
    y = lengthen_vowels(y, sr=sr, vowel_factor=vowel_factor, other_factor=other_factor)
    y = pitch_shift(y, semitones, sr=sr)
    y = envelope_warp(y, formant_factor * 2.0 ** (-semitones / 12.0), sr=sr)
    y = spectral_tilt(y, sr=sr, boost_db=tilt_boost_db, boost_hz=tilt_boost_hz,
                      cut_db=tilt_cut_db)
    y = effort_saturate(y, drive=drive)
    peak = float(np.max(np.abs(y)))
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


def sample_lombard_params(rng):
    """Draw one clip's Lombard parameters.

    Ranges are deliberately a spread, not a point: real Lombard response scales
    with the interfering level, so training material should cover mild through
    strong rather than one canonical setting.
    """
    return dict(
        semitones=float(rng.uniform(1.0, 4.0)),
        formant_factor=float(rng.uniform(1.00, 1.06)),
        vowel_factor=float(rng.uniform(1.10, 1.35)),
        other_factor=float(rng.uniform(1.00, 1.06)),
        tilt_boost_db=float(rng.uniform(4.0, 9.0)),
        tilt_boost_hz=float(rng.uniform(1800.0, 2800.0)),
        tilt_cut_db=float(rng.uniform(2.0, 5.0)),
        drive=float(rng.uniform(1.2, 2.2)),
    )
