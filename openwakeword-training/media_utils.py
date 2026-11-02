"""Shared audio/feature helpers for the v6 music-robustness pipeline."""
import numpy as np
import scipy.signal
import soundfile as sf

SR = 16000


def read_wav16k(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        audio = scipy.signal.resample_poly(audio, SR // g, int(sr) // g)
    return audio.astype(np.float32)


def write_wav16k(path, audio):
    sf.write(path, (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16), SR,
             subtype="PCM_16")


def rms(audio):
    return float(np.sqrt(np.mean(np.square(audio))) + 1e-12)


def pick_segment(audio, n_samples, rng):
    if len(audio) < n_samples:
        reps = int(np.ceil(n_samples / max(1, len(audio)))) + 1
        audio = np.tile(audio, reps)
    start = int(rng.integers(0, len(audio) - n_samples + 1))
    return audio[start:start + n_samples]


def mix_at_snr(fg, bg, snr_db):
    """Mix so fg sits snr_db above bg. Peak-normalize to <=0.95 if clipping."""
    bg = bg[:len(fg)]
    scale = (rms(fg) / rms(bg)) * (10.0 ** (-snr_db / 20.0))
    mixed = fg + bg * scale
    peak = np.abs(mixed).max()
    if peak > 0.95:
        mixed = mixed * (0.95 / peak)
    return mixed.astype(np.float32)


#: Default ThreadPool width for embed().
#:
#: READ THIS BEFORE CHANGING IT.  There are two knobs in openwakeword called
#: `ncpu` and only ONE of them is safe:
#:
#:   * embed_clips(x, ncpu=K)      -> a ThreadPool over *independent* per-clip /
#:                                    per-window onnxruntime calls.  Provably
#:                                    BIT-IDENTICAL for any K.  This is it.
#:   * AudioFeatures(ncpu=K)       -> onnxruntime intra_op/inter_op thread
#:                                    counts.  CHANGES THE COMPUTED FEATURES.
#:                                    Never route this constant there.
#:
#: Measured on this box (24 cores, idle, .venv-onnx2tf, CPUExecutionProvider),
#: one pass per K after a warm-up pass, max_abs_diff vs K=1 is 0.0 at every K:
#:
#:   20 x 30 s eval_media/music clips (599.8 s of audio)
#:     K=1  7.789 s | K=4 2.629 | K=8 1.753 | K=12 1.449 | K=16 1.472 | K=24 1.420
#:   27 x ~3 s ~/openwakeword-train-data/eval clips (81.0 s of audio)
#:     K=1  0.896 s | K=4 0.378 | K=8 0.370 | K=12 0.358 | K=16 0.384 | K=24 0.460
#:
#: 12 is chosen because it is at/near the optimum for both clip lengths; 24
#: regresses on short clips (0.460 s vs 0.358 s) where per-call pool setup is a
#: larger share.  Speedup over the previous default of 1: 5.38x on 30 s clips,
#: 2.50x on the short eval clips.
EMBED_NCPU = 12


def embed(feat, audio, ncpu=EMBED_NCPU):
    """AudioFeatures embedding for one float32 clip -> (frames, 96).

    `ncpu` is embed_clips' ThreadPool width (bit-identical for any value) --
    see EMBED_NCPU.  Multi-process callers should pass ncpu=1 so that N
    processes do not each spawn EMBED_NCPU threads.
    """
    audio16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    return feat.embed_clips(audio16[None, :], batch_size=64, ncpu=ncpu)[0]


def windows16(emb, stride=1):
    if emb.shape[0] < 16:
        return np.empty((0, 16, 96), dtype=emb.dtype)
    w = np.lib.stride_tricks.sliding_window_view(emb, 16, axis=0)[::stride]
    return np.ascontiguousarray(w.transpose(0, 2, 1))
