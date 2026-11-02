#!/usr/bin/env python
"""Step 2: download and prepare all training data.

Every stage checks for its own output first, so re-running after an
interruption only redoes what's missing.

    ./.venv/bin/python 02_download_data.py                  # medium scale (default)
    ./.venv/bin/python 02_download_data.py --audioset-shards 4 --music-hours 2
    ./.venv/bin/python 02_download_data.py --only audioset  # single stage

Notes on where this departs from the Colab notebook:
  * AudioSet moved from `bal_trainNN.tar` (flac) to `data/bal_train/NN.parquet`
    with the audio embedded as flac bytes. The notebook's tar URLs now 404.
  * FMA is pulled straight from the upstream switch.ch zip. The HF loader the
    notebook used ("rudraml/fma") is a legacy script-dataset that only wraps
    that same zip and no longer loads on modern `datasets`.
Both changes let us drop the `datasets==2.14.6` pin entirely.
"""
import argparse
import io
import os
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import scipy.signal
import soundfile as sf
from tqdm import tqdm

HERE = Path(__file__).parent.resolve()
DL = HERE / "downloads"
SR = 16000

AUDIOSET_URL = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/bal_train/{:02d}.parquet"
FMA_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
RIR_REPO = "https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses"
FEATURES = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/{}"

SECONDS_PER_FMA_CLIP = 30


def sh(cmd, **kw):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kw)


def wget(url, dest):
    """Resumable download; skips if the file is already complete."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # A progress bar redirected to a log file is thousands of useless lines;
    # fall back to one dot per 100 MB when stdout isn't a terminal.
    progress = "bar:force:noscroll" if sys.stdout.isatty() else "dot:giga"
    sh(["wget", "-c", f"--progress={progress}", "-O", str(dest), url])
    return dest


def to_16k_mono(audio, sr):
    """Resample to 16 kHz mono and return int16 samples."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        audio = scipy.signal.resample_poly(audio, SR // g, int(sr) // g)
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


# --------------------------------------------------------------------------
# Room impulse responses (MIT)
# --------------------------------------------------------------------------
def stage_rirs():
    out = HERE / "mit_rirs"
    if out.exists() and any(out.glob("*.wav")):
        print(f"[rirs] {len(list(out.glob('*.wav')))} files present, skipping")
        return
    print("[rirs] downloading MIT environmental impulse responses")
    repo = DL / "MIT_environmental_impulse_responses"
    if not repo.exists():
        sh(["git", "lfs", "install"])
        sh(["git", "clone", RIR_REPO, str(repo)])
    out.mkdir(parents=True, exist_ok=True)
    srcs = sorted((repo / "16khz").glob("*.wav"))
    for src in tqdm(srcs, desc="[rirs] converting"):
        audio, sr = sf.read(src, dtype="float32")
        scipy.io.wavfile.write(out / src.name, SR, to_16k_mono(audio, sr))
    print(f"[rirs] wrote {len(srcs)} impulse responses -> {out}")


# --------------------------------------------------------------------------
# AudioSet background noise
# --------------------------------------------------------------------------
def _convert_audioset_shard(args):
    shard_path, out_dir = args
    import pyarrow.parquet as pq

    n = 0
    pf = pq.ParquetFile(shard_path)
    for batch in pf.iter_batches(batch_size=32):
        for row in batch.to_pylist():
            name = f"{row['video_id']}.wav"
            dest = Path(out_dir) / name
            if dest.exists():
                continue
            try:
                audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            except Exception:
                continue  # a handful of clips in the set are truncated
            scipy.io.wavfile.write(dest, SR, to_16k_mono(audio, sr))
            n += 1
    return n


def stage_audioset(n_shards):
    out = HERE / "audioset_16k"
    out.mkdir(parents=True, exist_ok=True)
    have = len(list(out.glob("*.wav")))
    # each shard holds 500 ten-second clips
    if have >= n_shards * 500 * 0.95:
        print(f"[audioset] {have} clips present, skipping")
        return

    shards = []
    for i in range(n_shards):
        p = DL / f"audioset_{i:02d}.parquet"
        if not p.exists():
            print(f"[audioset] downloading shard {i + 1}/{n_shards}")
            wget(AUDIOSET_URL.format(i), p)
        shards.append(p)

    print(f"[audioset] decoding {len(shards)} shards -> 16 kHz wav")
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        total = sum(tqdm(ex.map(_convert_audioset_shard, [(str(s), str(out)) for s in shards]),
                         total=len(shards), desc="[audioset] shards"))
    print(f"[audioset] wrote {total} new clips ({len(list(out.glob('*.wav')))} total) -> {out}")


# --------------------------------------------------------------------------
# Free Music Archive
# --------------------------------------------------------------------------
def _convert_mp3(args):
    data, dest = args
    # ffmpeg is far more tolerant of FMA's occasional malformed mp3 headers
    # than any python decoder.
    p = subprocess.run(
        ["ffmpeg", "-loglevel", "quiet", "-i", "pipe:0", "-ac", "1", "-ar", str(SR),
         "-acodec", "pcm_s16le", "-f", "wav", "-y", dest],
        input=data, capture_output=True,
    )
    return p.returncode == 0


def stage_fma(hours):
    out = HERE / "fma"
    out.mkdir(parents=True, exist_ok=True)
    want = int(hours * 3600 // SECONDS_PER_FMA_CLIP)
    have = len(list(out.glob("*.wav")))
    if have >= want * 0.95:
        print(f"[fma] {have} clips present (~{have * SECONDS_PER_FMA_CLIP / 3600:.1f} h), skipping")
        return

    zip_path = DL / "fma_small.zip"
    if not zip_path.exists():
        print("[fma] downloading fma_small.zip (7.7 GB)")
        wget(FMA_URL, zip_path)

    print(f"[fma] extracting {want} clips (~{hours} h)")
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".mp3")]
        # Spread the selection across the archive: consecutive files are
        # same-genre/same-artist, and we want varied music as negatives.
        step = max(1, len(names) // want)
        picked = names[::step][:want]
        jobs = []
        for n in picked:
            dest = out / (Path(n).stem + ".wav")
            if dest.exists():
                continue
            jobs.append((z.read(n), str(dest)))
            if len(jobs) >= 64:
                with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
                    list(ex.map(_convert_mp3, jobs))
                jobs = []
                print(f"  ...{len(list(out.glob('*.wav')))}/{want}", end="\r", flush=True)
        if jobs:
            with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
                list(ex.map(_convert_mp3, jobs))
    n = len(list(out.glob("*.wav")))
    print(f"[fma] wrote {n} clips (~{n * SECONDS_PER_FMA_CLIP / 3600:.1f} h) -> {out}")


# --------------------------------------------------------------------------
# Pre-computed openWakeWord features
# --------------------------------------------------------------------------
def stage_features():
    targets = {
        "openwakeword_features_ACAV100M_2000_hrs_16bit.npy": 17280000128,
        "validation_set_features.npy": 184836608,
    }
    for name, size in targets.items():
        final = HERE / name
        if final.exists() and final.stat().st_size == size:
            print(f"[features] {name} present, skipping")
            continue
        staged = DL / name
        if not (staged.exists() and staged.stat().st_size == size):
            print(f"[features] downloading {name} ({size / 1e9:.1f} GB)")
            wget(FEATURES.format(name), staged)
        if staged.stat().st_size != size:
            sys.exit(f"[features] {name} is truncated -- re-run to resume the download")
        shutil.move(str(staged), str(final))
        print(f"[features] ready: {final}")


STAGES = {
    "rirs": lambda a: stage_rirs(),
    "audioset": lambda a: stage_audioset(a.audioset_shards),
    "fma": lambda a: stage_fma(a.music_hours),
    "features": lambda a: stage_features(),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audioset-shards", type=int, default=10,
                    help="parquet shards of 500 x 10s clips each (max 38, default 10 ~= 14 h)")
    ap.add_argument("--music-hours", type=float, default=10,
                    help="hours of music to extract from FMA (default 10)")
    ap.add_argument("--only", choices=list(STAGES), action="append",
                    help="run only these stages (repeatable)")
    args = ap.parse_args()

    DL.mkdir(parents=True, exist_ok=True)
    for name in (args.only or list(STAGES)):
        print(f"\n=== {name} ===")
        STAGES[name](args)
    print("\nAll data ready.")


if __name__ == "__main__":
    main()
