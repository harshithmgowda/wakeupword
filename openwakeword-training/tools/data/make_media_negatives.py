#!/usr/bin/env python
"""openWakeWord features of music/TV audio -> negative training data.

    ./.venv/bin/python tools/data/make_media_negatives.py --out media_negatives.npy

Performance note (deviation from the task-8 brief, see task-8-report.md):
mine_hard_negatives.py embeds the same fma/ and tv_mix/ files this script
does (that script additionally covers audioset_16k/). Embedding -- not the
stride-8/stride-1 windowing, which is cheap numpy slicing -- is the
dominant cost, so each file's frame-level embedding is computed once and
cached to disk under .embed_cache/ (float32, keyed by source path).
Whichever of the two scripts runs first pays the embedding cost for the
files it shares with the other; that other script then loads the cached
embedding instead of recomputing it. This mirrors the caching pattern
already used in eval_music.py (task 7) for the same reason. Output shape,
dtype, and CLI are unchanged from the brief.

Parallelism note (deviation, requested mid-run -- see task-8-report.md): a
first single-process run showed openwakeword's onnxruntime-backed
AudioFeatures embedding pinned at ~100% of one core (no CUDAExecutionProvider
is installed in .venv, so `device="gpu"` silently falls back to CPU inside
onnxruntime -- see below), on a 24-core box, with ~34 h of audio still
ahead. Embedding is now spread across a ProcessPoolExecutor of worker
processes, each constructing its own AudioFeatures with the *same*
device="gpu" if torch.cuda.is_available() else "cpu" expression as before
(unchanged config/venv/dtype -- only the process topology changed).
Verified numerically identical before switching over: embedding the same
file via a single main-process AudioFeatures vs. via a forked worker's
fresh AudioFeatures produced bit-for-bit identical (16, 96) float32 arrays
(np.array_equal, max abs diff 0.0), including under concurrent execution
across 10 workers on 30 files. Determinism given --seed is preserved by
consuming ProcessPoolExecutor.map() in submission order (it yields results
in input order regardless of completion order), so windows are
concatenated in the same fixed per-directory, sorted-filename order as the
sequential version -- required because np.random.default_rng(seed).choice
during cap-subsampling indexes into that concatenation, not just its length.

GPU note: onnxruntime in .venv only has CPUExecutionProvider/
AzureExecutionProvider (no onnxruntime-gpu wheel installed); adding one
would be a new pip dependency, which is out of scope here, so embedding is
CPU-bound by design and process-level parallelism is the available lever.
"""
import sys
import argparse
import concurrent.futures as cf
import os
from pathlib import Path

import numpy as np
import torch
import openwakeword.utils

# python puts the *script's* directory on sys.path, not the cwd, so the
# repo-root modules below have to be found explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from media_utils import read_wav16k, embed, windows16

#: Base directory for the per-file embedding cache. The entries actually used
#: live in a PROVENANCE-SCOPED subdirectory chosen by
#: embed_cache.scoped_cache_dir() at startup -- see _resolve_cache_dir below.
CACHE_BASE = Path(".embed_cache")
#: Rebound in main() before any worker is forked, so workers inherit it.
CACHE_DIR = CACHE_BASE
N_WORKERS = min(10, os.cpu_count() or 1)


def atomic_save(path, arr):
    """Write `arr` to `path` atomically.

    Hard requirement (see task-8-report.md): a prior run of this script was
    killed mid-write and left a media_negatives.npy whose header claimed a
    full shape but whose bytes were 33% truncated -- np.load couldn't even
    open it, and the file sat there looking like a real artifact until
    someone tried to use it. Writing straight to `path` (the brief's
    literal np.save(args.out, data)) is exactly that failure mode. Instead:
    write the complete array to a temp file in the same directory (so the
    final os.replace is on the same filesystem and atomic), fsync it so the
    bytes are durable even across a hard kill, and only then swap it into
    place. A kill at any point before the replace leaves `path` untouched
    (either absent or the previous good version); a kill after leaves the
    new file fully written. There is never a partially-written file at the
    final path.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            np.save(f, arr)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _valid_cache_array(emb, dtype=np.float32, feat_dim=96):
    """Cheap sanity check on a loaded cache entry: right rank/dtype/last
    dim, at least one frame, all finite. This is defense in depth, not the
    primary fix -- see the Fix round 1 note in `cached_embed` below.
    Deliberately does *not* check the frame count against the source
    audio's duration (would require re-reading/decoding the wav, defeating
    the point of the cache); a torn read that happens to produce a
    wrong-but-plausibly-shaped array of the *same* frame count would slip
    past this, which is exactly why the write side must be atomic too."""
    return (emb.ndim == 2 and emb.shape[0] > 0 and emb.shape[1] == feat_dim
            and emb.dtype == dtype and np.isfinite(emb).all())


def cached_embed(feat, path):
    """Frame-level embedding for `path`, cached to disk across script runs
    and across worker processes (all workers share CACHE_DIR).

    Self-healing: a cache file that fails to load, or loads but fails
    `_valid_cache_array`, is silently recomputed and overwritten rather
    than raised. Safe under parallel workers within one script run: each
    file is assigned to exactly one worker task, so there is no concurrent
    writer for the same cache key *within a run*.

    Fix round 1 (see task-8-report.md): this used to write with a bare
    `np.save(key, emb)`. A code-review pass correctly identified that as
    the real root cause of a corrupted row observed during development --
    not "IPC corruption" as originally (wrongly) diagnosed in this task's
    report. `windows16()`/`.astype()` return freshly-copied arrays and
    ProcessPoolExecutor's pickle transport doesn't silently corrupt without
    raising, so the parallel-aggregation path was never the culprit; a
    concurrently-running process (this box was not actually idle -- see
    the report) writing that same cache entry via a non-atomic `np.save`
    while this process's `np.load(key)` was reading it is the mechanically
    plausible explanation, and is fixed by routing the write through
    `atomic_save()` (temp file + fsync + os.replace) so any reader always
    sees either the complete old file or the complete new one, never a
    partial one. `_valid_cache_array` above is an additional, independent
    guard against any *other* source of a corrupted-but-parseable cache
    entry (e.g. a future write path that bypasses `atomic_save`).
    """
    key = CACHE_DIR / (str(path).replace("/", "__") + ".npy")
    if key.exists():
        try:
            emb = np.load(key)
            if _valid_cache_array(emb):
                return emb
        except Exception:
            pass
    # ncpu=1 on purpose: this runs inside N_WORKERS worker processes, so the
    # media_utils.EMBED_NCPU default (12, tuned for the single-process eval
    # path) would mean N_WORKERS x 12 threads fighting over 24 cores. The value
    # is a ThreadPool width only -- bit-identical at any setting -- so this
    # changes speed, never the cached numbers.
    emb = embed(feat, read_wav16k(str(path)), ncpu=1).astype(np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_save(key, emb)
    return emb


def verify_against_cache(ok_paths, stride, data):
    """Rebuild the window set single-threaded, straight from the (now
    fully-populated) on-disk embed cache, and assert it matches the
    ProcessPoolExecutor-collected `data` bit-for-bit.

    Why this exists (see task-8-report.md): during development, one run's
    parallel-collected array had a single row (out of 105556) that did not
    match a hand reconstruction from the same on-disk cache, and did not
    equal *any* row produced by re-running the same code. This check turns
    that class of mismatch -- the parallel-collected result disagreeing
    with what a clean rebuild produces -- into a loud failure before
    anything is written to `--out`, which is valuable regardless of root
    cause (e.g. a future regression in the aggregation logic, an off-by-one
    in the ordering, etc).

    Fix round 1 correction (see task-8-report.md): the original diagnosis
    here ("genuine corruption in the multiprocessing result path") was
    wrong and has been retracted -- a code-review pass traced the actual
    mechanism to a non-atomic write in `cached_embed`'s cache (fixed
    above). Note the resulting caveat: for *that specific* failure mode,
    this check is not a reliable safety net, because both the parallel
    path and this function's "ground truth" rebuild read the same on-disk
    cache file -- if a torn write had left a persistently wrong-but-valid-
    looking cache entry (rather than resolving before this function runs,
    which is what happened to get caught the one time this was observed),
    the two reads could agree with each other while both being wrong. The
    actual fix for that failure mode is the cache write now being atomic
    (`atomic_save` in `cached_embed`) plus `_valid_cache_array` validating
    what gets read back. This function remains useful as a check on the
    aggregation path itself, just not as the primary defense against a
    corrupted cache.
    """
    chunks = []
    for f in ok_paths:
        key = CACHE_DIR / (str(f).replace("/", "__") + ".npy")
        emb = np.load(key)
        w = windows16(emb, stride=stride)
        if len(w):
            chunks.append(w.astype(np.float16))
    truth = np.concatenate(chunks) if chunks else np.empty((0, 16, 96), np.float16)
    if truth.shape != data.shape or not np.array_equal(truth, data):
        n_diff = ("shape mismatch" if truth.shape != data.shape
                  else f"{int((truth != data).any(axis=(1, 2)).sum())} differing rows")
        raise SystemExit(
            f"[negatives] FATAL: parallel-collected data does not match a "
            f"single-threaded rebuild from the same cache ({n_diff}); "
            f"refusing to save a possibly-corrupted array.")
    print(f"[negatives] verified: parallel result matches single-threaded "
          f"cache rebuild ({len(truth)} rows)")


def gather_files(dirs):
    """Per-directory sorted .wav listing, in `dirs` order. Hard error on a
    missing directory (see module docstring)."""
    per_dir_found = {}
    files = []
    for d in dirs:
        dp = Path(d)
        if not dp.is_dir():
            raise SystemExit(f"[negatives] ERROR: directory not found: {d}")
        found = sorted(dp.glob("*.wav"))
        if not found:
            print(f"[negatives] WARNING: no .wav files in {d}")
        per_dir_found[d] = len(found)
        files.extend(found)
    return files, per_dir_found


def _worker_device():
    """The device _init_worker will build its AudioFeatures on. Kept in one
    place so the cache identity computed in the parent matches the workers'."""
    return "gpu" if torch.cuda.is_available() else "cpu"


def _worker_provenance():
    """The feature provenance AS SEEN INSIDE A FORKED WORKER.

    THIS MUST NOT BE PROBED IN THE PARENT. Measured on this box: the parent
    process builds AudioFeatures(device="gpu") and really does get
    CUDAExecutionProvider, but every forked worker fails
    `cudaSetDevice` with "CUDA failure 3: initialization error" (a CUDA
    context cannot be inherited across fork) and onnxruntime SILENTLY falls
    back to CPUExecutionProvider. So the workers -- which compute every
    embedding this script caches and concatenates -- are on a different
    provider from the parent, and CPU and GPU embeddings differ by up to
    ~1.1e-4.

    That divergence predates any caching work; it became visible only when the
    cache started recording provenance. Probing through an actual worker makes
    the recorded identity match the bytes that get written, and
    `_init_worker` below then asserts the two still agree at runtime, so this
    can never go silently wrong again.
    """
    import embed_cache
    with cf.ProcessPoolExecutor(max_workers=1, initializer=_init_worker) as ex:
        return ex.submit(_report_provenance).result()


def _report_provenance(_=None):
    import embed_cache
    return embed_cache.feature_identity(_worker_feat, ncpu=1)


def _resolve_cache_dir():
    """Point CACHE_DIR at a subdirectory keyed by the WORKERS' feature
    provenance.

    THE BUG THIS FIXES. The cache key used to be
    `str(path).replace("/","__") + ".npy"` -- source path only. No execution
    provider, no ONNX model identity. Every one of the ~7,387 entries in
    `.embed_cache/` was written on the CPU provider, before onnxruntime-gpu
    was installed on this box; a later run on a different provider would
    silently concatenate CPU-computed rows for the cached files with
    differently-computed rows for anything new into ONE array. Neither
    `_valid_cache_array` (rank / dtype / finiteness only) nor
    `verify_against_cache` (which rebuilds FROM the same cache) can detect it.

    Putting the provenance in the PATH makes the mixture structurally
    impossible rather than merely unlikely. Legacy top-level entries are
    reported and ignored, never adopted: their provenance is unknown, and
    guessing it is exactly the failure this is meant to prevent.
    """
    global CACHE_DIR
    import embed_cache
    prov = _worker_provenance()
    CACHE_DIR = embed_cache.scoped_cache_dir_for_provenance(CACHE_BASE, prov)
    print(f"[negatives] embedding cache: {CACHE_DIR} "
          f"(worker provider={prov['identity']['provider']})")
    return CACHE_DIR


_worker_feat = None


def _init_worker():
    """Runs once per worker process (after fork). Builds a fresh
    AudioFeatures with the exact same config as the main process would use
    -- see the Parallelism note in the module docstring."""
    global _worker_feat
    _worker_feat = openwakeword.utils.AudioFeatures(device=_worker_device())
    # Hard-fail rather than mislabel: if this worker's provider is not the one
    # CACHE_DIR was named for, every entry it writes would carry the wrong
    # provenance. Skipped while CACHE_DIR is still the unscoped base (i.e.
    # during the one-worker probe that establishes it in the first place).
    if CACHE_DIR != CACHE_BASE:
        import embed_cache
        want = embed_cache.identity_key(embed_cache.feature_identity(_worker_feat))
        if CACHE_DIR.name != want:
            raise RuntimeError(
                f"worker feature provenance {want} does not match the cache "
                f"directory {CACHE_DIR.name} chosen by the parent -- refusing "
                f"to write mislabelled embeddings "
                f"(provider={_worker_feat.onnx_execution_provider})")


def _process_file(task):
    """Runs in a worker process. Returns (path_str, windows_f16, error)."""
    path_str, stride = task
    try:
        w = windows16(cached_embed(_worker_feat, path_str), stride=stride)
        w16 = w.astype(np.float16) if len(w) else np.empty((0, 16, 96), np.float16)
        return (path_str, w16, None)
    except Exception as e:
        return (path_str, None, repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["fma", "tv_mix"])
    ap.add_argument("--out", default="media_negatives.npy")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--cap", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    _resolve_cache_dir()  # before any fork, so workers inherit CACHE_DIR
    dirs = [str(Path(d)) for d in args.dirs]

    rng = np.random.default_rng(args.seed)
    chunks = []
    files, per_dir_found = gather_files(dirs)
    print(f"[negatives] {len(files)} files from {dirs}: "
          f"{ {d: per_dir_found[d] for d in dirs} } ({N_WORKERS} workers)")

    processed = {d: 0 for d in dirs}
    failed = {d: 0 for d in dirs}
    ok_paths = []
    tasks = [(str(f), args.stride) for f in files]
    done = 0
    with cf.ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker) as ex:
        # .map() yields results in task-submission order (not completion
        # order), which is required for determinism -- see module docstring.
        for path_str, w, err in ex.map(_process_file, tasks, chunksize=4):
            d = str(Path(path_str).parent)
            done += 1
            if err is not None:
                failed[d] += 1
                print(f"[negatives] WARN: failed to process {path_str}: {err}", flush=True)
            else:
                if len(w):
                    chunks.append(w)
                ok_paths.append(path_str)
                processed[d] += 1
            if done % 200 == 0:
                print(f"  {done}/{len(files)}", flush=True)

    for d in dirs:
        note = f", {failed[d]} failed" if failed[d] else ""
        print(f"[negatives] {d}: {processed[d]}/{per_dir_found[d]} processed{note}")

    data = np.concatenate(chunks)
    verify_against_cache(ok_paths, args.stride, data)
    if len(data) > args.cap:
        data = data[rng.choice(len(data), args.cap, replace=False)]
    assert data.ndim == 3 and data.shape[1:] == (16, 96) and data.dtype == np.float16
    assert np.isfinite(data.astype(np.float32)).all()
    atomic_save(args.out, data)
    print(f"[negatives] saved {data.shape} float16 -> {args.out}")


if __name__ == "__main__":
    main()
