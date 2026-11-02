#!/usr/bin/env python
"""Windows of media audio that v5 scores high -> hard-negative features.

    ./.venv/bin/python tools/data/mine_hard_negatives.py --out mined_negatives.npy

Performance note (deviation from the task-8 brief, see task-8-report.md):
make_media_negatives.py embeds the same fma/ and tv_mix/ files this script
does (this script additionally covers audioset_16k/). Embedding is the
dominant cost, so each file's frame-level embedding is computed once and
cached to disk under .embed_cache/ (float32, keyed by source path), shared
with make_media_negatives.py. See that script's docstring for the same
note; the cache format is identical so either script can run first.

Correctness fix (deviation, verified by direct repro -- see
task-8-report.md): the brief's scoring code batches up to 4096 windows per
sess.run call, but my_custom_model_v5/ouh_ke-i_nahbu.onnx has a *fixed*
batch dimension of 1 baked into its graph (sess.get_inputs()[0].shape ==
[1, 16, 96]); onnxruntime raises InvalidArgument for any other batch size,
so the brief's literal code cannot run against this model. This mirrors a
quirk already documented in compare_models.py's `_can_batch` (that one for
the onnx2tf tflite export) -- same fix here: probe once whether the loaded
model accepts a batch, and fall back to scoring one window at a time if
not. Confirmed fast either way: benchmarked at ~89k single-window calls/sec
on this model, so batch=1 scoring of the full ~1.4M-window corpus costs
well under a minute.

Parallelism note (deviation, requested mid-run -- see task-8-report.md and
make_media_negatives.py's matching docstring section): embedding (the
dominant cost, same onnxruntime CPU path as make_media_negatives.py -- no
CUDAExecutionProvider is installed) is spread across a ProcessPoolExecutor,
each worker building its own AudioFeatures with the unchanged
device="gpu" if torch.cuda.is_available() else "cpu" expression. Only
embedding is parallelized: the v5 onnx scoring session is not picklable
and is cheap (see Correctness fix note above), so it stays single-instance
in the parent, fed by workers' (path, stride-1 windows) results consumed
via .map() in submission order -- preserving the per-file 8-frame local-
maximum rule and overall determinism given --seed exactly as in the
sequential version. Numeric equivalence of parallel vs. sequential
embedding was verified once in make_media_negatives.py (bit-identical,
np.array_equal) since both scripts share the same cached_embed/AudioFeatures
construction; not re-verified independently here.

Robustness note: see make_media_negatives.py's docstring -- a missing
--dirs entry is a hard error, a single file that fails to decode/embed is
logged and skipped (not silently dropped), and per-directory
found/scanned/kept counts are always printed.
"""
import sys
import argparse
import concurrent.futures as cf
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
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
    """Write `arr` to `path` atomically -- see make_media_negatives.py's
    identical helper/docstring for why (the media_negatives.npy truncation
    this task inherited is exactly what this prevents)."""
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
    """Cheap sanity check on a loaded cache entry -- see
    make_media_negatives.py's identical helper/docstring for what this
    does and doesn't guard against."""
    return (emb.ndim == 2 and emb.shape[0] > 0 and emb.shape[1] == feat_dim
            and emb.dtype == dtype and np.isfinite(emb).all())


def cached_embed(feat, path):
    """Frame-level embedding for `path`, cached to disk across script runs
    and across worker processes (self-healing against a truncated or
    invalid cache file -- see make_media_negatives.py for the identical
    helper/format).

    Fix round 1 (see task-8-report.md): the cache write used to be a bare
    `np.save(key, emb)`; that non-atomic write, not "IPC corruption" as
    this task's report originally (wrongly) concluded, is the mechanically
    plausible cause of a corrupted row observed during development -- a
    concurrently-running process (this box was not actually idle) writing
    the same cache entry while another process's `np.load(key)` read it.
    Fixed by routing the write through `atomic_save()` and validating what
    comes back on read via `_valid_cache_array`.
    """
    key = CACHE_DIR / (str(path).replace("/", "__") + ".npy")
    if key.exists():
        try:
            emb = np.load(key)
            if _valid_cache_array(emb):
                return emb
        except Exception:
            pass
    # ncpu=1 on purpose: runs inside N_WORKERS worker processes, so the
    # media_utils.EMBED_NCPU default (12) would oversubscribe the box. Pure
    # ThreadPool width -- bit-identical at any value.
    emb = embed(feat, read_wav16k(str(path)), ncpu=1).astype(np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_save(key, emb)
    return emb


def verify_against_cache(ok_paths, sess, in_name, batch, score_min, kept, kept_scores):
    """Rebuild `kept`/`kept_scores` single-threaded, straight from the (now
    fully-populated) on-disk embed cache, re-scoring with the same onnx
    session, and assert it matches the ProcessPoolExecutor-collected result
    exactly.

    Fix round 1 correction (see task-8-report.md and
    make_media_negatives.py's identical function): a rare row-level
    mismatch was observed once during development between the parallel
    result and this rebuild. It was originally (wrongly) attributed to
    multiprocessing/IPC corruption; a code-review pass traced the actual
    mechanism to a non-atomic cache write, now fixed in `cached_embed`
    above. Caveat this function does *not* fully cover: both this rebuild
    and the parallel path read the same on-disk cache, so a persistently
    wrong-but-valid-looking cache entry could in principle be agreed on by
    both and slip through -- the atomic write + `_valid_cache_array` are
    the actual fix for that; this remains a useful check on the
    aggregation/ordering logic itself. Only re-embedding is skipped here
    (cache is warm); scoring is genuinely redone, which is cheap (~89k
    single-window calls/sec, benchmarked -- see the Correctness fix note
    above).
    """
    r_kept, r_scores = [], []
    for path_str in ok_paths:
        key = CACHE_DIR / (str(path_str).replace("/", "__") + ".npy")
        emb = np.load(key)
        w = windows16(emb, stride=1)
        if not len(w):
            continue
        scores = score_windows(sess, in_name, w, batch)
        hot = np.flatnonzero(scores >= score_min)
        last = -100
        for j in hot:
            if j - last >= 8:
                r_kept.append(w[j].astype(np.float16))
                r_scores.append(float(scores[j]))
                last = j
    truth = np.stack(r_kept) if r_kept else np.empty((0, 16, 96), np.float16)
    data = np.stack(kept) if kept else np.empty((0, 16, 96), np.float16)
    ok = (truth.shape == data.shape and np.array_equal(truth, data)
          and r_scores == kept_scores)
    if not ok:
        n_diff = ("shape mismatch" if truth.shape != data.shape
                  else f"{int((truth != data).any(axis=(1, 2)).sum())} differing rows")
        raise SystemExit(
            f"[mine] FATAL: parallel-collected result does not match a "
            f"single-threaded rebuild from the same cache ({n_diff}); "
            f"refusing to save a possibly-corrupted array.")
    print(f"[mine] verified: parallel result matches single-threaded cache "
          f"rebuild ({len(truth)} kept windows)")


def gather_files(dirs):
    per_dir_found = {}
    files = []
    for d in dirs:
        dp = Path(d)
        if not dp.is_dir():
            raise SystemExit(f"[mine] ERROR: directory not found: {d}")
        found = sorted(dp.glob("*.wav"))
        if not found:
            print(f"[mine] WARNING: no .wav files in {d}")
        per_dir_found[d] = len(found)
        files.extend(found)
    return files, per_dir_found


def probe_batch_size(sess, in_name, max_batch=4096):
    """Does this onnx graph accept a batch dim other than 1? See the
    Correctness fix note in the module docstring. Probed once (not per
    file) to keep the per-file loop simple."""
    try:
        sess.run(None, {in_name: np.zeros((2, 16, 96), dtype=np.float32)})
        return max_batch
    except Exception:
        return 1


def score_windows(sess, in_name, w, batch):
    return np.concatenate([
        sess.run(None, {in_name: w[j:j + batch].astype(np.float32)})[0].reshape(-1)
        for j in range(0, len(w), batch)])


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
    print(f"[mine] embedding cache: {CACHE_DIR} "
          f"(worker provider={prov['identity']['provider']})")
    return CACHE_DIR


_worker_feat = None


def _init_worker():
    """Runs once per worker process (after fork) -- see the Parallelism
    note in the module docstring."""
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


def _process_file(path_str):
    """Runs in a worker process. Returns (path_str, stride1_windows_f32,
    error). Windows stay float32 (not downcast) so the parent's onnx
    scoring sees the same precision as the sequential version."""
    try:
        w = windows16(cached_embed(_worker_feat, path_str), stride=1)
        return (path_str, w, None)
    except Exception as e:
        return (path_str, None, repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="my_custom_model_v5/ouh_ke-i_nahbu.onnx")
    ap.add_argument("--dirs", nargs="+", default=["fma", "tv_mix", "audioset_16k"])
    ap.add_argument("--out", default="mined_negatives.npy")
    ap.add_argument("--score-min", type=float, default=0.3)
    ap.add_argument("--cap", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()
    _resolve_cache_dir()  # before any fork, so workers inherit CACHE_DIR
    dirs = [str(Path(d)) for d in args.dirs]

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    batch = probe_batch_size(sess, in_name)

    kept, kept_scores, scanned = [], [], 0
    files, per_dir_found = gather_files(dirs)
    print(f"[mine] scanning {len(files)} files from {dirs}: "
          f"{ {d: per_dir_found[d] for d in dirs} } (batch={batch}, {N_WORKERS} workers)")

    processed = {d: 0 for d in dirs}
    failed = {d: 0 for d in dirs}
    ok_paths = []
    tasks = [str(f) for f in files]
    done = 0
    with cf.ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker) as ex:
        # .map() yields results in task-submission order (not completion
        # order) -- required so the per-file local-maximum rule and the
        # final ordering stay identical to the sequential version.
        for path_str, w, err in ex.map(_process_file, tasks, chunksize=4):
            d = str(Path(path_str).parent)
            done += 1
            if err is not None:
                failed[d] += 1
                print(f"[mine] WARN: failed to process {path_str}: {err}", flush=True)
                continue
            processed[d] += 1
            ok_paths.append(path_str)
            if not len(w):
                continue
            scanned += len(w)
            scores = score_windows(sess, in_name, w, batch)
            hot = np.flatnonzero(scores >= args.score_min)
            # keep local maxima at least 8 frames apart, not every
            # overlapping window -- `last` is reset per file, so this
            # never merges across file boundaries.
            last = -100
            for j in hot:
                if j - last >= 8:
                    kept.append(w[j].astype(np.float16))
                    kept_scores.append(float(scores[j]))
                    last = j
            if done % 200 == 0:
                print(f"  {done}/{len(files)}  kept={len(kept)}", flush=True)

    for d in dirs:
        note = f", {failed[d]} failed" if failed[d] else ""
        print(f"[mine] {d}: {processed[d]}/{per_dir_found[d]} processed{note}")

    print(f"[mine] {len(kept)} hot windows out of {scanned}")
    if kept_scores:
        s = np.array(kept_scores)
        # >=score-min is len(kept) by construction (that's the selection
        # threshold) -- printed anyway so the report doesn't need to infer it.
        print(f"[mine] score distribution of kept (local-max) windows: "
              f">={args.score_min}: {len(s)}  "
              f">=0.5: {int((s >= 0.5).sum())}  "
              f">=0.7: {int((s >= 0.7).sum())}  "
              f">=0.9: {int((s >= 0.9).sum())}  "
              f"(min={s.min():.3f} max={s.max():.3f} median={np.median(s):.3f})")
    if len(kept) < 500:
        print("WARNING: <500 mined windows; re-run with --score-min 0.2")
    verify_against_cache(ok_paths, sess, in_name, batch, args.score_min, kept, kept_scores)
    data = np.stack(kept) if kept else np.empty((0, 16, 96), np.float16)
    if len(data) > args.cap:
        rng = np.random.default_rng(args.seed)
        data = data[rng.choice(len(data), args.cap, replace=False)]
    atomic_save(args.out, data)
    print(f"[mine] saved {data.shape} float16 -> {args.out}")


if __name__ == "__main__":
    main()
