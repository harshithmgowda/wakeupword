"""Persisted openWakeWord embeddings for the immutable eval corpora.

WHY THIS IS THE MOST DANGEROUS FILE IN THE REPO, AND HOW IT DEFENDS ITSELF
=========================================================================
Embedding is ~95% of the cost of every eval run, and `eval_media/` and the
23.924 h false-alarm corpus never change, so caching it across invocations is
free money.  It is also the one optimisation that can corrupt every future
result *in a way that looks exactly like a model change*: a stale or
foreign-provenance cache silently substitutes different feature values, the
scores move, and nobody can tell that from a real regression.

So this cache refuses to serve anything it cannot prove is right:

1. **Provenance.**  A `manifest.json` records the identity of whatever computed
   the entries: the onnxruntime execution provider, the SHA-256 of the
   melspectrogram and embedding ONNX files, SHA-256 hashes of the actual
   PYTHON CODE that determines embedding values, the feature dimension and
   dtype, and this module's own format version.  If ANY of those differ from
   the current process, the cache is rejected whole -- not partially reused.
   This is the exact defect in the existing `.embed_cache/` directories used
   by `make_media_negatives.py` / `mine_hard_negatives.py`: their key is
   `str(path).replace("/","__") + ".npy"` -- no provider, no model identity --
   and all 7387 of their entries were written on the CPU provider before
   `onnxruntime_gpu` was installed, so re-running either builder today would
   concatenate CPU rows for cached files with GPU rows for new ones into one
   array.  (CPU and GPU embeddings differ by up to ~1.1e-4; that is not noise
   you want silently mixed into a negative bank.)

   The provider IS part of the identity.  `ncpu` deliberately is NOT: it is
   `embed_clips`' ThreadPool width over independent per-window ORT calls and is
   provably bit-identical at every value (max_abs_diff 0.0 for K in
   1/2/4/12/24 on real clips -- see tests/test_eval_path_equivalence.py).
   Including it would cause spurious full rebuilds for no correctness gain.  It
   is recorded in the manifest for diagnostics only.

   CORRECTED 2026-07-30 -- CODE IDENTITY WAS A CONSTANT.  This identity used
   to include `getattr(openwakeword, "__version__", "unknown")`.  The vendored
   package has no `__version__` attribute at all (verified: `hasattr` is
   False), so that field was **always the string `"unknown"`**, on every
   commit, forever -- it could never detect a change to the code that computes
   embeddings.  Demonstrated exploit: seed a cache on real clips, edit
   `_get_melspectrogram`, re-run `embed_corpus` in a fresh process -> the old
   scheme reported hits for every file, `cache_rejected=None`, and served
   values that differed from what the edited code actually computes by
   7.09e-3 -- silently.  This is exactly the class of change patch 3 (in
   `patches/README.md`) made historically: it shifted embeddings by 3.3% and
   trained wake-word scores by up to 0.11, entirely inside
   `openwakeword/utils.py`, which no field of the old identity could see.
   `feature_identity()` now hashes the CONTENT (not mtime -- see #2 for why
   that distinction matters) of every `.py` file that participates in
   `embed_clips`' call graph: `openwakeword/utils.py` (melspectrogram +
   embedding computation, `_get_melspectrogram`/`_get_embeddings_from_melspec`/
   `_get_melspectrogram_batch`/`_get_embeddings_batch`/`embed_clips` all live
   there) and this repo's own `media_utils.py` (`read_wav16k`'s decode/resample
   is the input to all of it). `openwakeword/data.py` is hashed too, as
   defence in depth, even though tracing the actual call graph shows it is
   NOT currently reached from `embed_clips` (only `compute_features_from_generator`,
   used by the training augment stage, imports from it) -- the cost of one
   more file hash is negligible and it removes the need to remember to add it
   back if that call graph ever changes.  It is deliberately hashed BY PATH
   (`Path(openwakeword.utils.__file__).parent / "data.py"`), never by
   `import openwakeword.data`: that import hard-fails in `.venv-onnx2tf`
   (`ModuleNotFoundError: No module named 'pronouncing'` -- a dependency only
   installed in `.venv`), which would break every eval script that builds a
   feature identity in the CPU-only venv.  See `code_sha256` in
   `feature_identity()`.  `tests/test_embed_cache.py` proves this
   non-vacuously: it mutates a COPY of `openwakeword/utils.py`'s embedding
   code, hashes the mutated copy with the same `_sha256_file` production uses,
   and shows `embed_corpus` rejects and fully recomputes rather than serving
   the field-swap fantasy the old `openwakeword_version` check could never
   actually exercise.

2. **Per-file staleness for corpus audio -- a HEURISTIC, not a proof.**  The
   index stores each source file's size and mtime_ns.  A file whose size or
   mtime moved is a MISS, recomputed.  A file absent from the index is a MISS.

   THE GAP, STATED HONESTLY: size+mtime is not content.  A tool that restores
   both after changing the bytes -- `cp --preserve=timestamps`, `rsync -t`,
   `tar -p`, or a backup restore -- serves the STALE entry, undetected
   (verified: 6 real clips modified this way still register as 6 hits / 0
   misses, and the served array differs from a from-scratch recompute by
   max_abs_diff 55.67). This claim used to read "there is no path by which an
   entry computed from different bytes can be served" -- that was false, and
   the sentence above is the correction.

   This is a real, ACCEPTED trade-off, not an oversight. Full content hashing
   was measured on this corpus: SHA-256 over all 2872 files / 2.76 GB of
   `eval_media/fa_baseline_24h` (the corpus this cache exists for) took
   **2.81 s (982.7 MB/s), page-cache warm** -- and that cost would be paid on
   EVERY lookup, not just a cold build, because there is no way to know
   content changed without reading it. Against the embedding phase's own
   measured warm-cache cost of 6 s (benchmarks on the experiments branch),
   always-on content
   hashing would add roughly **47% wall clock to the exact number this cache
   exists to produce**, to close a gap whose real-world triggers (tools that
   deliberately preserve timestamps across a content change) do not occur
   anywhere in this repo's normal workflow -- `git`, every builder script
   here, and a plain edit all change mtime. Code and model files (see #1) ARE
   fully content-hashed despite the same kind of cost existing in principle,
   because there the files are small (hashing them is unmeasurable) AND the
   mtime-preserving-edit scenario (a vendored patch applied, a `git checkout`
   of an old commit, a backup restore of `openwakeword/`) is exactly the
   routine case that class of file must defend against -- the trade-off is
   evaluated per file class, not applied uniformly.

   A file absent from the index, or whose size/mtime moved, is still always a
   MISS -- recomputed, never served partially or approximately. The gap above
   is specifically: same path, same size, same mtime, different bytes.

3. **Shape sanity.**  Every served array is checked for rank, feature width,
   dtype and finiteness before it leaves this module.

4. **Atomic writes.**  The store is written to sibling temp files, fsync'd, and
   `os.replace`d into place, so a kill mid-write leaves the previous good store
   or nothing -- never a half-written one.  (`make_media_negatives.py` learned
   this the hard way; see its `atomic_save`.)

5. **An escape hatch that is honest.**  `use_cache=False` bypasses everything.
   `tests/test_embed_cache.py` runs both paths over real audio and asserts
   `np.array_equal`.

STORAGE
    <cache_dir>/emb.npy        one concatenated (total_frames, 96) float32 array
    <cache_dir>/index.npz      paths, offsets, lengths, sizes, mtimes
    <cache_dir>/manifest.json  the provenance above
Measured at ~16.8 MB per audio-hour, i.e. ~0.40 GB for the 23.924 h corpus.
Frame embeddings are cached, NOT stride-1 windows: windows are 16x larger and
`media_utils.windows16` rebuilds them for ~0.036 s per 10 minutes of audio.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

FORMAT_VERSION = 1
FEATURE_DIM = 96
DTYPE = np.float32


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
def _sha256_file(path, _cache={}):
    path = str(path)
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    if key not in _cache:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _cache[key] = h.hexdigest()
    return _cache[key]


def _code_identity():
    """SHA-256 of the CONTENT of every `.py` file that determines an embedding
    value -- see the module docstring's item 1 for why this replaced
    `openwakeword.__version__` (which does not exist, so it was a constant).

    Resolved BY PATH, not by importing each module:
      * `openwakeword.utils` is already imported by any caller that built a
        `feat` -- importing it again here is free and safe.
      * `openwakeword.data` is deliberately NEVER imported here. It sits next
        to utils.py in the same package directory, so its path is derived
        from `openwakeword.utils.__file__` instead. `import openwakeword.data`
        raises `ModuleNotFoundError: No module named 'pronouncing'` in
        `.venv-onnx2tf` (that dependency is only installed in `.venv`), which
        would break every eval script -- all of them run in `.venv-onnx2tf`.
      * `media_utils` (this repo's own decode/resample step, `read_wav16k`) is
        this module's sibling, imported at module scope below.
    """
    import openwakeword.utils as _oww_utils

    import media_utils as _media_utils

    oww_dir = Path(_oww_utils.__file__).resolve().parent
    return {
        "openwakeword_utils": _sha256_file(oww_dir / "utils.py"),
        # Not on the current embed_clips() call graph (only
        # compute_features_from_generator, used by the training augment
        # stage, imports from it) -- hashed anyway as defence in depth; see
        # the module docstring. Hashed by path so this never imports it.
        "openwakeword_data": _sha256_file(oww_dir / "data.py"),
        "media_utils": _sha256_file(Path(_media_utils.__file__).resolve()),
    }


def feature_identity(feat, ncpu=None):
    """The provenance dict for an `openwakeword.utils.AudioFeatures` instance.

    Everything in `identity` participates in cache validity. `ncpu` and
    `built_at` are recorded outside it, for diagnostics only -- see the module
    docstring for why ncpu must not invalidate.
    """

    def _model_path(session):
        # onnxruntime InferenceSession keeps the path it was constructed with.
        p = getattr(session, "_model_path", None)
        return str(p) if p else None

    mel_p = _model_path(getattr(feat, "melspec_model", None))
    emb_p = _model_path(getattr(feat, "embedding_model", None))
    identity = {
        "format_version": FORMAT_VERSION,
        "provider": str(feat.onnx_execution_provider),
        "feature_dim": FEATURE_DIM,
        "dtype": np.dtype(DTYPE).str,
        "code_sha256": _code_identity(),
        "melspec_model_sha256": _sha256_file(mel_p) if mel_p else "unknown",
        "embedding_model_sha256": _sha256_file(emb_p) if emb_p else "unknown",
    }
    return {"identity": identity, "ncpu": ncpu,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def identity_key(provenance):
    """Short stable hash of the identity dict -- usable as a directory name."""
    blob = json.dumps(provenance["identity"], sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def scoped_cache_dir(base, feat, ncpu=None, log=print):
    """`base/<identity hash>` for the process holding `feat`.

    CAUTION: if the embeddings are actually computed in FORKED WORKERS, do not
    use this -- probe a worker and use scoped_cache_dir_for_provenance instead.
    A CUDA context cannot be inherited across fork, so a parent on
    CUDAExecutionProvider can have workers that silently fall back to
    CPUExecutionProvider; naming the directory after the parent would then
    mislabel every entry. See make_media_negatives._worker_provenance.
    """
    return scoped_cache_dir_for_provenance(base, feature_identity(feat, ncpu=ncpu),
                                           log=log)


def scoped_cache_dir_for_provenance(base, prov, log=print):
    """`base/<identity hash>`, created, with a manifest.json written into it.

    For the flat per-file caches in make_media_negatives.py /
    mine_hard_negatives.py, which key only on the source path. Putting the
    identity in the PATH makes it structurally impossible for entries computed
    under different providers or different ONNX models to land in one
    directory and be concatenated into a single negative bank.

    Entries sitting directly in `base` predate this scoping and have UNKNOWN
    provenance, so they are ignored and reported rather than adopted -- see the
    module docstring for why silently reusing them is the exact hazard here.
    """
    base = Path(base)
    d = base / identity_key(prov)
    d.mkdir(parents=True, exist_ok=True)
    man = d / "manifest.json"
    if man.is_file():
        try:
            old = json.loads(man.read_text())
        except Exception:
            old = {}
        if old.get("identity") != prov["identity"]:
            raise SystemExit(
                f"[embed_cache] {man} does not match its own directory hash. "
                "Refusing to use it; delete the directory and rebuild.")
    else:
        man.write_text(json.dumps(prov, indent=2, sort_keys=True))

    legacy = sorted(base.glob("*.npy"))
    if legacy:
        log(f"[embed_cache] NOTE: {len(legacy)} legacy entries sit directly in "
            f"{base}/ with no recorded provenance (no provider, no model "
            f"identity). They are IGNORED -- mixing embeddings from different "
            f"execution providers into one negative bank changes values by up "
            f"to ~1.1e-4. Delete them once you no longer need them.")
    return d


def _valid(emb):
    return (isinstance(emb, np.ndarray) and emb.ndim == 2
            and emb.shape[1] == FEATURE_DIM
            and emb.dtype == np.dtype(DTYPE) and np.isfinite(emb).all())


def _atomic_replace(tmp, final):
    with open(tmp, "rb+") as fh:
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, final)


class EmbeddingCache:
    """Read/write store for per-file frame embeddings. Single writer.

    Concurrency: reads are safe from any number of processes (the store is
    replaced atomically). Two processes writing the SAME cache_dir at the same
    time is not supported -- the last writer wins wholesale, which is safe
    (the store is always internally consistent) but wasteful. The eval scripts
    build their caches serially.
    """

    def __init__(self, cache_dir, provenance):
        self.dir = Path(cache_dir)
        self.provenance = provenance
        self._emb = None
        self._index = {}          # str path -> (offset, length, size, mtime_ns)
        self.reject_reason = None

    # ---------------------------------------------------------------- load
    def load(self):
        """Attach to an existing store. Returns True only if it is usable AND
        its provenance matches. Sets self.reject_reason otherwise."""
        self._emb, self._index = None, {}
        man_p, idx_p, emb_p = (self.dir / "manifest.json",
                               self.dir / "index.npz", self.dir / "emb.npy")
        if not (man_p.is_file() and idx_p.is_file() and emb_p.is_file()):
            self.reject_reason = "no cache present"
            return False
        try:
            man = json.loads(man_p.read_text())
        except Exception as exc:
            self.reject_reason = f"unreadable manifest ({exc!r})"
            return False
        if man.get("identity") != self.provenance["identity"]:
            want, got = self.provenance["identity"], man.get("identity") or {}
            diff = {k: (got.get(k), want[k]) for k in want if got.get(k) != want[k]}
            self.reject_reason = (
                "provenance mismatch -- REFUSING to serve embeddings computed "
                f"by a different configuration. cached vs wanted: {diff}")
            return False
        try:
            idx = np.load(idx_p, allow_pickle=False)
            emb = np.load(emb_p, mmap_mode="r")
        except Exception as exc:
            self.reject_reason = f"unreadable store ({exc!r})"
            return False
        if emb.ndim != 2 or emb.shape[1] != FEATURE_DIM or emb.dtype != np.dtype(DTYPE):
            self.reject_reason = f"bad store shape/dtype {emb.shape} {emb.dtype}"
            return False
        paths = [str(p) for p in idx["paths"]]
        if not (len(paths) == len(idx["offsets"]) == len(idx["lengths"])
                == len(idx["sizes"]) == len(idx["mtimes"])):
            self.reject_reason = "index arrays disagree in length"
            return False
        if len(paths) and int(idx["offsets"][-1] + idx["lengths"][-1]) > emb.shape[0]:
            self.reject_reason = "index runs past the end of the store"
            return False
        self._emb = emb
        self._index = {p: (int(o), int(n), int(s), int(m)) for p, o, n, s, m
                       in zip(paths, idx["offsets"], idx["lengths"],
                              idx["sizes"], idx["mtimes"])}
        return True

    # -------------------------------------------------------------- lookup
    def lookup(self, path):
        """Cached embedding for `path`, or None on any miss.

        A miss is: not indexed, source file gone, or source size/mtime moved.
        Never a partial or approximate hit.
        """
        if self._emb is None:
            return None
        key = str(path)
        rec = self._index.get(key)
        if rec is None:
            return None
        off, n, size, mtime = rec
        try:
            st = os.stat(key)
        except OSError:
            return None
        if st.st_size != size or st.st_mtime_ns != mtime:
            return None
        emb = np.array(self._emb[off:off + n], dtype=DTYPE)
        if emb.shape[0] != n or not _valid(emb):
            return None
        return emb

    # --------------------------------------------------------------- write
    def write(self, entries):
        """Replace the store with `entries`, a list of (path, embedding).

        Written to temp files and os.replace'd, so a kill mid-write leaves the
        previous store intact.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        entries = [(str(p), e) for p, e in entries if e is not None and len(e)]
        for p, e in entries:
            if not _valid(e):
                raise ValueError(f"refusing to cache an invalid array for {p}: "
                                 f"shape={getattr(e, 'shape', None)} "
                                 f"dtype={getattr(e, 'dtype', None)}")
        total = sum(e.shape[0] for _, e in entries)
        tmp_emb = self.dir / "emb.tmp.npy"
        fp = open_memmap(tmp_emb, mode="w+", dtype=DTYPE,
                         shape=(total, FEATURE_DIM))
        paths, offs, lens, sizes, mtimes = [], [], [], [], []
        row = 0
        for p, e in entries:
            fp[row:row + e.shape[0]] = e
            st = os.stat(p)
            paths.append(p)
            offs.append(row)
            lens.append(e.shape[0])
            sizes.append(st.st_size)
            mtimes.append(st.st_mtime_ns)
            row += e.shape[0]
        fp.flush()
        del fp

        tmp_idx = self.dir / "index.tmp.npz"  # np.savez appends .npz otherwise
        np.savez(tmp_idx, paths=np.array(paths, dtype=object).astype("U"),
                 offsets=np.array(offs, dtype=np.int64),
                 lengths=np.array(lens, dtype=np.int64),
                 sizes=np.array(sizes, dtype=np.int64),
                 mtimes=np.array(mtimes, dtype=np.int64))
        tmp_man = self.dir / "manifest.tmp.json"
        man = dict(self.provenance)
        man["n_files"] = len(paths)
        man["n_rows"] = int(total)
        tmp_man.write_text(json.dumps(man, indent=2, sort_keys=True))

        # index + manifest first: an emb.npy newer than its index is the only
        # ordering that load() cannot detect, so make it impossible.
        _atomic_replace(tmp_idx, self.dir / "index.npz")
        _atomic_replace(tmp_man, self.dir / "manifest.json")
        _atomic_replace(tmp_emb, self.dir / "emb.npy")
        self.load()
        return total


def embed_corpus(paths, compute, feat, cache_dir, ncpu=None, use_cache=True,
                 log=print, log_every=200):
    """Embed every path, using and refreshing a provenance-checked cache.

    Args:
        paths:     iterable of audio file paths, in the order the caller wants
                   the results back
        compute:   callable(path) -> (frames, 96) float32 array, or raising.
                   This is the ONLY place embeddings are produced; the cache
                   never transforms a value it stores or serves.
        feat:      the AudioFeatures instance `compute` uses (for provenance)
        cache_dir: where to keep the store
        use_cache: False bypasses the cache entirely -- the reference path

    Returns:
        (results, stats) where results is a list of (path, embedding_or_None)
        in input order (None = compute raised, with the exception in stats),
        and stats is a dict of counters including per-file failures.
    """
    paths = [str(p) for p in paths]
    stats = {"n_paths": len(paths), "hits": 0, "misses": 0, "errors": [],
             "cache_used": bool(use_cache), "cache_rejected": None,
             "rewrote_cache": False}

    cache = None
    if use_cache:
        cache = EmbeddingCache(cache_dir, feature_identity(feat, ncpu=ncpu))
        if not cache.load():
            stats["cache_rejected"] = cache.reject_reason
            log(f"[embed_cache] not serving from cache: {cache.reject_reason}")

    t0 = time.time()
    results = []
    for i, p in enumerate(paths):
        emb = cache.lookup(p) if cache is not None else None
        if emb is not None:
            stats["hits"] += 1
        else:
            try:
                emb = np.asarray(compute(p), dtype=DTYPE)
            except Exception as exc:
                stats["errors"].append((os.path.basename(p), repr(exc)))
                results.append((p, None))
                continue
            stats["misses"] += 1
        results.append((p, emb))
        if log_every and i % log_every == 0:
            log(f"[embed_cache] {i}/{len(paths)} files "
                f"({stats['hits']} cached, {stats['misses']} computed), "
                f"{time.time()-t0:.0f}s elapsed")

    if cache is not None and stats["misses"]:
        good = [(p, e) for p, e in results if e is not None]
        rows = cache.write(good)
        stats["rewrote_cache"] = True
        log(f"[embed_cache] wrote {len(good)} files / {rows} rows to "
            f"{cache_dir}")
    stats["seconds"] = round(time.time() - t0, 3)
    return results, stats
