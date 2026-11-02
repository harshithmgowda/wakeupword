#!/usr/bin/env python
"""The embedding cache must be exact, and must REFUSE anything it can't prove.

A silently-stale feature cache corrupts every future eval in a way that looks
like a model change, so this test spends most of its effort on the refusal
paths, not the happy path.

  1. cached == uncached, np.array_equal, on real audio
  2. a touched source file is a MISS (mtime), a rewritten one is a MISS (size)
  2b. DOCUMENTED GAP: a source file whose original size+mtime is restored
      after its DATA changes serves a STALE hit (size+mtime is a heuristic,
      not a content hash -- see embed_cache.py's docstring item 2 for the
      measured cost that keeps it that way for corpus audio)
  3. a provenance mismatch rejects the WHOLE cache, not just the odd entry --
     checked for every field of the identity, including the provider and each
     of the three code_sha256 entries
  3b. the code_sha256 identity reacts to an ACTUAL mutation of a COPY of
      openwakeword/utils.py's embedding code (not a hand-typed field --
      openwakeword has no __version__, which made the field this replaces
      permanently vacuous), and embed_corpus() rejects and recomputes
      everything under it
  4. ncpu is NOT part of the identity (it is provably bit-identical, so making
     it invalidate would cause spurious rebuilds)
  5. a truncated / corrupt / half-written store is rejected, not served
  6. use_cache=False really does bypass everything

Run:  ./.venv/bin/python tests/test_embed_cache.py
"""
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from repo_paths import DATA_ROOT  # noqa: E402
os.chdir(ROOT)

import embed_cache as EC  # noqa: E402
from media_utils import EMBED_NCPU, embed, read_wav16k  # noqa: E402
import openwakeword.utils as oww_utils  # noqa: E402

CLIP_DIRS = [ROOT / "eval_media" / "music",
             DATA_ROOT / "eval"]


def pick(n=6):
    for d in CLIP_DIRS:
        if d.is_dir():
            f = sorted(p for p in d.iterdir() if p.suffix.lower() == ".wav")[:n]
            if f:
                return f
    raise SystemExit("no real clip corpus found")


src = pick()
tmp = Path(tempfile.mkdtemp(prefix="oww_embcache_"))
work = tmp / "audio"
work.mkdir()
# Copy the clips so the staleness tests can mutate them without touching the
# real corpora.
files = []
for p in src:
    q = work / p.name
    shutil.copy2(p, q)
    files.append(q)
print(f"{len(files)} real clips copied to a scratch corpus")

feat = oww_utils.AudioFeatures(device="cpu")
calls = {"n": 0}


def compute(path):
    calls["n"] += 1
    return embed(feat, read_wav16k(path), ncpu=EMBED_NCPU)


cdir = tmp / "cache"
try:
    # ---------------------------------------------------------------- 1 + 6
    calls["n"] = 0
    ref, st_ref = EC.embed_corpus(files, compute, feat, cdir,
                                  ncpu=EMBED_NCPU, use_cache=False,
                                  log=lambda *a: None)
    assert calls["n"] == len(files) and st_ref["hits"] == 0, st_ref
    assert not (cdir / "emb.npy").exists(), "use_cache=False wrote a cache"

    calls["n"] = 0
    cold, st_cold = EC.embed_corpus(files, compute, feat, cdir,
                                    ncpu=EMBED_NCPU, log=lambda *a: None)
    assert st_cold["misses"] == len(files) and st_cold["hits"] == 0, st_cold
    assert st_cold["rewrote_cache"] and (cdir / "emb.npy").exists()

    calls["n"] = 0
    warm, st_warm = EC.embed_corpus(files, compute, feat, cdir,
                                    ncpu=EMBED_NCPU, log=lambda *a: None)
    assert st_warm["hits"] == len(files) and st_warm["misses"] == 0, st_warm
    assert calls["n"] == 0, "warm run recomputed something"

    for (p0, a), (p1, b), (p2, c) in zip(ref, cold, warm):
        assert p0 == p1 == p2
        assert a.shape == b.shape == c.shape, (p0, a.shape, b.shape, c.shape)
        assert a.dtype == c.dtype == np.float32
        assert np.array_equal(a, b), (p0, float(np.abs(a - b).max()))
        assert np.array_equal(a, c), (
            f"CACHE IS NOT EXACT for {p0}: max_abs_diff="
            f"{float(np.abs(a - c).max())}")
    rows = sum(a.shape[0] for _, a in ref)
    print(f"PASS cached == uncached, np.array_equal on {len(files)} real "
          f"clips / {rows} frames; use_cache=False bypasses and writes nothing")

    # -------------------------------------------------------------------- 2
    os.utime(files[0], ns=(time.time_ns(), time.time_ns()))
    calls["n"] = 0
    _, st = EC.embed_corpus(files, compute, feat, cdir, ncpu=EMBED_NCPU,
                            log=lambda *a: None)
    assert st["misses"] == 1 and st["hits"] == len(files) - 1, st
    assert calls["n"] == 1

    data = files[1].read_bytes()
    files[1].write_bytes(data + data[-4000:])          # size changes
    calls["n"] = 0
    _, st = EC.embed_corpus(files, compute, feat, cdir, ncpu=EMBED_NCPU,
                            log=lambda *a: None)
    assert st["misses"] == 1 and calls["n"] == 1, st
    print("PASS a touched (mtime) or rewritten (size) source file is a MISS")

    prov = EC.feature_identity(feat, ncpu=EMBED_NCPU)

    # -------------------------------------------------------------------- 2b
    # DEFECT-B, DOCUMENTED (not "fixed" -- see embed_cache.py's module
    # docstring item 2 for the measured content-hashing cost that is why
    # per-file staleness for CORPUS AUDIO stays a size+mtime HEURISTIC, not a
    # content hash). This test exists so the corrected docstring's "known
    # gap" claim is verified, not asserted on faith: restoring a source
    # file's original size AND mtime after changing its DATA serves the
    # STALE entry. Isolated to files[2], untouched by the mtime/size test
    # above, and restored before any later section depends on it.
    victim = files[2]
    victim_st = os.stat(victim)
    orig_bytes = victim.read_bytes()
    orig_audio = read_wav16k(victim)

    # Flip every byte of the PCM sample data only (never the RIFF/fmt header)
    # so the file stays a valid, SAME-LENGTH wav with genuinely different
    # audio -- not padding, like the size-change case above.
    data_off = orig_bytes.index(b"data") + 8
    mutated = bytearray(orig_bytes)
    for i in range(data_off, len(mutated)):
        mutated[i] ^= 0xFF
    mutated_bytes = bytes(mutated)
    assert len(mutated_bytes) == len(orig_bytes), "mutation changed the file size"

    victim.write_bytes(mutated_bytes)
    os.utime(victim, ns=(victim_st.st_atime_ns, victim_st.st_mtime_ns))
    st_after = os.stat(victim)
    assert st_after.st_size == victim_st.st_size, "size did not survive restoration"
    assert st_after.st_mtime_ns == victim_st.st_mtime_ns, "mtime did not survive restoration"

    mutated_audio = read_wav16k(victim)
    assert mutated_audio.shape == orig_audio.shape
    assert not np.array_equal(mutated_audio, orig_audio), (
        "byte-flipping the sample data did not change the decoded audio -- "
        "fix this test's mutation, it isn't testing anything")

    calls["n"] = 0
    _, st = EC.embed_corpus([victim], compute, feat, cdir, ncpu=EMBED_NCPU,
                            log=lambda *a: None)
    served = EC.EmbeddingCache(cdir, prov)
    assert served.load()
    served_emb = served.lookup(str(victim))
    truth_emb = compute(str(victim))     # what the file's CURRENT bytes decode to

    # Restore the file (bytes + mtime) before any later section touches it.
    victim.write_bytes(orig_bytes)
    os.utime(victim, ns=(victim_st.st_atime_ns, victim_st.st_mtime_ns))

    assert st["hits"] == 1 and st["misses"] == 0, (
        "size+mtime no longer has this gap -- if a real fix landed (e.g. "
        "content hashing), correct embed_cache.py's docstring instead of "
        "deleting this test")
    assert served_emb is not None and served_emb.shape == truth_emb.shape
    diff = float(np.abs(served_emb - truth_emb).max())
    assert diff > 0, (
        "served embedding matched the mutated content's truth -- the gap "
        "this test documents has closed; update the docstring, don't just "
        "delete this test")
    print(f"PASS (DOCUMENTED GAP, NOT A REGRESSION) restoring a source "
          f"file's original size+mtime after changing its DATA serves the "
          f"STALE entry -- served vs current-content max_abs_diff={diff:.4g}. "
          f"This is embed_cache.py's module docstring item 2's known, "
          f"measured, accepted trade-off, not a bug.")

    # -------------------------------------------------------------------- 3
    for field, bogus in (("provider", "CUDAExecutionProvider"),
                         ("embedding_model_sha256", "0" * 64),
                         ("melspec_model_sha256", "0" * 64),
                         ("feature_dim", 128),
                         ("format_version", 999),
                         ("dtype", "<f8")):
        alt = json.loads(json.dumps(prov))
        alt["identity"][field] = bogus
        c = EC.EmbeddingCache(cdir, alt)
        assert not c.load(), f"cache accepted a different {field}!"
        assert "provenance mismatch" in (c.reject_reason or ""), c.reject_reason
        assert c.lookup(str(files[0])) is None
    for subfield in ("openwakeword_utils", "openwakeword_data", "media_utils"):
        alt = json.loads(json.dumps(prov))
        alt["identity"]["code_sha256"][subfield] = "0" * 64
        c = EC.EmbeddingCache(cdir, alt)
        assert not c.load(), f"cache accepted a different code_sha256.{subfield}!"
        assert "provenance mismatch" in (c.reject_reason or ""), c.reject_reason
        assert c.lookup(str(files[0])) is None
    print("PASS every identity field rejects the whole cache when it differs "
          "(incl. provider -- CPU and GPU embeddings differ by ~1.1e-4 -- and "
          "each of the three code_sha256 entries)")

    # -------------------------------------------------------------------- 3b
    # DEFECT-A REGRESSION.  The field this used to check here was
    # "openwakeword_version" swapped to a bogus string -- vacuous, because the
    # vendored package has no __version__ attribute, so that field was always
    # the constant "unknown" and could never organically differ. This proves
    # the REPLACEMENT (code_sha256) reacts to an ACTUAL mutation of the
    # embedding code, not a hand-typed field, and that embed_corpus() -- not
    # just EmbeddingCache.load() in isolation -- refuses to serve it.
    #
    # Mutate a COPY of the real openwakeword/utils.py (never the live
    # checkout) at the exact function the demonstrated exploit patched
    # (_get_melspectrogram's transform), hash the copy with the same
    # _sha256_file production uses, and inject the resulting identity the way
    # a FRESH PROCESS that had loaded the mutated file would compute it
    # naturally (feature_identity is monkeypatched for exactly one call --
    # nothing about compute() or the audio itself changes).
    oww_utils_path = Path(oww_utils.__file__).resolve()
    real_src = oww_utils_path.read_text()
    marker = "melspec_transform: Callable = lambda x: x/10 + 2"
    assert marker in real_src, (
        "openwakeword/utils.py's _get_melspectrogram signature moved -- "
        "update this test's marker")
    mutated_src = real_src.replace(
        marker, "melspec_transform: Callable = lambda x: x/10 + 999")
    assert mutated_src != real_src
    mutated_path = tmp / "mutated_utils.py"
    mutated_path.write_text(mutated_src)

    real_code_hash = EC._sha256_file(oww_utils_path)
    mutated_code_hash = EC._sha256_file(mutated_path)
    assert mutated_code_hash != real_code_hash, (
        "mutating openwakeword/utils.py's embedding code did not change its "
        "hash -- _sha256_file is broken")
    assert prov["identity"]["code_sha256"]["openwakeword_utils"] == real_code_hash, (
        "feature_identity() is not hashing the real openwakeword/utils.py it "
        "claims to -- defect A is not fixed")

    mutated_prov = json.loads(json.dumps(prov))
    mutated_prov["identity"]["code_sha256"]["openwakeword_utils"] = mutated_code_hash

    a_dir = tmp / "defect_a_cache"          # isolated: never touches cdir
    calls["n"] = 0
    _, st_seed = EC.embed_corpus(files, compute, feat, a_dir, ncpu=EMBED_NCPU,
                                 log=lambda *a: None)
    assert st_seed["misses"] == len(files) and calls["n"] == len(files)
    calls["n"] = 0
    _, st_before = EC.embed_corpus(files, compute, feat, a_dir, ncpu=EMBED_NCPU,
                                   log=lambda *a: None)
    assert st_before["hits"] == len(files) and calls["n"] == 0, (
        "sanity check failed before the mutation was even introduced")

    real_feature_identity = EC.feature_identity
    EC.feature_identity = lambda f, ncpu=None: mutated_prov
    try:
        calls["n"] = 0
        _, st_after = EC.embed_corpus(files, compute, feat, a_dir,
                                      ncpu=EMBED_NCPU, log=lambda *a: None)
    finally:
        EC.feature_identity = real_feature_identity

    assert st_after["cache_rejected"], (
        "embed_corpus SERVED a cache populated under different embedding "
        "code -- this is the exact exploit: seed a cache, change the "
        "melspectrogram code, re-run in a fresh process, and it must miss "
        "everything, not silently serve stale values")
    assert st_after["misses"] == len(files) and calls["n"] == len(files), st_after
    print("PASS mutating a COPY of openwakeword/utils.py's embedding code "
          "(not a hand-typed field) changes its hash and forces embed_corpus "
          "to reject the cache and recompute every file -- the vacuous "
          "openwakeword_version constant is gone")

    # -------------------------------------------------------------------- 4
    other = EC.feature_identity(feat, ncpu=1)
    assert other["identity"] == prov["identity"], (
        "ncpu leaked into the cache identity -- it is provably bit-identical, "
        "so this would only cause spurious rebuilds")
    c = EC.EmbeddingCache(cdir, other)
    assert c.load(), c.reject_reason
    calls["n"] = 0
    _, st = EC.embed_corpus(files, compute, feat, cdir, ncpu=1,
                            log=lambda *a: None)
    assert st["hits"] == len(files) and calls["n"] == 0, st
    print("PASS ncpu is recorded but does not invalidate")

    # -------------------------------------------------------------------- 5
    shutil.copytree(cdir, tmp / "corrupt")
    bad = tmp / "corrupt"
    # (a) store truncated relative to its index
    good_emb = np.load(bad / "emb.npy")
    np.save(bad / "emb.npy", good_emb[: len(good_emb) // 2])
    c = EC.EmbeddingCache(bad, prov)
    assert not c.load(), "accepted a store shorter than its index"
    assert "past the end" in (c.reject_reason or ""), c.reject_reason
    # (b) garbage store
    (bad / "emb.npy").write_bytes(b"not a numpy file")
    c = EC.EmbeddingCache(bad, prov)
    assert not c.load() and "unreadable" in (c.reject_reason or ""), c.reject_reason
    # (c) missing manifest
    (bad / "manifest.json").unlink()
    c = EC.EmbeddingCache(bad, prov)
    assert not c.load(), "accepted a store with no manifest"
    # (d) unparsable manifest
    (bad / "manifest.json").write_text("{ not json")
    c = EC.EmbeddingCache(bad, prov)
    assert not c.load() and "manifest" in (c.reject_reason or ""), c.reject_reason
    # (e) empty directory
    c = EC.EmbeddingCache(tmp / "nothing_here", prov)
    assert not c.load() and c.lookup(str(files[0])) is None
    print("PASS truncated / corrupt / manifest-less / absent stores are all "
          "rejected, never served")

    # a rejected cache must still return CORRECT embeddings, by recomputing
    calls["n"] = 0
    res, st = EC.embed_corpus(files, compute, feat, tmp / "corrupt",
                              ncpu=EMBED_NCPU, log=lambda *a: None)
    assert st["cache_rejected"] and st["misses"] == len(files)
    for (_, a), (_, d) in zip(ref, res):
        assert np.array_equal(a, d)
    print("PASS a rejected cache falls back to recomputing, still exact")

    # -------------------------------------------------------------------- 7
    # A file that cannot be embedded is reported, not cached, not fatal.
    junk = work / "junk.wav"
    junk.write_bytes(b"RIFF....WAVEjunk")
    res, st = EC.embed_corpus(files + [junk], compute, feat, cdir,
                              ncpu=EMBED_NCPU, log=lambda *a: None)
    assert res[-1][1] is None and len(st["errors"]) == 1, st
    c = EC.EmbeddingCache(cdir, prov)
    c.load()
    assert c.lookup(str(junk)) is None
    print("PASS an unreadable file is reported, skipped, and never cached")
    # -------------------------------------------------------------------- 8
    # scoped_cache_dir: the flat per-file caches in make_media_negatives.py /
    # mine_hard_negatives.py get their provenance from the PATH.
    base = tmp / "scoped"
    logs = []
    d1 = EC.scoped_cache_dir(base, feat, ncpu=EMBED_NCPU, log=logs.append)
    assert d1.parent == base and (d1 / "manifest.json").is_file()
    d2 = EC.scoped_cache_dir(base, feat, ncpu=1, log=logs.append)
    assert d1 == d2, "ncpu changed the scoped directory"

    alt = json.loads(json.dumps(prov))
    alt["identity"]["provider"] = "SomeOtherExecutionProvider"
    d3 = EC.scoped_cache_dir_for_provenance(base, alt, log=logs.append)
    assert d3 != d1, "a different provider reused the same directory!"

    # a legacy, provenance-free entry sitting in the base must be reported
    (base / "legacy__thing.npy").write_bytes(b"")
    logs.clear()
    EC.scoped_cache_dir(base, feat, log=logs.append)
    assert any("legacy" in m for m in logs), logs

    # a manifest that disagrees with its own directory name is fatal
    (d1 / "manifest.json").write_text(json.dumps(alt))
    try:
        EC.scoped_cache_dir(base, feat, log=logs.append)
    except SystemExit:
        pass
    else:
        raise AssertionError("a manifest/directory-name mismatch was tolerated")
    print("PASS scoped_cache_dir separates providers, ignores provenance-free "
          "legacy entries, and refuses a manifest that contradicts its path")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("PASS test_embed_cache")
