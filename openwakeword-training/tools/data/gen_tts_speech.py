#!/usr/bin/env python
"""Generate random TTS speech clips (TV-dialog stand-in) with piper.

Sentences are random word sequences from the CMU dictionary: unnatural prose,
but natural *speech audio*, which is what matters for a background masker.
    ./.venv/bin/python tools/data/gen_tts_speech.py --out tv_mix_speech --n 3000 --seed 1

Resumable: file names and their target texts are derived deterministically
from --seed, independent of what has already been generated. On each run we
recompute the full (text, file_name) list and only ask generate_samples() to
synthesize the ones still missing (or truncated/corrupt) on disk -- it has no
skip-existing logic of its own and will happily overwrite file index 0..N-1
from scratch on every call, so we must filter *before* calling it.
"""
import argparse, random, sys, wave
from pathlib import Path

# python puts the *script's* directory on sys.path, not the cwd, so the
# repo-root modules below have to be found explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from repo_paths import REPO_ROOT  # noqa: E402

HERE = REPO_ROOT
sys.path.insert(0, str(HERE / "piper-sample-generator"))
from generate_samples import generate_samples  # noqa: E402
import pronouncing  # noqa: E402


def _is_valid_wav(path):
    try:
        with wave.open(str(path), "rb") as f:
            return f.getnframes() > 0
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=100)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    words = sorted({w for w in pronouncing.cmudict.words()
                    if w.isalpha() and 3 <= len(w) <= 10})
    texts = [" ".join(rng.sample(words, rng.randint(8, 14)))
             for _ in range(args.n)]
    file_names = [f"speech_{args.seed}_{i:05d}.wav" for i in range(args.n)]

    missing = [(t, f) for t, f in zip(texts, file_names)
               if not _is_valid_wav(out / f)]
    if not missing:
        print(f"[tts] {len(file_names)} clips present, skipping")
        return

    print(f"[tts] {len(file_names) - len(missing)}/{len(file_names)} clips "
          f"already present, generating {len(missing)} more")
    m_texts, m_names = zip(*missing)
    generate_samples(text=list(m_texts), output_dir=str(out),
                     max_samples=len(missing), batch_size=args.batch_size,
                     auto_reduce_batch_size=True,
                     length_scales=[0.8, 1.0, 1.2],
                     file_names=list(m_names))
    print(f"[tts] wrote {len(list(out.glob('*.wav')))} clips -> {out}")


if __name__ == "__main__":
    main()
