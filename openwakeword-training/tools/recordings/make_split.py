#!/usr/bin/env python
"""Split the real recordings into train/test, stratified by speaker and condition.

The test clips are held out before any training happens and are never used for
training, validation, or model selection. With only ~20 test clips, one clip is
worth 5% -- so treat small differences as noise, not improvement.

    ./.venv-analysis/bin/python tools/recordings/make_split.py

NOTE: this does NOT describe the split model v5 was actually built on. That one
is already materialised as directories -- <root>/train (106 clips) and
<root>/eval (27) -- and was decided later and differently: by v5 every clip of
the original ~/rec set was being trained on, and the hold-out had moved to the
tagged conditions instead. Running this rewrites split_train.txt/split_test.txt
with a fresh, independent draw that will disagree with those directories. Use it
to design a *new* split, not to recover v5's.
"""
import sys
import os
import random
from collections import defaultdict
from pathlib import Path

# python puts the *script's* directory on sys.path, not the cwd, so the
# repo-root modules below have to be found explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from repo_paths import DATA_ROOT, REPO_ROOT  # noqa: E402

#: the repo's own metadata directory (labels, splits) -- not the audio corpus
DATA_DIR = REPO_ROOT / "data"

SEED = 1337          # fixed so the split is reproducible across runs
TEST_FRACTION = 0.4


labels = {}
for line in (DATA_DIR / "labels.tsv").read_text().splitlines():
    if line.strip():
        name, cond = line.split("\t")
        labels[name] = cond

# The pool is the 133 original recordings, train/ + eval/ together. unused/ is
# excluded on purpose: rec_trimmed/ and rec_trimmed_25/ are re-cuts of clips
# already in train/ and reuse their stems, so globbing the whole root would
# enter the same recording into the split three times.
pool = [DATA_ROOT / "train", DATA_ROOT / "eval"]
files = sorted((f for d in pool if d.is_dir() for f in d.glob("*.wav")),
               key=lambda p: p.name)
if not files:
    raise SystemExit(f"no wav files under {DATA_ROOT}/{{train,eval}}")

groups = defaultdict(list)
for f in files:
    speaker = f.stem.rsplit("_", 1)[0]
    groups[(speaker, labels.get(f.stem, "unknown"))].append(f.stem)

rng = random.Random(SEED)
train, test = [], []
for key in sorted(groups):
    names = sorted(groups[key])
    rng.shuffle(names)
    n_test = max(1, round(len(names) * TEST_FRACTION))
    test += names[:n_test]
    train += names[n_test:]
    print(f"  {key[0]:<6} {key[1]:<7} total {len(names):>2}  ->  train {len(names) - n_test:>2}  test {n_test:>2}")

(DATA_DIR / "split_train.txt").write_text("\n".join(sorted(train)) + "\n")
(DATA_DIR / "split_test.txt").write_text("\n".join(sorted(test)) + "\n")
print(f"\ntrain {len(train)} clips -> split_train.txt")
print(f"test  {len(test)} clips -> split_test.txt   (HELD OUT -- never trained on)")
