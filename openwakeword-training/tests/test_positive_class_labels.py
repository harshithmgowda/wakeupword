#!/usr/bin/env python
"""Verify the positive-class label assignment patch in openwakeword/train.py.

Upstream hardcoded `positive -> 1` and every other feature-data class -> 0, and
only inserted `feature_data_files["positive"]` *after* building that dict, so the
loop had no way to recognise a second positive class. Giving the real over-music
recordings their own class key -- the mechanism that makes their share of each
batch a hyperparameter instead of an accident of corpus size -- therefore trained
them as NEGATIVES, silently: no error, no warning, just a worse model.

`make_label_transforms` replaces that loop. A class is positive if its key is
"positive", starts with "positive_", or is named in the config's optional
`positive_classes` list.

Run with ./.venv/bin/python (no pytest).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "openwakeword"))
import numpy as np  # noqa: E402
from openwakeword.data import mmap_batch_generator  # noqa: E402
from openwakeword.train import make_label_transforms  # noqa: E402

TMP = os.path.join(os.path.dirname(__file__), "_tmp_positive_classes")
os.makedirs(TMP, exist_ok=True)


def labels_for(keys, positive_classes=None):
    """Run each class's transform over one dummy row and return the class->label map."""
    lt = make_label_transforms(keys, positive_classes=positive_classes)
    assert set(lt) == set(keys), f"missing transforms for {set(keys) - set(lt)}"
    out = {}
    for key, fn in lt.items():
        got = fn([key])
        assert len(got) == 1, f"{key}: transform must be 1:1 over the batch, got {got}"
        out[key] = got[0]
    return out


def old_upstream_labels(keys):
    """The behaviour being fixed, reproduced so the test states what regressed."""
    return {key: (1 if key == "positive" else 0) for key in keys}


# --------------------------------------------------------------- unit coverage
# The v7 config's actual class set, plus one extra positive class as phase 3 wants.
V7_KEYS = ["positive", "ACAV100M_sample", "adversarial_negative", "media_negatives"]

got = labels_for(V7_KEYS)
assert got == {"positive": 1, "ACAV100M_sample": 0,
               "adversarial_negative": 0, "media_negatives": 0}, got
assert got == old_upstream_labels(V7_KEYS), \
    "v7's own class set must be labelled exactly as before -- this patch is a no-op for it"
print("  PASS  v7's existing class set is labelled identically to upstream")

# 1. the `positive_` prefix convention
keys = V7_KEYS + ["positive_real_music"]
got = labels_for(keys)
assert got["positive_real_music"] == 1, \
    f"a `positive_`-prefixed class must be labelled positive, got {got['positive_real_music']}"
assert got["positive"] == 1 and got["media_negatives"] == 0, got
assert old_upstream_labels(keys)["positive_real_music"] == 0, \
    "the regression this guards against is upstream labelling it 0"
print("  PASS  a `positive_`-prefixed extra class is labelled 1 (upstream: 0)")

# 2. the explicit config list, for a class key that does not follow the convention
keys = V7_KEYS + ["real_music", "replay_positives"]
got = labels_for(keys, positive_classes=["real_music", "replay_positives"])
assert got["real_music"] == 1 and got["replay_positives"] == 1, got
assert got["ACAV100M_sample"] == 0 and got["adversarial_negative"] == 0, got
print("  PASS  `positive_classes` promotes arbitrary class keys to label 1")

# 3. classes not named stay negative, and an empty/absent list changes nothing
assert labels_for(keys, positive_classes=None)["real_music"] == 0
assert labels_for(keys, positive_classes=[])["real_music"] == 0
assert labels_for(V7_KEYS, positive_classes=None) == labels_for(V7_KEYS)
print("  PASS  no `positive_classes` (None or []) preserves upstream labelling exactly")

# 4. a duplicated key (the call site passes ["positive"] + config keys + [...]) is fine
got = labels_for(["positive", "positive", "ACAV100M_sample"])
assert got == {"positive": 1, "ACAV100M_sample": 0}, got
print("  PASS  duplicate keys in the input list are harmless")

# --------------------------------------------------- integration through batching
# The unit test above checks the map; this checks that a batch actually carries
# label 1 for the extra positive class, through the real generator.
N_ROWS, FRAMES, FEATS = 8, 16, 96
CLASS_MARKER = {"positive": 1.0, "positive_real_music": 2.0,
                "ACAV100M_sample": 3.0, "adversarial_negative": 4.0}
data_files = {}
for name, marker in CLASS_MARKER.items():
    path = os.path.join(TMP, f"{name}.npy")
    np.save(path, np.full((N_ROWS, FRAMES, FEATS), marker, dtype=np.float32))
    data_files[name] = path

n_per_class = {name: 2 for name in CLASS_MARKER}
gen = mmap_batch_generator(
    data_files,
    n_per_class=n_per_class,
    label_transform_funcs=make_label_transforms(list(data_files.keys())),
)
X, y = next(gen)
assert X.shape == (2 * len(CLASS_MARKER), FRAMES, FEATS), X.shape

# Recover each row's class from its marker value and check the label it was given.
marker_to_class = {v: k for k, v in CLASS_MARKER.items()}
expected_positive = {"positive", "positive_real_music"}
seen = set()
for row, label in zip(X, y):
    cls = marker_to_class[float(row[0, 0])]
    seen.add(cls)
    want = 1 if cls in expected_positive else 0
    assert label == want, f"{cls} row got label {label}, expected {want}"
assert seen == set(CLASS_MARKER), f"batch did not include every class: {seen}"
assert int((y == 1).sum()) == 4, f"expected 4 positive rows (2 classes x 2), got {(y == 1).sum()}"
print("  PASS  a real batch labels both positive classes 1 and both negative classes 0")

print("\nPASS test_positive_class_labels")
