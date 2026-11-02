#!/usr/bin/env python
"""Verify the .train()/.eval() patch in openwakeword/train.py.

Upstream never calls either (zero grep hits), so the model stays in whatever mode
it was constructed in for the whole run. That is harmless for the shipped
LayerNorm heads -- LayerNorm and the plain Linear stack behave identically in both
modes -- and silently wrong for anything with BatchNorm or Dropout:

  * validation would normalise with the *validation batch's own* statistics
    instead of the running estimates, and
  * every validation forward would additionally *mutate* those running estimates,
    so validation would corrupt the very thing it is measuring.

Both are the concrete blockers to a mel branch using BatchNorm (which folds into
the preceding conv at export, unlike GroupNorm).

The test wraps a BatchNorm1d around the real head so those two effects become
observable, then runs a short `train_model`. Run with ./.venv/bin/python.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "openwakeword"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from openwakeword.train import Model  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

FRAMES, FEATS = 16, 96
MAX_STEPS, VAL_STEP = 6, 4


class BNProbe(nn.Module):
    """The real head with a BatchNorm in front, recording mode + running stats.

    Every forward appends (self.training, running_mean snapshot taken BEFORE the
    BatchNorm sees this batch), which is what lets the assertions below tell a
    training forward from a validation one and detect stat mutation.
    """

    def __init__(self, inner):
        super().__init__()
        self.bn = nn.BatchNorm1d(FRAMES)
        self.inner = inner
        self.log = []

    def forward(self, x):
        self.log.append((self.training, self.bn.running_mean.detach().clone()))
        return self.inner(self.bn(x))


def batch(n, scale, seed):
    """(x, y) with half positives, drawn far from the other batches' distribution."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, FRAMES, FEATS, generator=g) * scale + scale
    y = torch.zeros(n, dtype=torch.int64)
    y[: n // 2] = 1
    return x, y


oww = Model(n_classes=1, input_shape=(FRAMES, FEATS), model_type="dnn", layer_dim=16)
probe = BNProbe(oww.model)
oww.model = probe

# Distinct scales so that a BatchNorm update from validation data would move
# running_mean by an unmistakable amount rather than a rounding error.
train_batches = [batch(200, 1.0, 100 + i) for i in range(MAX_STEPS)]
val_batch = [batch(64, 20.0, 900)]
fp_val_batch = [batch(64, 40.0, 901)]

oww.train_model(
    X=iter(train_batches),
    max_steps=MAX_STEPS,
    warmup_steps=1,
    hold_steps=1,
    X_val=val_batch,
    false_positive_val_data=fp_val_batch,
    val_steps=[VAL_STEP],
    negative_weight_schedule=[10],
    val_set_hrs=1,
)

modes = [m for m, _ in probe.log]
stats = [s for _, s in probe.log]
assert len(modes) >= MAX_STEPS + 2, f"expected training + validation forwards, got {len(modes)}"

# 1. Some forward ran in eval mode. This alone is the regression: unpatched, every
#    entry is True because .eval() is never called anywhere in the trainer.
assert False in modes, ("no forward ran in eval mode -- train_model never called "
                        ".eval(), so validation used batch statistics")
print("  PASS  validation forwards run in eval mode")

# 2. Training forwards ran in train mode (a blanket .eval() would be just as wrong:
#    BatchNorm would never accumulate running statistics at all).
assert modes[0] is True, "the first training forward must be in train mode"
assert modes[-1] is True, "training must resume in train mode after validation"
assert sum(modes) >= MAX_STEPS, f"expected >= {MAX_STEPS} train-mode forwards, got {sum(modes)}"
print("  PASS  training forwards run in train mode, and resume after validation")

# 3. Exactly the validation forwards were in eval mode (2: one fp-val, one X_val),
#    and they are contiguous -- i.e. the mode is flipped around the validation
#    block, not left flipped or toggled per forward.
eval_ndcs = [i for i, m in enumerate(modes) if not m]
assert len(eval_ndcs) == 2, f"expected 2 eval-mode forwards, got {len(eval_ndcs)} at {eval_ndcs}"
assert eval_ndcs == list(range(eval_ndcs[0], eval_ndcs[0] + 2)), \
    f"eval-mode forwards are not contiguous: {eval_ndcs}"
print("  PASS  exactly the two validation forwards are in eval mode, contiguously")

# 4. The validation forwards did not mutate the BatchNorm running statistics. In
#    train mode each would drag running_mean toward its own (deliberately far off)
#    batch mean, so validation would corrupt the statistics it is measuring.
first_eval = eval_ndcs[0]
after_val = first_eval + 2
assert torch.equal(stats[first_eval], stats[first_eval + 1]), \
    "running_mean changed between the two validation forwards -- BatchNorm was in train mode"
assert torch.equal(stats[first_eval], stats[after_val]), \
    "running_mean changed across the validation block -- validation mutated the running stats"
# And a sanity check on the probe itself: training forwards DO move the stats, so
# the equality above is evidence of eval mode, not of a dead BatchNorm.
assert not torch.equal(stats[0], stats[1]), \
    "running_mean never moved at all -- the probe is not measuring anything"
print("  PASS  validation left BatchNorm running statistics untouched")

# 5. average_models() hands back a model in eval mode: those weights were never
#    trained as such and every caller validates or exports it directly. It must also
#    survive a BatchNorm state_dict at all -- upstream's `tensor /= n` over every
#    entry raises on BatchNorm's int64 `num_batches_tracked`.
assert oww.best_models, "no checkpoints were saved, cannot test average_models"
averaged = oww.average_models(models=oww.best_models)
assert averaged.training is False, "average_models must return a model in eval mode"
print("  PASS  average_models accepts a BatchNorm state_dict and returns it in eval mode")

# 6. Averaging is still an exact elementwise mean of the float weights, and the
#    integer counter is carried over rather than mangled.
a = Model(n_classes=1, input_shape=(FRAMES, FEATS), model_type="dnn", layer_dim=16)
b = Model(n_classes=1, input_shape=(FRAMES, FEATS), model_type="dnn", layer_dim=16)
mean = a.average_models(models=[BNProbe(a.model), BNProbe(b.model)])
sd_a, sd_b, sd_mean = a.model.state_dict(), b.model.state_dict(), mean.state_dict()
for key in sd_mean:
    if key.startswith("inner."):
        want = (sd_a[key[len("inner."):]] + sd_b[key[len("inner."):]]) / 2
        assert torch.allclose(sd_mean[key], want, atol=0, rtol=0), f"{key} is not the exact mean"
assert sd_mean["bn.num_batches_tracked"].dtype == torch.int64, "integer buffer changed dtype"
print("  PASS  float weights are the exact elementwise mean; the int64 buffer is preserved")

print("\nPASS test_train_eval_modes")
