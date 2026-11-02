#!/usr/bin/env bash
# Run every remaining stage to completion, waiting on whatever is still in
# flight. Safe to re-run: each underlying step skips work already done.
#
#   nohup ./resume_all.sh > resume.log 2>&1 &
#   tail -f resume.log
#   CONFIG=my_other_model.yaml nohup ./resume_all.sh > resume.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

# CONFIG is exported into this process's environment (or defaulted here) so
# every ./04_train.sh / ./05_export_tflite.sh call below inherits the same
# value automatically -- this script doesn't pass it explicitly, bash just
# propagates env vars to child processes. Only resolved here for the final
# artifact listing, which used to hardcode my_custom_model/ and silently
# pointed at the wrong directory for any other CONFIG.
CONFIG=${CONFIG:-my_model.yaml}
OUTPUT_DIR=$(python -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['output_dir'])")

step() { echo; echo "### $* ($(date '+%F %H:%M:%S'))"; }

# Wait for any download or generation started outside this script, so we never
# get two writers on the same file.
step "Waiting for in-flight downloads/generation"
while pgrep -f "ACAV100M_2000_hrs" >/dev/null || pgrep -f "train.py.*generate_clips" >/dev/null; do
    sleep 30
done
echo "nothing in flight"

step "Clips"
./04_train.sh clips

step "Feature files"
python 02_download_data.py --only features

step "Sanity-checking the feature arrays"
python - <<'PY'
import numpy as np
for name, want_dims in (("openwakeword_features_ACAV100M_2000_hrs_16bit.npy", 3),
                        ("validation_set_features.npy", 2)):
    a = np.load(name, mmap_mode="r")
    assert a.ndim == want_dims, f"{name}: unexpected shape {a.shape}"
    # touch both ends: a resumed download that silently went wrong shows up here
    float(np.asarray(a[0]).sum()); float(np.asarray(a[-1]).sum())
    print(f"  {name}: {a.shape} {a.dtype} OK")
PY

step "Augment"
./04_train.sh augment

step "Train"
./04_train.sh train

step "Export tflite"
./05_export_tflite.sh

step "Done"
ls -la "$OUTPUT_DIR"/*.onnx "$OUTPUT_DIR"/*.tflite
