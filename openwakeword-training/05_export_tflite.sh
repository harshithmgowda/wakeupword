#!/usr/bin/env bash
# Step 5: convert the trained .onnx model to .tflite.
#
# openwakeword's built-in converter goes through onnx_tf, which is dead on
# modern python. The notebook works around this with the onnx2tf CLI, and so do
# we. Its dependency set (tensorflow + friends) conflicts with the numpy<2 pin
# the training env needs, so it lives in a throwaway venv of its own.
set -euo pipefail
cd "$(dirname "$0")"

CONFIG=${CONFIG:-my_model.yaml}
MODEL_NAME=$(./.venv/bin/python -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['model_name'])")
OUTPUT_DIR=$(./.venv/bin/python -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['output_dir'])")
ONNX="$OUTPUT_DIR/${MODEL_NAME}.onnx"
[ -f "$ONNX" ] || { echo "Missing $ONNX -- run ./04_train.sh first"; exit 1; }

if [ ! -d .venv-onnx2tf ]; then
    echo "Creating the onnx2tf conversion environment..."
    uv venv --python 3.10 .venv-onnx2tf
    .venv-onnx2tf/bin/python -m ensurepip -q 2>/dev/null || true
    VIRTUAL_ENV=.venv-onnx2tf uv pip install -q \
        onnx2tf onnx==1.19.1 onnxruntime==1.22.1 onnxsim \
        ai_edge_litert==1.4.0 onnx_graphsurgeon sng4onnx \
        tensorflow tf_keras psutil
fi

# -kat / --keep_shape_absolutely_input_names keeps the named input tensor's axis
# order (and its dynamic batch dim) exactly as the ONNX has it, instead of letting
# onnx2tf apply its automatic NCHW->NHWC channel transposition, which would reorder
# the feature axes and silently produce a model that never activates.
#
# The name must be read out of the ONNX, never hardcoded: onnx2tf's own help says
# "If a nonexistent INPUT OP name is specified, it is ignored" -- a typo or a name
# that changed does not fail, it just drops the protection. This used to be
# hardcoded as `onnx____Flatten_0`, which is the name torch invents from the first
# traced op when `input_names` is not passed, so it changed the moment the head
# topology changed. `train.py` now passes `input_names=["input"]`; reading the name
# here keeps both old and new ONNX files working.
#
# onnx2tf sanitizes ':' to '__' in tensor names, so `onnx::Flatten_0` in the ONNX is
# `onnx____Flatten_0` to the flag -- apply the same substitution here.
KAT_NAMES=$(.venv-onnx2tf/bin/python - "$ONNX" <<'PY'
import sys
import onnx

model = onnx.load(sys.argv[1], load_external_data=False)
initializers = {i.name for i in model.graph.initializer}
names = [i.name.replace(":", "__") for i in model.graph.input if i.name not in initializers]
if not names:
    sys.exit("no graph inputs found in the ONNX -- refusing to convert")
print(" ".join(names))
PY
)
echo "Preserving input axis order for: $KAT_NAMES"

# shellcheck disable=SC2086  # KAT_NAMES is a deliberately word-split name list
VIRTUAL_ENV=.venv-onnx2tf .venv-onnx2tf/bin/onnx2tf \
    -i "$ONNX" -o "$OUTPUT_DIR/" -kat $KAT_NAMES

mv "$OUTPUT_DIR/${MODEL_NAME}_float32.tflite" "$OUTPUT_DIR/${MODEL_NAME}.tflite"

# Because a missed -kat name is ignored silently, assert the axis order actually
# survived: the tflite input shape must match the ONNX input shape (dynamic dims
# excepted). Without this check the failure mode is a model that loads, runs, and
# never activates.
.venv-onnx2tf/bin/python - "$ONNX" "$OUTPUT_DIR/${MODEL_NAME}.tflite" <<'PY'
import sys
import onnx
from ai_edge_litert.interpreter import Interpreter

onnx_path, tflite_path = sys.argv[1], sys.argv[2]
model = onnx.load(onnx_path, load_external_data=False)
initializers = {i.name for i in model.graph.initializer}
graph_inputs = [i for i in model.graph.input if i.name not in initializers]
onnx_shape = [d.dim_value if d.dim_param == "" else -1
              for d in graph_inputs[0].type.tensor_type.shape.dim]

tflite_shape = list(Interpreter(model_path=tflite_path).get_input_details()[0]["shape"])
if len(graph_inputs) == 1:
    # Axis 0 is the batch dim; -kat is also what lets it stay dynamic, so it is
    # allowed to differ (1 in the ONNX, -1 in the tflite). Every other axis must
    # match exactly, in order.
    mismatch = len(onnx_shape) != len(tflite_shape) or any(
        o != -1 and t != -1 and o != t
        for o, t in zip(onnx_shape[1:], tflite_shape[1:])
    )
    if mismatch:
        sys.exit(f"input axis order was NOT preserved: onnx {onnx_shape} vs tflite {tflite_shape}. "
                 "The -kat name almost certainly did not match -- onnx2tf ignores unknown names.")
    print(f"input axis order preserved: onnx {onnx_shape} -> tflite {tflite_shape}")
else:
    print(f"{len(graph_inputs)} graph inputs; skipping the single-input axis-order check")
PY

echo
echo "Final models:"
ls -la "$OUTPUT_DIR/${MODEL_NAME}.onnx" "$OUTPUT_DIR/${MODEL_NAME}.tflite"
