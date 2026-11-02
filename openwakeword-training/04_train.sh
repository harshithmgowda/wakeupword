#!/usr/bin/env bash
# Step 4: the three training stages.
#
#   ./04_train.sh              # generate clips -> augment -> train
#   ./04_train.sh train        # re-run just one stage
#
# Stages are separate processes (as in the notebook) because each one frees its
# GPU memory before the next starts. Output lands in $CONFIG's output_dir
# (./my_custom_model by default; a different CONFIG can name a different one,
# which is how several variants coexist side by side).
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

CONFIG=${CONFIG:-my_model.yaml}
[ -f "$CONFIG" ] || { echo "No $CONFIG -- run 03_make_config.py first"; exit 1; }

TRAIN=openwakeword/openwakeword/train.py
mkdir -p logs

MODEL_NAME=$(python -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['model_name'])")
OUTPUT_DIR=$(python -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['output_dir'])")

run_stage() {
    local flag=$1 name=$2
    echo
    echo "=============================================================="
    echo " $name  ($(date '+%H:%M:%S'))"
    echo "=============================================================="
    set +e
    python "$TRAIN" --training_config "$CONFIG" --"$flag" 2>&1 | tee "logs/${name}.log"
    local rc=${PIPESTATUS[0]}
    set -e

    # train.py always ends by calling its built-in tflite converter, which
    # imports the long-dead onnx_tf. That blows up *after* the .onnx is
    # written, so it's not a real failure -- 05_export_tflite.sh does the
    # conversion instead. Any other non-zero exit is genuine.
    if [ "$rc" -ne 0 ]; then
        if [ "$name" = train_model ] \
           && grep -q "No module named 'onnx_tf'" "logs/${name}.log" \
           && [ -f "$OUTPUT_DIR/${MODEL_NAME}.onnx" ]; then
            echo "  (ignoring the expected onnx_tf failure -- the .onnx was saved)"
        else
            echo "  $name failed with exit code $rc"
            exit "$rc"
        fi
    fi
}

# train.py sizes each generation batch by counting files already on disk, so a
# clip left empty by an interrupted run would be counted as done and then break
# augmentation. Prune those first; the stage regenerates whatever is missing.
prune_clips() {
    echo "Verifying existing clips..."
    python verify_clips.py --config "$CONFIG"
}

# Real recordings are added to positive_train BETWEEN clip generation and
# augmentation, so the augment stage gives every duplicate its own background and
# RIR draw. Configs that declare `real_clip_sources` get them; configs that don't
# are unaffected and this is a no-op. Idempotent, so it runs before every augment
# rather than only after generation -- an `augment`-only invocation on a fresh
# clone would otherwise train on synthetic positives alone and look fine.
#
# This used to be a manual step nobody scripted, which is why v6/v7/v8 could not
# be rebuilt from their own configs. See docs/pipeline.md.
inject_real_clips() {
    echo
    echo "Injecting real clips (if this config declares any)..."
    python tools/data/inject_real_clips.py --config "$CONFIG"
}

case "${1:-all}" in
    clips)   prune_clips; run_stage generate_clips generate_clips; inject_real_clips ;;
    augment) inject_real_clips; run_stage augment_clips augment_clips ;;
    train)   run_stage train_model    train_model    ;;
    all)
        prune_clips
        run_stage generate_clips generate_clips
        inject_real_clips
        run_stage augment_clips  augment_clips
        run_stage train_model    train_model
        ;;
    *) echo "usage: $0 [all|clips|augment|train]"; exit 1 ;;
esac

echo
echo "Training done. Run ./05_export_tflite.sh to produce the .tflite."
echo "Artifacts in $OUTPUT_DIR:"
ls -la "$OUTPUT_DIR"/*.onnx "$OUTPUT_DIR"/*.tflite 2>/dev/null || true
