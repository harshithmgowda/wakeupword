#!/usr/bin/env bash
# Step 0: create the Python environment and fetch the repos + base models.
# Safe to re-run; existing pieces are skipped.
set -euo pipefail
cd "$(dirname "$0")"

PY=3.10

if [ ! -d .venv ]; then
    uv venv --python "$PY" .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# --- repos ---------------------------------------------------------------
if [ ! -d piper-sample-generator ]; then
    git clone https://github.com/rhasspy/piper-sample-generator
    (cd piper-sample-generator && git checkout 213d4d5)
fi
if [ ! -d openwakeword ]; then
    git clone https://github.com/dscripka/openwakeword
fi

# --- torch first, so the CUDA build wins over any transitive CPU pin ------
# cu121 wheels cover the RTX 4090 (sm_89).
uv pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu121

# --- everything else -----------------------------------------------------
# numpy is held below 2.0: piper-sample-generator, audiomentations 0.33 and
# speechbrain 0.5.14 all predate the numpy 2 ABI break.
uv pip install \
    "numpy<2" \
    "setuptools<81" \
    piper-tts piper-phonemize-cross webrtcvad \
    mutagen==1.47.0 \
    torchinfo==1.8.0 \
    torchmetrics==1.2.0 \
    speechbrain==0.5.14 \
    audiomentations==0.33.0 \
    torch-audiomentations==0.11.0 \
    acoustics==0.2.6 \
    pronouncing==0.2.0 \
    deep-phonemizer==0.0.19 \
    onnx==1.19.1 onnxsim \
    scipy scikit-learn tqdm pyyaml requests soundfile

# --- onnxruntime: GPU build ---------------------------------------------
# The feature extractors (melspectrogram.onnx, embedding_model.onnx) run under
# onnxruntime, and they are the dominant cost of the whole pipeline. The plain
# `onnxruntime` wheel is CPU-only, so asking for CUDAExecutionProvider merely warns
# and falls back -- on this box that meant every feature was computed on one core
# (2 h 48 m of a 3 h 03 m run). The GPU wheel takes a 32-clip batch from 520 ms to
# 16 ms. Only one of onnxruntime / onnxruntime-gpu may be installed: they provide
# the same `onnxruntime` module.
#
# No separate CUDA install is needed -- the CUDA EP dlopens the cuBLAS/cuDNN that
# torch's nvidia-* wheels already provide, which is why torch must be imported
# before onnxruntime (openwakeword/train.py does). If the EP still fails to load,
# compute_features_from_generator logs it and falls back to a properly threaded
# CPU path rather than silently running on one core.
uv pip install onnxruntime-gpu==1.23.2

# openwakeword itself, without deps (we pinned them all above)
uv pip install -e ./openwakeword --no-deps

# --- pretrained feature extractors --------------------------------------
REL=https://github.com/dscripka/openWakeWord/releases/download/v0.5.1
MODELS=./openwakeword/openwakeword/resources/models
mkdir -p "$MODELS"
for f in embedding_model.onnx embedding_model.tflite melspectrogram.onnx melspectrogram.tflite; do
    [ -s "$MODELS/$f" ] || wget -q --show-progress "$REL/$f" -O "$MODELS/$f"
done

# --- piper TTS voice -----------------------------------------------------
mkdir -p piper-sample-generator/models
VOICE=piper-sample-generator/models/en_US-libritts_r-medium.pt
[ -s "$VOICE" ] || wget -q --show-progress \
    'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt' \
    -O "$VOICE"

echo
echo "Environment ready. Verifying CUDA:"
python -c "import torch; print(' torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
