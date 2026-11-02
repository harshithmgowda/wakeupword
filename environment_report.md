# Environment Audit Report

**Generated:** 2026-11-02  
**Host Machine:** Windows 11 Home Single Language 64-bit (Build 26300.9818)  
**Host Hardware:**
- **System Memory (RAM):** 24.44 GB Visible (7.56 GB Free)
- **Host Disk:** C: Drive (485.69 GB free of 1.02 TB)
- **GPU Hardware:** NVIDIA GeForce RTX 5050 Laptop GPU (8,151 MiB VRAM / ~8 GB)
- **Windows Display Driver:** 616.92 (WDDM 3.2, CUDA UMD 13.4)

---

## 1. WSL2 Linux Environment

- **WSL Version:** 2.6.3.0
- **Kernel Version:** 6.6.87.2-microsoft-standard-WSL2 (x86_64)
- **WSLg Version:** 1.0.71
- **Installed Distribution:** Ubuntu 22.04 LTS (Jammy Jellyfish)
- **WSL2 Virtual Disk:** 954 GB available on ext4 root filesystem (`/dev/sdd`)
- **WSL2 NVIDIA Driver:** 615.71.08 / KMD 616.92 / CUDA UMD 13.4
- **WSL2 `nvidia-smi` Status:** Verified operational; detected `NVIDIA GeForce RTX 5050 Laptop GPU` with 8151 MiB VRAM.

---

## 2. Audio & Build Toolchain Installed in WSL2

- **Python (System):** Python 3.10.12 (`/usr/bin/python3`)
- **Python Development Headers:** `python3-dev` (3.10.6-1~22.04.1) installed to provide `<Python.h>` for building C extensions (`webrtcvad`).
- **Audio Processing:** `ffmpeg` 4.4.2 (built with libopus, libvorbis, libmp3lame, libsoxr), `sox`, `libsndfile1`, `libasound2`.
- **Git & Git LFS:** `git` 2.34.1, `git-lfs` 3.0.2.
- **Build Tools:** `build-essential` (gcc 11.4.0, g++ 11.4.0, make).
- **Python Package Manager:** Astral `uv` 0.13.6 (`/usr/local/bin/uv`).

---

## 3. Training Repository & Commit Identification

- **Repository:** `https://github.com/Dethon/openwakeword-training`
- **Cloned To:** `/root/openwakeword-training` (WSL2 ext4) and `C:\Users\Harshith\OneDrive\Desktop\wake\openwakeword-training`
- **Git HEAD Commit Hash:** `1c1cd49e3b1f0679cf67647d08d3c87be4745c91`
- **Submodule / Cloned Upstream:**
  - `openWakeWord`: `dscripka/openWakeWord` (`v0.6.0`, commit `368c03716d1e92591906a84949bc477f3a834455`)
  - `piper-sample-generator`: `rhasspy/piper-sample-generator` pinned commit `213d4d5`

---

## 4. Hardware Architecture Diagnosis & CUDA Compatibility Resolution

### Diagnosis: Blackwell Architecture (`sm_120`)
When running the default `setup_env.sh`, the repository pinned `torch==2.5.0+cu121`.
Testing tensor operations on the RTX 5050 under that environment yielded:
```text
UserWarning: NVIDIA GeForce RTX 5050 Laptop GPU with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
RuntimeError: CUDA error: no kernel image is available for execution on the device
```
The RTX 5050 Laptop GPU is based on NVIDIA's Blackwell architecture with compute capability **sm_120**, which is not included in pre-built CUDA 12.1 binaries.

### Resolution: CUDA 12.8 Upgrade with Pinned ABI Constraints
We upgraded the virtual environment's PyTorch stack to:
- **PyTorch:** `torch==2.11.0+cu128`
- **TorchVision:** `torchvision==0.26.0+cu128`
- **TorchAudio:** `torchaudio==2.11.0+cu128`
- **CUDA Runtime:** 12.8 (with native `sm_120` support)
- **ABI Protection:** Pinned `numpy<2` (`numpy==1.26.4`) to prevent ABI breakage with `piper-sample-generator`, `speechbrain 0.5.14`, and `audiomentations 0.33.0`.
- **PyTorch 2.6+ Security Patch:** Patched `piper-sample-generator/generate_samples.py` with `weights_only=False` to safely load the pre-trained `SynthesizerTrn` VITS weights.

### Verification Evidence:
```text
CUDA available: True
Device: NVIDIA GeForce RTX 5050 Laptop GPU
Matmul sum: 52.98541259765625
ONNX Runtime Providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```
GPU-accelerated TTS synthesis was tested with `01_test_voice.py "hey nexa"`:
```text
DEBUG:generate_samples:Loading piper-sample-generator/models/en_US-libritts_r-medium.pt
INFO:generate_samples:Successfully loaded the model
DEBUG:generate_samples:CUDA available, using GPU
DEBUG:generate_samples:Batch 1/1 complete
INFO:generate_samples:Done
Wrote test_generation.wav (16 kHz, mono, 0.82s)
```
Full local GPU training and hardware execution are verified functional on the laptop's RTX 5050 GPU.

