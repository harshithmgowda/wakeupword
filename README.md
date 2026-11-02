# Nexaura Wake Word ("Hey Nexa") 🎙️

Custom, fully offline on-device wake-word detection engine for the voice phrase **"Hey Nexa"** in the **Nexaura** AI assistant mobile application.

Trained using local GPU acceleration on an **NVIDIA GeForce RTX 5050 Laptop GPU** with **openWakeWord** architecture.

---

## 🚀 Quick Start for Flutter Integration

If you are integrating this model into your Flutter project:

1. **Drop-in Folder**: The [`hey_nexa_flutter_dropin/`](./hey_nexa_flutter_dropin/) directory contains the pre-structured files:
   - `assets/models/`: ONNX and TFLite models (`melspectrogram`, `embedding_model`, `hey_nexa`)
   - `lib/services/nexa_wake_word_service.dart`: Production audio chunking (80ms PCM16), state machine, debounce, and mutual exclusion
   - `lib/screens/wake_word_demo.dart`: Interactive test screen with probability gauge and threshold sliders
2. **AI Integration**: Copy the prompt in [`hey_nexa_flutter_dropin/AI_AGENT_PROMPT.txt`](./hey_nexa_flutter_dropin/AI_AGENT_PROMPT.txt) and paste it into your AI coding assistant (Cursor / Antigravity / Claude) in your Flutter app codebase.
3. **Release Archive**: Download [`HEY_NEXA_RELEASE.zip`](./HEY_NEXA_RELEASE.zip) for the standalone bundle.

---

## 📊 Model & Benchmark Performance

- **Target Phrase**: `hey nexa` (Phonetics: `N EH1 K S AH0`)
- **Classifier Precision**: **96.50%** at recommended threshold `0.50`
- **Adversarial Negative Rejection**: **98.3%** of hard negatives rejected (*"Hey Alexa"*, *"Hey Nexus"*, *"Hey Nessa"*, *"Next"*)
- **Inference Latency**: **0.019 ms / frame** on CPU (4,200× faster than the 80.0 ms real-time audio chunk deadline)
- **Mobile Latency**: Expected **1.5 – 3.0 ms / frame** on mobile ARM

### Benchmark Metrics Table:

| Threshold | True Positives | Misses (FN) | False Positives | True Negatives | Recall | Precision | Hard Negatives Rejected |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.50 (Recommended)** | 276 | 324 | 10 | 590 | 46.00% | **96.50%** | **590 / 600 (98.3%)** |
| **0.70** | 258 | 342 | 10 | 590 | 43.00% | **96.27%** | **590 / 600 (98.3%)** |
| **0.85** | 230 | 370 | 7 | 593 | 38.33% | **97.05%** | **593 / 600 (98.8%)** |
| **0.90** | 195 | 405 | 4 | 596 | 32.50% | **97.99%** | **596 / 600 (99.3%)** |

*(Detailed benchmarks available in [evaluation_report.md](./evaluation_report.md) and [evaluation.json](./evaluation.json))*

---

## 🧠 Pipeline Architecture

Wake-word inference runs in 3 stages:

```
[16 kHz Audio Stream]
        │
        ▼ (1280 samples / 80ms chunks)
┌───────────────────────────────────────────────┐
│ Stage 1: Mel Spectrogram Generator            │
│ melspectrogram.onnx / melspectrogram.tflite   │
└───────────────────────────────────────────────┘
        │
        ▼ (Mel features [1, 76, 32, 1])
┌───────────────────────────────────────────────┐
│ Stage 2: Audio Embedding Extractor            │
│ embedding_model.onnx / embedding_model.tflite │
└───────────────────────────────────────────────┘
        │
        ▼ (Acoustic embedding [1, 96])
┌───────────────────────────────────────────────┐
│ Stage 3: "Hey Nexa" Custom Classifier         │
│ hey_nexa.onnx / hey_nexa.tflite               │
└───────────────────────────────────────────────┘
        │
        ▼ (Wake Probability [0.0 - 1.0])
[Trigger if probability >= 0.50 for 2 frames]
```

---

## 🔒 Verified Checksums (SHA-256)

```
70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f  models/onnx/embedding_model.onnx
1dffa5b7367fc164ffbf85b65aabf46cb538779a31e2ced184b346950683c588  models/onnx/hey_nexa.onnx
ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f  models/onnx/melspectrogram.onnx
c0aea21eb84a4ce90a08c870da41b7a7173b45269e6a3207c71d67c40f3a59d8  models/tflite/embedding_model.tflite
dfc5bc9395388b6b1371139b9e9a3fd12efd5883f0c40c32efb75809519530e6  models/tflite/hey_nexa.tflite
96fa0adccb6e8cf95cb14465409a1a2898ee4a96a85bb9ed3c7eb0e68bf163e8  models/tflite/melspectrogram.tflite
```

---

## 📜 Documentation & Provenance

- [README_START_HERE.md](./README_START_HERE.md) — Comprehensive guide to files and architecture
- [evaluation_report.md](./evaluation_report.md) — Detailed threshold metrics & benchmarking
- [integration_notes.md](./integration_notes.md) — Production audio pipeline & mutual exclusion architecture
- [android_changes.md](./android_changes.md) — AndroidManifest & Gradle configuration
- [LICENSES_AND_DATA_PROVENANCE.md](./LICENSES_AND_DATA_PROVENANCE.md) — Clean training data provenance (Option B)
- [environment_report.md](./environment_report.md) — GPU, CUDA, and environment audit
