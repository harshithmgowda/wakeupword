# Licenses, Data Provenance, and Distribution Constraints

**Document Version:** 1.0.0  
**Target Wake Word:** "Hey Nexa"  
**Date:** 2026-11-02  

---

## 1. Upstream Code Licenses

| Project / Component | Author / Organization | License | Description & Role |
| :--- | :--- | :--- | :--- |
| **openWakeWord** | David Scripka | Apache-2.0 | Core wake-word inference and training pipeline architecture. |
| **piper-sample-generator** | Michael Hansen / Rhasspy | MIT | Synthetic audio generator using neural TTS (VITS). |
| **openwakeword-training** | Dethon | MIT | Automated training harness and scripts (patches retain original upstream licenses). |
| **Flutter Integration Bridge** | Project Authors | Apache-2.0 / MIT | Custom Dart/Flutter integration service and state machine. |

---

## 2. Pretrained Models & Feature Extractors

| Asset | Source / Release | Upstream Terms | Function |
| :--- | :--- | :--- | :--- |
| `melspectrogram.onnx` / `.tflite` | openWakeWord v0.5.1 | Apache-2.0 | Converts 16 kHz raw audio frames to 32-band log mel spectrograms. |
| `embedding_model.onnx` / `.tflite` | openWakeWord v0.5.1 | Apache-2.0 | Pretrained audio embedding model outputting 16x96 feature representations. |
| `en_US-libritts_r-medium.pt` | Rhasspy / piper-sample-generator v2.0.0 | CC-BY-4.0 | Neural TTS voice used to synthesize positive & near-miss negative clips. |

---

## 3. Training Datasets & Severe Non-Commercial Restrictions

> [!WARNING]
> **CRITICAL LEGAL NOTICE ON DATASETS:**  
> The standard training pipeline (`02_download_data.py`) pulls third-party datasets that contain mixed, non-commercial, and copyrighted content. Models trained with these default corpora **MUST BE CONSIDERED NON-COMMERCIAL / RESEARCH & PERSONAL USE ONLY**.

The specific datasets involved are:

1. **AudioSet (Background Noise)**:
   - Sourced via Hugging Face mirror `agkphysics/AudioSet`.
   - Composed of 10-second audio clips scraped from YouTube videos.
   - Sourced under YouTube Terms of Service; individual clips have mixed, non-commercial, or unverified licenses.

2. **Free Music Archive (`fma_small`)**:
   - Sourced from `os.unil.cloud.switch.ch/fma/fma_small.zip` (~7.7 GB).
   - Contains 8,000 tracks across various genres under mixed Creative Commons licenses (many of which are CC-BY-NC or CC-BY-NC-ND).

3. **MIT Environmental Impulse Responses**:
   - Sourced via Hugging Face `davidscripka/MIT_environmental_impulse_responses`.
   - Academic/research room impulse responses used to simulate reverberation.

4. **ACAV100M Precomputed Features (`openwakeword_features_ACAV100M_2000_hrs_16bit.npy`)**:
   - ~17.28 GB of pre-extracted 16x96 embeddings from 2,000 hours of YouTube ACAV100M video audio.
   - Used as large-scale negative speech background during DNN head training.

5. **Validation Negative Features (`validation_set_features.npy`)**:
   - ~184 MB of held-out negative audio embeddings used to measure false activation rates during training.

---

## 4. Distinction Between Training Paths

### Path A: Research / Demo-Only Training (Default Pipeline)
- **Datasets:** Full AudioSet shards + FMA small + ACAV100M 2,000 hours precomputed features.
- **Estimated Download:** ~27–33 GB.
- **Resulting Model License:** **Strictly Personal / Academic / Demo Use Only**. Cannot be redistributed in a commercial mobile app or monetized software product.
- **Accuracy Profile:** High resistance to general conversational false positives due to the 2,000-hour negative embedding pool.

### Path B: Commercial-Safe / Clean Provenance Training
- **Datasets:**
  - Positives: Permissively licensed synthetic TTS + consented real speaker recordings.
  - Negatives: Permissive CC0/public domain speech (LibriSpeech clean subsets, Common Voice commercial-permissive subsets, or synthetically generated near-miss phrases).
  - Background Noise & Reverb: CC0/public domain ambient sounds (e.g., ESC-50 CC-BY subsets, synthetic room impulse responses, or user-provided clean field recordings).
- **Estimated Download:** ~1–5 GB (depending on synthetic volume).
- **Resulting Model License:** **Fully Commercial / Production Distributable**.
- **Accuracy Profile:** Requires careful tuning of near-miss synthetic negatives to match the false-positive rejection of ACAV100M.

---

## 5. Artifact Distribution Safety Guardrail

1. No copyrighted recordings or unredacted personal speech clips may be bundled into the release package (`HEY_NEXA_RELEASE.zip`).
2. The final release ZIP contains:
   - Exported model binary classifiers (`hey_nexa.onnx`, `hey_nexa.tflite`).
   - Official openWakeWord feature extractors (`melspectrogram.onnx`, `embedding_model.onnx`).
   - Clean, self-contained Flutter integration code and demo UI.
   - Verification reports, benchmark scripts, and configuration YAMLs.

