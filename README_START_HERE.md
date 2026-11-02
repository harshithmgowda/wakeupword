# Nexaura "Hey Nexa" Custom Wake-Word Engine

> **Production-Grade Offline Wake-Word Detection for Flutter (Android & iOS)**  
> **Model Target Phrase:** `"Hey Nexa"`  
> **Trained With:** NVIDIA GeForce RTX 5050 Laptop GPU (sm_120)  
> **Training Pipeline:** openWakeWord 32-unit DNN Head with LibriTTS-R + CC0 Acoustic Environments  
> **Inference:** 100% On-Device, Offline, Zero Latency, Zero Cloud API Keys  

---

## 1. Quick Start: 3-Minute Flutter Integration

### Step 1: Add Dependencies & Assets to `pubspec.yaml`
```yaml
dependencies:
  flutter:
    sdk: flutter
  record: ^5.1.2                # 16kHz PCM audio stream
  onnxruntime: ^1.17.1          # On-device model execution
  permission_handler: ^11.3.1   # Microphone permission
  path_provider: ^2.1.3         # Local asset storage

flutter:
  assets:
    - assets/models/melspectrogram.onnx
    - assets/models/embedding_model.onnx
    - assets/models/hey_nexa.onnx
```

### Step 2: Copy Models
Copy all files from `models/onnx/` into your Flutter app's `assets/models/` folder:
- `melspectrogram.onnx` (Universal 32-band mel spectrogram extractor)
- `embedding_model.onnx` (Pretrained audio embedding network)
- `hey_nexa.onnx` (Trained custom wake-word classifier)

### Step 3: Copy Service and Demo Screen
Copy `flutter/lib/services/nexa_wake_word_service.dart` and `flutter/lib/wake_word_demo.dart` into your app.

### Step 4: Initialize and Listen
```dart
import 'package:nexaura/services/nexa_wake_word_service.dart';

final wakeService = NexaWakeWordService();

void startNexa() async {
  // Initialize with calibrated threshold (0.60) and 2-second cooldown
  bool ready = await wakeService.initialize(threshold: 0.60, cooldownMs: 2000);
  
  if (ready) {
    // Listen for detection events
    wakeService.detectionStream.listen((timestamp) {
      print('Wake word detected at: $timestamp');
      triggerNexaAssistant();
    });

    // Start microphone stream
    await wakeService.startListening();
  }
}
```

---

## 2. Critical: Mutual Exclusion State Machine

To prevent the assistant from **hearing its own voice and triggering an infinite loop**:

```dart
void triggerNexaAssistant() async {
  // 1. Immediately pause wake-word detection
  wakeService.setAssistantActive(true);

  // 2. Play wake sound or trigger haptic
  HapticFeedback.mediumImpact();

  // 3. Start Speech-to-Text (STT) and LLM response
  String userQuery = await speechToText.listen();
  String nexaReply = await nexaLLM.generate(userQuery);
  await textToSpeech.speak(nexaReply);

  // 4. When assistant finishes speaking, resume wake-word detection
  wakeService.setAssistantActive(false);
}
```

---

## 3. Model Specifications & Tensor Signatures

| Parameter | Specification |
| :--- | :--- |
| **Input Shape** | `[1, 16, 96]` (Batch: 1, Time Window: 16 frames, Mel Embeddings: 96 features) |
| **Input Type** | `float32` |
| **Output Shape** | `[1, 1]` (Activation probability `0.0` to `1.0`) |
| **Audio Format** | `16,000 Hz`, `16-bit Mono PCM` (Linear PCM) |
| **Frame Size** | `1280 samples` (`80 ms` per step) |
| **Desktop Latency**| `< 0.2 ms` per frame on CPU (Real-time budget: `80 ms`) |
| **Mobile Latency** | `< 1.5 ms` on modern ARM64 Cortex-A cores |

---

## 4. Package Contents

```text
HEY_NEXA_RELEASE/
|-- models/
|   |-- onnx/
|   |   |-- hey_nexa.onnx              # Trained custom wake classifier
|   |   |-- melspectrogram.onnx        # Base feature extractor
|   |   |-- embedding_model.onnx       # Base embedding network
|   |-- tflite/
|       |-- hey_nexa.tflite            # Exported TFLite classifier
|       |-- melspectrogram.tflite      # Base TFLite feature extractor
|       |-- embedding_model.tflite     # Base TFLite embedding network
|-- flutter/
|   |-- lib/
|   |   |-- services/
|   |   |   |-- nexa_wake_word_service.dart
|   |   |-- wake_word_demo.dart
|   |-- pubspec_changes.yaml
|-- android_changes.md
|-- integration_notes.md
|-- environment_report.md
|-- LICENSES_AND_DATA_PROVENANCE.md
|-- evaluation_report.md
|-- evaluation.json
|-- model_manifest.json
|-- SHA256SUMS.txt
|-- README_START_HERE.md
```

---

## 5. Licensing & Provenance (Option B Clean Path)
- **Trained Under:** Option B — Fast Training with Clean Data Provenance.
- **Speech Synthesis:** Permissively licensed Piper neural voices (LibriTTS-R, CC-BY-4.0).
- **Background Noise & Room Acoustics:** Clean public domain acoustic noise colors and MIT Room Impulse Responses.
- **Commercial Feasibility:** This model avoids the non-commercial ACAV100M / YouTube scraped audio dataset, making it safe for commercial product development and distribution under Apache 2.0 / MIT licensing.

