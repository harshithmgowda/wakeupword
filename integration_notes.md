# Nexaura "Hey Nexa" Wake-Word Integration Guide

This guide explains how to integrate the offline wake-word detector into the Nexaura Flutter application, connect it with Speech-to-Text (STT) and Text-to-Speech (TTS), and maintain strict mutual exclusion.

---

## 1. System Architecture & Separation of Concerns

```
                  +-----------------------------------+
                  |        Microphone (16kHz)         |
                  +-----------------------------------+
                                    |
                    PCM Chunks (1280 samples / 80ms)
                                    v
+------------------------------------------------------------------------+
|                     NexaWakeWordService (Offline)                      |
|                                                                        |
|  [State: LISTENING]                                                    |
|         |                                                              |
|         v                                                              |
|  [melspectrogram.onnx] -> [embedding_model.onnx] -> [hey_nexa.onnx]   |
|                                                              |         |
|                                                      Score >= 0.60?    |
|                                                              |         |
+--------------------------------------------------------------|---------+
                                                               v [WAKE DETECTED]
                                              +--------------------------------+
                                              | Nexaura Assistant State Machine|
                                              +--------------------------------+
                                                               |
                                            1. setAssistantActive(true)
                                            2. Pause Wake-Word Detection
                                            3. Play Wake Sound / Haptic
                                                               v
                                              +--------------------------------+
                                              |  STT (Speech-to-Text Engine)   |
                                              |   Listens to user command      |
                                              +--------------------------------+
                                                               v
                                              +--------------------------------+
                                              |    LLM / Assistant Backend     |
                                              |       Generates response       |
                                              +--------------------------------+
                                                               v
                                              +--------------------------------+
                                              |   TTS (Text-to-Speech Engine)  |
                                              |      Speaks back to user       |
                                              +--------------------------------+
                                                               |
                                            4. Assistant finishes speaking
                                            5. setAssistantActive(false)
                                            6. Resume Wake-Word Detection
```

---

## 2. Preventing Self-Triggering & Infinite Loops (Mutual Exclusion)

The most common failure in voice assistants is **acoustic feedback loop**:
- The assistant's own speaker plays audio containing words that sound like "Nexa" or background noise.
- The microphone hears the assistant's own speaker and wakes up again while responding.

### The Solution:
`NexaWakeWordService` implements a dedicated `setAssistantActive(bool active)` method:
1. When the assistant starts listening for the user's intent or speaks a response:
   ```dart
   nexaWakeService.setAssistantActive(true);
   ```
2. When the assistant completes speaking:
   ```dart
   nexaWakeService.setAssistantActive(false);
   ```

While `assistantActive` is `true`:
- The audio buffer is immediately cleared.
- No inference passes are executed.
- State transitions to `assistantActive`.
- Even if audio contains "Hey Nexa", the trigger is dropped.

---

## 3. Cooldown & Debounce

The service enforces a **2000 ms cooldown** after each detection:
- Immediately upon detection, state moves to `detected`, then `cooldown`.
- During cooldown, subsequent frames do not re-trigger.
- After 2000 ms, if the assistant is not active, listening automatically resumes.

---

## 4. App Lifecycle Integration (Background & Sleep)

In `AppLifecycleListener` or `WidgetsBindingObserver`:

```dart
@override
void didChangeAppLifecycleState(AppLifecycleState state) {
  if (state == AppLifecycleState.paused) {
    // App minimized or screen turned off
    // If foreground service is NOT enabled, pause listening to save battery:
    nexaWakeService.stopListening();
  } else if (state == AppLifecycleState.resumed) {
    // App returned to foreground
    nexaWakeService.startListening();
  }
}
```

---

## 5. Model Asset Checksums & Verification

Always verify model hashes before releasing updates:
- Check `SHA256SUMS.txt` against the deployed `.onnx` and `.tflite` files.
- The `melspectrogram` and `embedding_model` are universal openWakeWord feature extractors.
- The `hey_nexa` classifier is the custom trained neural network head (32-unit DNN).

