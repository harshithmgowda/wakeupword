# How to Use "hey_nexa_flutter_dropin" in Your Flutter App

This folder is designed to be a direct **drop-in** package into your existing Flutter application.

---

## Step 1: Copy to Your App Root
Copy the contents of this folder directly into your Flutter app project root:
- The `assets/models/` folder will merge into your app's `assets/models/`
- The `lib/services/nexa_wake_word_service.dart` will go into your app's `lib/services/`
- The `lib/screens/wake_word_demo.dart` will go into your app's `lib/screens/`

---

## Step 2: Update `pubspec.yaml`
Add these lines to your Flutter app's `pubspec.yaml`:

```yaml
dependencies:
  flutter:
    sdk: flutter
  
  # Wake word detection dependencies
  open_wake_word: ^0.2.0
  record: ^5.1.2
  permission_handler: ^11.3.1

flutter:
  assets:
    - assets/models/melspectrogram.onnx
    - assets/models/embedding_model.onnx
    - assets/models/hey_nexa.onnx
```

Then run in your terminal:
```bash
flutter pub get
```

---

## Step 3: Update Android Settings
1. In `android/app/src/main/AndroidManifest.xml`, add:
   ```xml
   <uses-permission android:name="android.permission.RECORD_AUDIO" />
   <uses-permission android:name="android.permission.MODIFY_AUDIO_SETTINGS" />
   ```

2. In `android/app/build.gradle`:
   - Set `minSdkVersion 24`
   - Add inside `android { ... }`:
     ```groovy
     aaptOptions {
         noCompress 'onnx'
     }
     ```

---

## Step 4: Let Your AI Agent Wire the Code!
Open `AI_AGENT_PROMPT.txt`, copy its text, and paste it to the AI assistant working inside your Flutter app codebase. It will automatically wire the wake word callback to your assistant UI with zero microphone conflicts!

