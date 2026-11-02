# Android Configuration Guide for Nexaura "Hey Nexa" Wake-Word Engine

This document outlines the required Android-side changes in your Flutter project (`android/`) to run local offline wake-word inference.

---

## 1. Android Manifest Permissions (`android/app/src/main/AndroidManifest.xml`)

Add the audio recording permissions inside the `<manifest>` tag:

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android">

    <!-- Required for real-time microphone capture -->
    <uses-permission android:name="android.permission.RECORD_AUDIO" />

    <!-- System-level background listening & Google Assistant-like popup -->
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE_MICROPHONE" />
    <uses-permission android:name="android.permission.SYSTEM_ALERT_WINDOW" />
    <uses-permission android:name="android.permission.WAKE_LOCK" />
    <uses-permission android:name="android.permission.POST_NOTIFICATIONS" />
    <uses-permission android:name="android.permission.RECEIVE_BOOT_COMPLETED" />
    <uses-permission android:name="android.permission.REQUEST_IGNORE_BATTERY_OPTIMIZATIONS" />

    <application
        android:label="Nexaura"
        android:name="${applicationName}"
        android:icon="@mipmap/ic_launcher">
        
        <!-- Application activity and services -->
        
    </application>
</manifest>
```

---

## 2. Gradle Configuration (`android/app/build.gradle`)

Ensure `minSdkVersion` is at least **24** and prevent compression of model assets:

```groovy
android {
    compileSdkVersion 34

    defaultConfig {
        applicationId "com.example.nexaura"
        minSdkVersion 24        // ONNX Runtime Android requires 21+, 24 recommended
        targetSdkVersion 34
        versionCode flutterVersionCode.toInteger()
        versionName flutterVersionName

        ndk {
            // Include modern 64-bit and 32-bit ARM architectures
            abiFilters "arm64-v8a", "armeabi-v7a", "x86_64"
        }
    }

    aaptOptions {
        // Prevent Gradle from compressing model files so they can be memory-mapped directly
        noCompress 'onnx', 'tflite'
    }

    packagingOptions {
        pickFirst 'lib/**/libonnxruntime.so'
    }
}
```

---

## 3. ProGuard / R8 Rules (`android/app/proguard-rules.pro`)

If code shrinking/obfuscation is enabled (`minifyEnabled true`), ensure the ONNX Runtime native JNI bindings are preserved:

```proguard
# Preserve ONNX Runtime JNI bindings
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**

# If using TensorFlow Lite:
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**
```

---

## 4. Audio Focus & Hardware Guidelines
- **Sample Rate:** openWakeWord models are trained strictly on **16,000 Hz, 16-bit Mono PCM**. Do not downsample with low-order interpolation at runtime; use Flutter's `record` plugin configured for `sampleRate: 16000`, `numChannels: 1`.
- **Acoustic Echo Cancellation (AEC):** Android devices often apply hardware noise cancellation. For wake words, standard voice recognition audio source (`MediaRecorder.AudioSource.VOICE_RECOGNITION`) is optimal as it applies mild noise suppression without aggressive clipping.

