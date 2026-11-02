import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:flutter_foreground_task/flutter_foreground_task.dart';
import 'package:open_wake_word/open_wake_word.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'wake_detector.dart';

/// Production-grade custom wake word detector for "Hey Nexa".
/// Supports both in-app listening and background listening (opening the app like "Hey Google").
class OpenWakeWordDetector implements WakeDetector {
  final AudioRecorder _recorder = AudioRecorder();
  StreamSubscription<Uint8List>? _audioSubscription;
  bool _active = false;
  bool _disposed = false;
  int _generation = 0;
  bool _initialized = false;
  
  // Audio chunk buffer: 1280 samples = 2560 bytes = 80ms at 16kHz 16-bit PCM mono
  static const int chunkSize = 1280;
  final List<int> _byteBuffer = [];
  
  // Detection calibration
  static const double _detectionThreshold = 0.40;
  DateTime? _lastTriggerTime;
  static const Duration _cooldownDuration = Duration(milliseconds: 2000);

  static Future<void> initForegroundTask() async {
    FlutterForegroundTask.init(
      androidNotificationOptions: AndroidNotificationOptions(
        channelId: 'nexa_wake_channel',
        channelName: 'Nexa Voice Assistant',
        channelDescription: 'Listening for "Hey Nexa" wake word in background',
        channelImportance: NotificationChannelImportance.LOW,
        priority: NotificationPriority.LOW,
      ),
      iosNotificationOptions: const IOSNotificationOptions(
        showNotification: false,
        playSound: false,
      ),
      foregroundTaskOptions: ForegroundTaskOptions(
        eventAction: ForegroundTaskEventAction.nothing(),
        autoRunOnBoot: false,
        allowWakeLock: true,
        allowWifiLock: false,
      ),
    );
  }

  Future<void> _ensureModelFilesVerified() async {
    final docsDir = await getApplicationDocumentsDirectory();
    final modelFiles = [
      'assets/models/melspectrogram.onnx',
      'assets/models/embedding_model.onnx',
      'assets/models/hey_nexa.onnx',
    ];

    for (final assetPath in modelFiles) {
      final filename = assetPath.split('/').last;
      final targetFile = File('${docsDir.path}/$filename');
      final ByteData data = await rootBundle.load(assetPath);
      
      // If file does not exist or size does not match, extract fresh copy
      if (!await targetFile.exists() || await targetFile.length() != data.lengthInBytes) {
        await targetFile.writeAsBytes(
          data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
          flush: true,
        );
        print('[OpenWakeWordDetector] Extracted model: $filename (${data.lengthInBytes} bytes)');
      }
    }
  }

  Future<void> _ensureModelInitialized() async {
    if (_initialized) return;
    print('[OpenWakeWordDetector] Initializing openWakeWord models...');
    
    await _ensureModelFilesVerified();

    final success = await OpenWakeWord.init(
      melModelAssetPath: 'assets/models/melspectrogram.onnx',
      embModelAssetPath: 'assets/models/embedding_model.onnx',
      wwModelAssetPaths: ['assets/models/hey_nexa.onnx'],
    );
    if (!success) {
      throw StateError('Failed to initialize openWakeWord models for Hey Nexa');
    }
    _initialized = true;
    print('[OpenWakeWordDetector] openWakeWord engine initialized successfully!');
  }

  @override
  Future<void> start(void Function() onWake, void Function() onError) async {
    final generation = ++_generation;
    _active = false;
    await _release();

    if (_disposed || generation != _generation) return;

    try {
      if (!await _recorder.hasPermission()) {
        throw StateError('Microphone permission is required for wake word detection.');
      }

      await _ensureModelInitialized();
      if (_disposed || generation != _generation) return;

      // Start Android Foreground Task so the OS doesn't mute mic when app is in background
      if (Platform.isAndroid) {
        try {
          await initForegroundTask();
          if (!await FlutterForegroundTask.isRunningService) {
            await FlutterForegroundTask.startService(
              serviceTypes: [ForegroundServiceTypes.microphone],
              notificationTitle: 'Nexa Voice Assistant Active',
              notificationText: 'Listening for "Hey Nexa"...',
            );
          }
        } catch (e) {
          print('[OpenWakeWordDetector] Foreground service notice: $e');
        }
      }

      final audioStream = await _recorder.startStream(
        const RecordConfig(
          encoder: AudioEncoder.pcm16bits,
          sampleRate: 16000,
          numChannels: 1,
        ),
      );

      if (generation != _generation) {
        await _recorder.stop();
        return;
      }

      _active = true;
      _byteBuffer.clear();
      print('[OpenWakeWordDetector] Microphone audio stream started (16kHz PCM16 Mono). Listening...');

      _audioSubscription = audioStream.listen(
        (bytes) {
          if (!_active || generation != _generation) return;
          _processIncomingBytes(bytes, onWake);
        },
        onError: (err) {
          print('[OpenWakeWordDetector] Audio stream error: $err');
          if (!_active || generation != _generation) return;
          _active = false;
          onError();
          unawaited(stop());
        },
        onDone: () {
          if (!_active || generation != _generation) return;
          _active = false;
          onError();
          unawaited(stop());
        },
      );
    } catch (e) {
      print('[OpenWakeWordDetector] Start failed: $e');
      _active = false;
      await _release();
      onError();
      rethrow;
    }
  }

  void _processIncomingBytes(Uint8List bytes, void Function() onWake) {
    _byteBuffer.addAll(bytes);

    // Process in complete 1280-sample chunks (2560 bytes = 80ms)
    while (_byteBuffer.length >= chunkSize * 2) {
      final chunkBytes = Uint8List.fromList(_byteBuffer.sublist(0, chunkSize * 2));
      _byteBuffer.removeRange(0, chunkSize * 2);

      final byteData = ByteData.sublistView(chunkBytes);
      final int16List = Int16List(chunkSize);
      for (int i = 0; i < chunkSize; i++) {
        int16List[i] = byteData.getInt16(i * 2, Endian.little);
      }

      // Feed 80ms audio to openWakeWord C++ engine
      OpenWakeWord.processAudio(int16List);
      
      final prob = OpenWakeWord.getProbability();
      final isActivated = OpenWakeWord.isActivated();

      if (prob > 0.05 || isActivated) {
        print('[OpenWakeWordDetector] Score: ${prob.toStringAsFixed(3)}, Activated: $isActivated');
      }

      final now = DateTime.now();
      final inCooldown = _lastTriggerTime != null &&
          now.difference(_lastTriggerTime!) < _cooldownDuration;

      // Detection condition: either C++ engine triggered activation or score crossed threshold
      if ((isActivated || prob >= _detectionThreshold) && !inCooldown) {
        _lastTriggerTime = now;
        print('[OpenWakeWordDetector] *** "Hey Nexa" WAKE WORD DETECTED! *** (score: $prob)');
        
        // If app is in background or screen is off, wake the screen and bring app to front!
        if (Platform.isAndroid) {
          try {
            FlutterForegroundTask.wakeUpScreen();
            FlutterForegroundTask.launchApp();
          } catch (e) {
            print('[OpenWakeWordDetector] LaunchApp notice: $e');
          }
        }

        _active = false;
        onWake();
        break;
      }
    }
  }

  @override
  Future<void> stop() async {
    _generation++;
    _active = false;
    await _release();
  }

  Future<void> _release() async {
    await _audioSubscription?.cancel();
    _audioSubscription = null;
    try {
      await _recorder.stop();
    } catch (_) {}
  }

  @override
  Future<void> dispose() async {
    _disposed = true;
    _generation++;
    _active = false;
    await _release();
    try {
      await _recorder.dispose();
    } catch (_) {}
    if (Platform.isAndroid) {
      try {
        await FlutterForegroundTask.stopService();
      } catch (_) {}
    }
    if (_initialized) {
      OpenWakeWord.destroy();
      _initialized = false;
    }
  }
}
