import 'dart:async';
import 'dart:io';
import 'wake_detector.dart';
import 'open_wake_word_detector.dart';
import 'package:flutter/widgets.dart';
import 'package:speech_to_text/speech_to_text.dart';

/// Foreground wake detection and explicit command dictation.
/// Passive wake monitoring is distinct from recording a command.
class NexaWake extends ChangeNotifier with WidgetsBindingObserver {
  final void Function(String) onDraft;
  final Function onAwake;
  final void Function(String)? onCommand;
  final WakeDetector? detector;

  NexaWake({
    required this.onDraft,
    required this.onAwake,
    SpeechToText? recognizer,
    this.onCommand,
    WakeDetector? wakeDetector,
  })  : detector = wakeDetector ??
            (recognizer == null && Platform.isAndroid ? OpenWakeWordDetector() : null),
        speech = recognizer ?? SpeechToText() {
    WidgetsBinding.instance.addObserver(this);
  }

  void _triggerAwake({bool fromBackground = false}) {
    if (onAwake is void Function({bool fromBackground})) {
      (onAwake as void Function({bool fromBackground}))(fromBackground: fromBackground);
    } else if (onAwake is void Function(bool)) {
      (onAwake as void Function(bool))(fromBackground);
    } else if (onAwake is VoidCallback) {
      (onAwake as VoidCallback)();
    } else {
      try {
        Function.apply(onAwake, [], {#fromBackground: fromBackground});
      } catch (_) {
        Function.apply(onAwake, []);
      }
    }
  }

  final SpeechToText speech;
  bool enabled = false, dictating = false, reviewing = false;
  bool _foreground = true,
      _suspended = false,
      _starting = false,
      _disposed = false;
  int _errors = 0;
  String status = 'Hey Nexa is off';
  String preferredLocale = 'en_IN';
  String? _locale;
  String _draft = '';
  bool _finishing = false;
  Timer? _restart;
  int _session = 0;
  bool _awaitingCommand = false;

  // Only the requested name, never unrelated assistant names or ordinary words.
  static final phrase =
      RegExp(r'^\s*(?:hey[\s,.!]+)?nexa\b[\s,.!?:-]*', caseSensitive: false);

  static String? commandAfterName(String words) {
    final match = phrase.firstMatch(words);
    return match == null ? null : words.substring(match.end).trim();
  }

  bool get listening =>
      dictating && speech.isListening && !_suspended && _foreground;

  void changed() {
    if (!_disposed) notifyListeners();
  }

  Future<void> start({bool direct = false, bool fromBackground = false}) async {
    if (!direct) enabled = true;
    dictating = direct;
    _awaitingCommand = false;
    _finishing = false;
    _draft = '';
    reviewing = false;
    _suspended = false;
    _errors = 0;
    if (!direct && detector != null) {
      await _listen();
      return;
    }

    speech.statusListener = (value) {
      if (_disposed) return;
      if (_suspended || !_foreground || reviewing) return;
      if (value == 'done') {
        if (dictating && _draft.isNotEmpty) {
          _finishDraft();
        } else {
          if (_awaitingCommand) {
            _awaitingCommand = false;
          } else {
            dictating = false;
          }
          _schedule();
        }
      }
      if (dictating || reviewing) changed();
    };

    speech.errorListener = (error) {
      if (_disposed || _suspended || !_foreground) return;
      final msg = error.errorMsg.toLowerCase();
      // Routine Android timeouts, no match, or transient client/busy conditions
      if (msg.contains('no_match') ||
          msg.contains('speech_timeout') ||
          msg.contains('busy') ||
          msg.contains('client')) {
        _errors = 0;
        if (dictating && _draft.isNotEmpty && !msg.contains('busy')) {
          _finishDraft();
        } else {
          if (msg.contains('no_match') || msg.contains('speech_timeout')) {
            dictating = false;
            status = 'No speech recognized. Tap the microphone to retry.';
          }
          _schedule();
        }
        changed();
        return;
      }
      _errors++;
      if (error.permanent || _errors >= 5) {
        enabled = false;
        dictating = false;
        status = 'Voice activation paused. Tap to retry.';
      } else {
        _schedule();
      }
      changed();
    };

    final ready = await speech.initialize(
      options: [SpeechToText.androidNoBluetooth],
      onStatus: speech.statusListener,
      onError: speech.errorListener,
      debugLogging: false,
    );

    if (!ready) {
      enabled = false;
      dictating = false;
      status = 'Device speech recognition unavailable';
      changed();
      return;
    }

    final locales = await speech.locales();
    _locale = preferredLocale == 'device' ? null : preferredLocale;
    bool found = false;
    for (final locale in locales) {
      if (locale.localeId.replaceAll('-', '_').toLowerCase() ==
          preferredLocale.replaceAll('-', '_').toLowerCase()) {
        _locale = locale.localeId;
        found = true;
        break;
      }
    }
    if (!found) _locale = null;

    if (_disposed) return;
    if (direct) _triggerAwake(fromBackground: fromBackground);
    await _listen();
  }

  Future<void> dictate({bool fromBackground = false}) async {
    await pause(); // Release the passive session before starting dictation.
    _suspended = false;
    _finishing = false;
    reviewing = false;
    _draft = '';
    _restart?.cancel();
    if (!_foreground) {
      // Allow Android OS 400ms to bring MainActivity window to foreground
      await Future.delayed(const Duration(milliseconds: 400));
    }
    await start(direct: true, fromBackground: fromBackground);
  }

  Future<void> finishDictation() async {
    if (_finishing) return;
    _finishing = true;
    await speech.stop();
    if (_draft.isNotEmpty) {
      status = reviewing ? status : 'Finishing your message…';
      _finishDraft();
    } else {
      _finishing = false;
      dictating = false;
      _restart?.cancel();
      status = enabled
          ? 'Getting ready for “Nexa”…'
          : 'Tap the microphone to try again';
      _schedule();
    }
    changed();
  }

  void _finishDraft() {
    if (reviewing || _disposed || _suspended) return;
    _finishing = false;
    dictating = false;
    reviewing = true;
    _errors = 0;
    _restart?.cancel();
    final command = normalizeCommand(_draft);
    if (command.isEmpty) {
      reviewing = false;
      _draft = '';
      _schedule();
      changed();
      return;
    }
    status = onCommand == null
        ? 'Review your message, then tap Send'
        : 'Sending your voice request…';
    if (command.isNotEmpty) onCommand?.call(command);
  }

  static String normalizeCommand(String text) =>
      (commandAfterName(text) ?? text).replaceAll(RegExp(r'\s+'), ' ').trim();

  void _schedule() {
    if (_restart?.isActive == true) return;
    if ((!enabled && !dictating) ||
        !_foreground ||
        _suspended ||
        reviewing ||
        _disposed) {
      return;
    }
    status = dictating
        ? 'Listening to your request…'
        : 'Waiting for “Hey Nexa” · no command active';
    _restart = Timer(const Duration(milliseconds: 450), () {
      if (_starting) {
        _schedule();
        return;
      }
      unawaited(_listen());
    });
  }

  Future<void> _listen() async {
    if (_starting ||
        (!enabled && !dictating) ||
        speech.isListening ||
        (!_foreground && detector == null) ||
        _suspended ||
        reviewing ||
        _disposed) {
      return;
    }
    _starting = true;
    final session = ++_session;
    var wakeInThisSession = false;
    try {
      if (!dictating && detector != null) {
        status = _foreground
            ? 'Ready · say “Hey Nexa”, pause, then speak'
            : 'Listening for “Hey Nexa” in background';
        changed();
        await detector!.start(() {
          if (_disposed || _suspended || !enabled) return;
          final triggeredFromBackground = !_foreground;
          unawaited(dictate(fromBackground: triggeredFromBackground));
        }, () {
          enabled = false;
          status =
              'Wake detection stopped. Check microphone permission and try again.';
          unawaited(detector!.stop());
          changed();
        });
        if (!_disposed && !_suspended && enabled) {
          status = _foreground
              ? 'Ready · say “Hey Nexa”, pause, then speak'
              : 'Listening for “Hey Nexa” in background';
        }
        return;
      }
      // A native end event does not clear the plugin's old pause timer.
      // Cancel it before rearming so it cannot cut off the next utterance.
      await speech.cancel();
      if (_disposed || _suspended || !_foreground || reviewing) return;
      status = dictating
          ? 'Listening to your request…'
          : 'Waiting for “Hey Nexa” · no command active';
      await speech.listen(
          listenOptions: SpeechListenOptions(
              localeId: _locale,
              listenFor: const Duration(seconds: 60),
              pauseFor: const Duration(seconds: 5),
              partialResults: true,
              cancelOnError: false,
              listenMode:
                  dictating ? ListenMode.dictation : ListenMode.confirmation),
          onResult: (result) {
            if (session != _session ||
                _disposed ||
                (!enabled && !dictating) ||
                _suspended ||
                !_foreground ||
                reviewing) {
              return;
            }
            var words = result.recognizedWords.trim();
            final command = commandAfterName(words);
            if (!dictating) {
              if (command == null || !result.finalResult) {
                // Passive listening: Ignore speech that does not contain the wake word.
                return;
              }
              _triggerAwake(fromBackground: false);
              _awaitingCommand = command.isEmpty;
              wakeInThisSession = true;
              dictating = true;
              words = command;
            } else if (wakeInThisSession && command != null) {
              words = command;
            }
            if (words.isNotEmpty) {
              _draft = words;
              onDraft(words);
            }
            status = 'Listening to your request…';
            if (result.finalResult && _draft.isNotEmpty) {
              _finishDraft();
              unawaited(speech.stop());
            }
            changed();
          });
    } catch (error) {
      enabled = false;
      dictating = false;
      status = error is StateError
          ? error.message
          : 'Offline wake detection could not start. Check microphone permission and restart Nexa.';
    } finally {
      _starting = false;
      changed();
    }
  }

  Future<void> stop() async {
    _session++;
    _finishing = false;
    enabled = false;
    dictating = false;
    reviewing = false;
    _restart?.cancel();
    await detector?.stop();
    await speech.cancel();
    status = 'Hey Nexa is off';
    changed();
  }

  Future<void> pause() async {
    _session++;
    _finishing = false;
    _suspended = true;
    dictating = false;
    status = enabled ? 'Voice activation paused' : 'Hey Nexa is off';
    _restart?.cancel();
    await detector?.stop();
    await speech.cancel();
    changed();
  }

  void resume() {
    if (_disposed) return;
    _suspended = false;
    reviewing = false;
    _schedule();
    changed();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    _foreground = state == AppLifecycleState.resumed;
    if (!_foreground) {
      _session++;
      if (dictating) {
        dictating = false;
        if (_draft.isNotEmpty) _finishDraft();
      }
      _restart?.cancel();
      unawaited(speech.cancel());
      // Keep background wake detector running so "Hey Nexa" can wake the app!
      if (detector == null) {
        status = 'Paused while app is in background';
      } else {
        status = 'Listening for “Hey Nexa” in background';
      }
    } else {
      _schedule();
    }
    changed();
  }

  @override
  void dispose() {
    _disposed = true;
    unawaited(detector?.dispose());
    _restart?.cancel();
    WidgetsBinding.instance.removeObserver(this);
    unawaited(speech.cancel());
    super.dispose();
  }
}
