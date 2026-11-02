import 'dart:async';
import 'package:flutter/material.dart';
import 'services/nexa_wake_word_service.dart';

/// Interactive Demo Page for the Nexaura "Hey Nexa" Wake-Word Engine.
class NexaWakeWordDemoScreen extends StatefulWidget {
  const NexaWakeWordDemoScreen({Key? key}) : super(key: key);

  @override
  State<NexaWakeWordDemoScreen> createState() => _NexaWakeWordDemoScreenState();
}

class _NexaWakeWordDemoScreenState extends State<NexaWakeWordDemoScreen> {
  final NexaWakeWordService _wakeService = NexaWakeWordService();

  // Subscriptions
  StreamSubscription<NexaWakeWordState>? _stateSub;
  StreamSubscription<double>? _scoreSub;
  StreamSubscription<DateTime>? _detectionSub;

  // UI state
  NexaWakeWordState _state = NexaWakeWordState.uninitialized;
  double _currentScore = 0.0;
  double _threshold = 0.60;
  int _cooldownMs = 2000;
  int _detectionCount = 0;
  DateTime? _lastDetectedTime;
  final List<String> _detectionLogs = [];

  // Simulated assistant response timer
  Timer? _assistantResponseTimer;

  @override
  void initState() {
    super.initState();
    _threshold = _wakeService.threshold;

    _stateSub = _wakeService.stateStream.listen((state) {
      if (mounted) setState(() => _state = state);
    });

    _scoreSub = _wakeService.scoreStream.listen((score) {
      if (mounted) setState(() => _currentScore = score);
    });

    _detectionSub = _wakeService.detectionStream.listen((timestamp) {
      if (mounted) {
        setState(() {
          _detectionCount++;
          _lastDetectedTime = timestamp;
          final timeStr = "${timestamp.hour.toString().padLeft(2, '0')}:${timestamp.minute.toString().padLeft(2, '0')}:${timestamp.second.toString().padLeft(2, '0')}";
          _detectionLogs.insert(0, "Detected \"Hey Nexa\" at $timeStr");
          if (_detectionLogs.length > 20) _detectionLogs.removeLast();
        });

        // Trigger assistant response simulation
        _simulateAssistantSpeech();
      }
    });

    // Auto-initialize on screen open
    _wakeService.initialize(threshold: _threshold, cooldownMs: _cooldownMs);
  }

  @override
  void dispose() {
    _assistantResponseTimer?.cancel();
    _stateSub?.cancel();
    _scoreSub?.cancel();
    _detectionSub?.cancel();
    _wakeService.stopListening();
    super.dispose();
  }

  /// Demonstrates mutual exclusion: while Nexa is speaking, wake-word detection is paused.
  void _simulateAssistantSpeech() {
    _wakeService.setAssistantActive(true);
    _assistantResponseTimer?.cancel();
    _assistantResponseTimer = Timer(const Duration(seconds: 4), () {
      if (mounted) {
        _wakeService.setAssistantActive(false);
      }
    });
  }

  Color _getStateColor() {
    switch (_state) {
      case NexaWakeWordState.listening:
        return Colors.greenAccent;
      case NexaWakeWordState.detected:
        return Colors.amberAccent;
      case NexaWakeWordState.cooldown:
        return Colors.orangeAccent;
      case NexaWakeWordState.assistantActive:
        return Colors.purpleAccent;
      case NexaWakeWordState.ready:
        return Colors.blueAccent;
      case NexaWakeWordState.error:
        return Colors.redAccent;
      default:
        return Colors.grey;
    }
  }

  String _getStateLabel() {
    switch (_state) {
      case NexaWakeWordState.uninitialized:
        return "Uninitialized";
      case NexaWakeWordState.initializing:
        return "Initializing Engine...";
      case NexaWakeWordState.ready:
        return "Ready";
      case NexaWakeWordState.listening:
        return "Listening for \"Hey Nexa\"";
      case NexaWakeWordState.detected:
        return "Wake-Word Detected!";
      case NexaWakeWordState.cooldown:
        return "Cooldown Active";
      case NexaWakeWordState.assistantActive:
        return "Assistant Active (Detection Paused)";
      case NexaWakeWordState.error:
        return "Engine Error";
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      backgroundColor: const Color(0xFF0F111A),
      appBar: AppBar(
        title: const Text('Nexa Offline Wake-Word'),
        backgroundColor: const Color(0xFF161926),
        elevation: 0,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            tooltip: 'Reset Engine',
            onPressed: () => _wakeService.initialize(threshold: _threshold, cooldownMs: _cooldownMs),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Status Card
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: const Color(0xFF1E2235),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: _getStateColor().withOpacity(0.4), width: 1.5),
              ),
              child: Column(
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Container(
                        width: 12,
                        height: 12,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: _getStateColor(),
                          boxShadow: [
                            BoxShadow(
                              color: _getStateColor().withOpacity(0.6),
                              blurRadius: 8,
                              spreadRadius: 2,
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(width: 10),
                      Text(
                        _getStateLabel(),
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color: _getStateColor(),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'Say: "Hey Nexa"',
                    style: theme.textTheme.headlineSmall?.copyWith(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 0.5,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Local offline classification (zero network latency)',
                    style: TextStyle(color: Colors.white.withOpacity(0.6), fontSize: 13),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),

            // Live Activation Probability Gauge
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: const Color(0xFF1E2235),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text(
                        'Activation Score',
                        style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600),
                      ),
                      Text(
                        '${(_currentScore * 100).toStringAsFixed(1)}%',
                        style: TextStyle(
                          color: _currentScore >= _threshold ? Colors.amberAccent : Colors.white70,
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Stack(
                    children: [
                      ClipRRect(
                        borderRadius: BorderRadius.circular(8),
                        child: LinearProgressIndicator(
                          value: _currentScore,
                          minHeight: 20,
                          backgroundColor: Colors.white10,
                          valueColor: AlwaysStoppedAnimation<Color>(
                            _currentScore >= _threshold ? Colors.amberAccent : Colors.indigoAccent,
                          ),
                        ),
                      ),
                      // Threshold Marker
                      Positioned(
                        left: MediaQuery.of(context).size.width * 0.8 * _threshold,
                        top: 0,
                        bottom: 0,
                        child: Container(
                          width: 2,
                          color: Colors.redAccent,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text('0%', style: TextStyle(color: Colors.white38, fontSize: 11)),
                      Text('Trigger Threshold: ${(_threshold * 100).toInt()}%',
                          style: const TextStyle(color: Colors.redAccent, fontSize: 11, fontWeight: FontWeight.bold)),
                      Text('100%', style: TextStyle(color: Colors.white38, fontSize: 11)),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),

            // Sensitivity and Cooldown Controls
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: const Color(0xFF1E2235),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Sensitivity Threshold: ${_threshold.toStringAsFixed(2)}',
                    style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600),
                  ),
                  Slider(
                    value: _threshold,
                    min: 0.30,
                    max: 0.95,
                    divisions: 13,
                    activeColor: Colors.indigoAccent,
                    onChanged: (val) {
                      setState(() => _threshold = val);
                      _wakeService.setThreshold(val);
                    },
                  ),
                  const SizedBox(height: 10),
                  Text(
                    'Cooldown Duration: ${_cooldownMs}ms',
                    style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600),
                  ),
                  Slider(
                    value: _cooldownMs.toDouble(),
                    min: 500,
                    max: 5000,
                    divisions: 9,
                    activeColor: Colors.indigoAccent,
                    onChanged: (val) {
                      setState(() => _cooldownMs = val.toInt());
                      _wakeService.setCooldown(val.toInt());
                    },
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),

            // Action Buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: _state == NexaWakeWordState.listening ? Colors.redAccent : Colors.indigoAccent,
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                    ),
                    icon: Icon(_state == NexaWakeWordState.listening ? Icons.mic_off : Icons.mic),
                    label: Text(
                      _state == NexaWakeWordState.listening ? 'Stop Listening' : 'Start Listening',
                      style: const TextStyle(fontWeight: FontWeight.bold),
                    ),
                    onPressed: () {
                      if (_state == NexaWakeWordState.listening) {
                        _wakeService.stopListening();
                      } else {
                        _wakeService.startListening();
                      }
                    },
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: OutlinedButton.icon(
                    style: OutlinedButton.styleFrom(
                      foregroundColor: Colors.purpleAccent,
                      side: const BorderSide(color: Colors.purpleAccent),
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                    ),
                    icon: const Icon(Icons.record_voice_over),
                    label: const Text('Simulate Nexa', style: TextStyle(fontWeight: FontWeight.bold)),
                    onPressed: _simulateAssistantSpeech,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),

            // Metrics Summary
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: const Color(0xFF1E2235),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text('Total Detections:', style: TextStyle(color: Colors.white70)),
                      Text('$_detectionCount',
                          style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16)),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text('Last Triggered:', style: TextStyle(color: Colors.white70)),
                      Text(
                        _lastDetectedTime != null
                            ? "${_lastDetectedTime!.hour}:${_lastDetectedTime!.minute}:${_lastDetectedTime!.second}"
                            : 'None',
                        style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),

            // Detection History Log
            if (_detectionLogs.isNotEmpty) ...[
              const Text(
                'Detection Event Log',
                style: TextStyle(color: Colors.white70, fontWeight: FontWeight.bold, fontSize: 14),
              ),
              const SizedBox(height: 8),
              Container(
                height: 140,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: const Color(0xFF161926),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: ListView.builder(
                  itemCount: _detectionLogs.length,
                  itemBuilder: (context, idx) {
                    return Padding(
                      padding: const EdgeInsets.symmetric(vertical: 4),
                      child: Text(
                        _detectionLogs[idx],
                        style: const TextStyle(color: Colors.amberAccent, fontSize: 13),
                      ),
                    );
                  },
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

