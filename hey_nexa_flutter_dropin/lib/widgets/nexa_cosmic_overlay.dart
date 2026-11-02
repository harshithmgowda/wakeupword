import 'dart:math' as math;
import 'dart:ui';
import 'package:flutter/material.dart';
import '../services/nexa_wake.dart';

/// Gemini-style Cosmic Voice Assistant Overlay for "Hey Nexa".
///
/// Features:
/// - Signature cosmic starlight / golden aura glow along the bottom horizon
/// - Floating contextual action capsule ("Share screen with Live")
/// - Floating rounded obsidian pill bar with:
///   - "+" button on the left
///   - Live spoken speech transcript / "Ask Nexa"
///   - Coral/mauve circular microphone button with breathing aura
///   - Dynamic 4-bar sound waveform visualizer
class NexaCosmicOverlay extends StatefulWidget {
  final NexaWake wake;
  final TextEditingController messageController;
  final bool isBusy;
  final bool isSpeaking;
  final VoidCallback onSend;
  final VoidCallback? onCancel;

  const NexaCosmicOverlay({
    super.key,
    required this.wake,
    required this.messageController,
    required this.isBusy,
    required this.isSpeaking,
    required this.onSend,
    this.onCancel,
  });

  /// Displays the Cosmic Gemini-style overlay as a translucent bottom sheet
  static Future<void> show({
    required BuildContext context,
    required NexaWake wake,
    required TextEditingController messageController,
    required bool isBusy,
    required bool isSpeaking,
    required VoidCallback onSend,
    VoidCallback? onCancel,
  }) async {
    await showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      barrierColor: Colors.black.withValues(alpha: 0.55),
      elevation: 0,
      builder: (_) => NexaCosmicOverlay(
        wake: wake,
        messageController: messageController,
        isBusy: isBusy,
        isSpeaking: isSpeaking,
        onSend: onSend,
        onCancel: onCancel,
      ),
    );
  }

  @override
  State<NexaCosmicOverlay> createState() => _NexaCosmicOverlayState();
}

class _NexaCosmicOverlayState extends State<NexaCosmicOverlay>
    with TickerProviderStateMixin {
  late final AnimationController _shimmerController;
  late final AnimationController _pulseController;
  late final AnimationController _waveController;

  @override
  void initState() {
    super.initState();
    _shimmerController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 4),
    )..repeat();

    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1400),
    )..repeat(reverse: true);

    _waveController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 700),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _shimmerController.dispose();
    _pulseController.dispose();
    _waveController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final bottomPadding = MediaQuery.of(context).padding.bottom;

    return AnimatedBuilder(
      animation: Listenable.merge([widget.wake, widget.messageController, _shimmerController]),
      builder: (context, _) {
        final draftText = widget.messageController.text.trim();
        final isListening = widget.wake.listening || widget.wake.dictating;
        final isBusy = widget.isBusy;
        final isSpeaking = widget.isSpeaking;

        return Stack(
          alignment: Alignment.bottomCenter,
          clipBehavior: Clip.none,
          children: [
            // 1. Cosmic Starlight / Golden Glow Canvas at bottom horizon
            Positioned(
              left: 0,
              right: 0,
              bottom: 0,
              height: 240,
              child: IgnorePointer(
                child: CustomPaint(
                  painter: _CosmicHorizonPainter(
                    phase: _shimmerController.value * math.pi * 2,
                    isListening: isListening,
                    isBusy: isBusy,
                  ),
                ),
              ),
            ),

            // 2. Main Content Container
            Padding(
              padding: EdgeInsets.only(
                left: 16,
                right: 16,
                bottom: bottomPadding + 14,
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Upper Floating Capsule: "Share screen with Live"
                  Align(
                    alignment: const Alignment(-0.65, 0),
                    child: Container(
                      margin: const EdgeInsets.only(bottom: 12),
                      decoration: BoxDecoration(
                        color: const Color(0xFF141418).withValues(alpha: 0.90),
                        borderRadius: BorderRadius.circular(22),
                        border: Border.all(
                          color: Colors.white.withValues(alpha: 0.14),
                          width: 1,
                        ),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withValues(alpha: 0.35),
                            blurRadius: 12,
                            offset: const Offset(0, 4),
                          ),
                        ],
                      ),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(22),
                        child: BackdropFilter(
                          filter: ImageFilter.blur(sigmaX: 16, sigmaY: 16),
                          child: InkWell(
                            onTap: () {
                              // Screen contextual action or live switch
                              ScaffoldMessenger.maybeOf(context)?.showSnackBar(
                                const SnackBar(
                                  content: Text('Live Screen mode active for Nexa'),
                                  duration: Duration(seconds: 2),
                                ),
                              );
                            },
                            borderRadius: BorderRadius.circular(22),
                            child: const Padding(
                              padding: EdgeInsets.symmetric(
                                horizontal: 14,
                                vertical: 9,
                              ),
                              child: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  Icon(
                                    Icons.ios_share_rounded,
                                    color: Colors.white,
                                    size: 15,
                                  ),
                                  SizedBox(width: 8),
                                  Text(
                                    'Share screen with Live',
                                    style: TextStyle(
                                      color: Colors.white,
                                      fontSize: 13,
                                      fontWeight: FontWeight.w500,
                                      letterSpacing: -0.2,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),

                  // Gemini-Style Floating Pill Bar
                  Container(
                    height: 64,
                    decoration: BoxDecoration(
                      color: const Color(0xFF18181D).withValues(alpha: 0.92),
                      borderRadius: BorderRadius.circular(34),
                      border: Border.all(
                        color: Colors.white.withValues(alpha: 0.15),
                        width: 1,
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withValues(alpha: 0.45),
                          blurRadius: 20,
                          offset: const Offset(0, 6),
                        ),
                        if (isListening)
                          BoxShadow(
                            color: const Color(0xFFE5949C).withValues(alpha: 0.20),
                            blurRadius: 24,
                            spreadRadius: 2,
                            offset: const Offset(0, 2),
                          ),
                      ],
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(34),
                      child: BackdropFilter(
                        filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
                        child: Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 10),
                          child: Row(
                            children: [
                              // Left Plus Icon
                              IconButton(
                                onPressed: () {
                                  // Quick add attachment or prompt
                                },
                                icon: const Icon(
                                  Icons.add_rounded,
                                  color: Colors.white,
                                  size: 26,
                                ),
                                splashRadius: 22,
                              ),
                              const SizedBox(width: 4),

                              // Text Prompt / Spoken Transcription
                              Expanded(
                                child: Text(
                                  draftText.isNotEmpty
                                      ? draftText
                                      : isBusy
                                          ? 'Nexa is thinking…'
                                          : isSpeaking
                                              ? 'Nexa is speaking…'
                                              : isListening
                                                  ? 'Ask Nexa…'
                                                  : 'Say "Hey Nexa" or tap mic',
                                  style: TextStyle(
                                    color: draftText.isNotEmpty
                                        ? Colors.white
                                        : Colors.white.withValues(alpha: 0.70),
                                    fontSize: 16,
                                    fontWeight: draftText.isNotEmpty
                                        ? FontWeight.w500
                                        : FontWeight.w400,
                                    letterSpacing: -0.2,
                                  ),
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                ),
                              ),
                              const SizedBox(width: 8),

                              // Circular Coral/Mauve Mic Button
                              GestureDetector(
                                onTap: () {
                                  if (isListening) {
                                    widget.wake.finishDictation();
                                  } else {
                                    widget.wake.dictate();
                                  }
                                },
                                child: AnimatedBuilder(
                                  animation: _pulseController,
                                  builder: (context, _) {
                                    final pulse = isListening
                                        ? _pulseController.value * 3.0
                                        : 0.0;
                                    return Container(
                                      width: 46,
                                      height: 46,
                                      decoration: BoxDecoration(
                                        shape: BoxShape.circle,
                                        color: const Color(0xFF8C5E64),
                                        boxShadow: [
                                          if (isListening)
                                            BoxShadow(
                                              color: const Color(0xFFE5949C)
                                                  .withValues(alpha: 0.50),
                                              blurRadius: 10 + pulse,
                                              spreadRadius: pulse,
                                            ),
                                        ],
                                      ),
                                      child: Center(
                                        child: Icon(
                                          isListening
                                              ? Icons.mic_rounded
                                              : Icons.mic_none_rounded,
                                          color: Colors.white,
                                          size: 23,
                                        ),
                                      ),
                                    );
                                  },
                                ),
                              ),
                              const SizedBox(width: 10),

                              // Sound Waveform Bars (|||) or Send Action
                              GestureDetector(
                                onTap: () {
                                  if (draftText.isNotEmpty) {
                                    widget.onSend();
                                  }
                                },
                                child: Container(
                                  padding: const EdgeInsets.symmetric(
                                    horizontal: 6,
                                    vertical: 8,
                                  ),
                                  child: draftText.isNotEmpty
                                      ? Container(
                                          width: 34,
                                          height: 34,
                                          decoration: const BoxDecoration(
                                            shape: BoxShape.circle,
                                            color: Color(0xFF6C5CE7),
                                          ),
                                          child: const Center(
                                            child: Icon(
                                              Icons.arrow_upward_rounded,
                                              color: Colors.white,
                                              size: 20,
                                            ),
                                          ),
                                        )
                                      : _GeminiWaveformBars(
                                          isListening: isListening,
                                          waveController: _waveController,
                                        ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        );
      },
    );
  }
}

/// 4 animated sound wave bars matching the Gemini pill design
class _GeminiWaveformBars extends StatelessWidget {
  final bool isListening;
  final AnimationController waveController;

  const _GeminiWaveformBars({
    required this.isListening,
    required this.waveController,
  });

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: waveController,
      builder: (context, _) {
        final t = waveController.value;
        // Dynamically modulate bar heights
        final h1 = isListening ? 8.0 + 10.0 * math.sin(t * math.pi) : 8.0;
        final h2 = isListening ? 14.0 + 8.0 * math.cos(t * math.pi) : 14.0;
        final h3 = isListening ? 10.0 + 12.0 * math.sin((t + 0.3) * math.pi) : 10.0;
        final h4 = isListening ? 6.0 + 8.0 * math.cos((t + 0.5) * math.pi) : 6.0;

        return Row(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            _bar(h1),
            const SizedBox(width: 2.5),
            _bar(h2),
            const SizedBox(width: 2.5),
            _bar(h3),
            const SizedBox(width: 2.5),
            _bar(h4),
          ],
        );
      },
    );
  }

  Widget _bar(double height) {
    return Container(
      width: 2.5,
      height: height.clamp(4.0, 24.0),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: isListening ? 0.95 : 0.45),
        borderRadius: BorderRadius.circular(2),
      ),
    );
  }
}

/// Custom painter for the cosmic starlight aura and warm golden particle wave at the bottom
class _CosmicHorizonPainter extends CustomPainter {
  final double phase;
  final bool isListening;
  final bool isBusy;

  _CosmicHorizonPainter({
    required this.phase,
    required this.isListening,
    required this.isBusy,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final w = size.width;
    final h = size.height;

    // Base cosmic horizon glow (warm amber / gold / subtle stardust violet)
    final glowRect = Rect.fromLTWH(0, h * 0.35, w, h * 0.65);
    final glowPaint = Paint()
      ..shader = RadialGradient(
        center: const Alignment(0.0, 1.1),
        radius: 1.0,
        colors: [
          const Color(0xFFE2A165).withValues(alpha: isListening ? 0.35 : 0.22),
          const Color(0xFFC7778F).withValues(alpha: isListening ? 0.25 : 0.15),
          const Color(0xFF6E56CF).withValues(alpha: isListening ? 0.15 : 0.08),
          Colors.transparent,
        ],
        stops: const [0.0, 0.45, 0.75, 1.0],
      ).createShader(glowRect);

    canvas.drawRect(glowRect, glowPaint);

    // Glowing starlight dots / particles scattered across the bottom
    final random = math.Random(42); // fixed seed for stationary cosmic stars
    final starPaint = Paint()..style = PaintingStyle.fill;

    for (int i = 0; i < 48; i++) {
      final sx = random.nextDouble() * w;
      // Stars concentrate near the bottom curve
      final sy = h * 0.55 + random.nextDouble() * (h * 0.42);
      final radius = 0.8 + random.nextDouble() * 1.5;

      // Twinkle effect
      final twinkle = 0.4 + 0.6 * math.sin(phase + i * 0.8).abs();
      final alpha = (random.nextDouble() * 0.5 + 0.3) * twinkle;

      starPaint.color = i % 3 == 0
          ? const Color(0xFFFFD494).withValues(alpha: alpha)
          : i % 3 == 1
              ? const Color(0xFFFFA6B9).withValues(alpha: alpha)
              : Colors.white.withValues(alpha: alpha);

      canvas.drawCircle(Offset(sx, sy), radius, starPaint);
    }

    // Curved horizon light beam
    final beamPath = Path();
    beamPath.moveTo(0, h);
    for (int i = 0; i <= 60; i++) {
      final x = (i / 60) * w;
      final wave = math.sin((x / w) * math.pi + phase * 0.5) * 8.0;
      final y = h * 0.68 - wave;
      beamPath.lineTo(x, y);
    }
    beamPath.lineTo(w, h);
    beamPath.close();

    final beamPaint = Paint()
      ..shader = LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [
          const Color(0xFFFFCC80).withValues(alpha: isListening ? 0.12 : 0.06),
          Colors.transparent,
        ],
      ).createShader(Rect.fromLTWH(0, h * 0.60, w, h * 0.40));

    canvas.drawPath(beamPath, beamPaint);
  }

  @override
  bool shouldRepaint(covariant _CosmicHorizonPainter oldDelegate) {
    return oldDelegate.phase != phase ||
        oldDelegate.isListening != isListening ||
        oldDelegate.isBusy != isBusy;
  }
}
