import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_foreground_task/flutter_foreground_task.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../theme/aura_theme.dart';

/// Interactive modal sheet that helps the user grant all permissions
/// required for "Hey Nexa" background wake-word detection (like Google Assistant).
class NexaPermissionSheet extends StatefulWidget {
  const NexaPermissionSheet({super.key});

  /// Automatically displays the sheet if any critical permissions are missing
  static Future<void> showIfRequired(BuildContext context, {bool force = false}) async {
    if (!Platform.isAndroid) return;

    final prefs = await SharedPreferences.getInstance();
    final alreadyPrompted = prefs.getBool('nexa_permissions_prompted') ?? false;

    final micGranted = await Permission.microphone.isGranted;
    final overlayGranted = await FlutterForegroundTask.canDrawOverlays;
    final batteryIgnored = (await FlutterForegroundTask.isIgnoringBatteryOptimizations) ||
        (await Permission.ignoreBatteryOptimizations.isGranted);

    // Show if forced or if critical permissions are missing and not yet dismissed
    if (force || (!alreadyPrompted && (!micGranted || !overlayGranted || !batteryIgnored))) {
      if (!context.mounted) return;
      await showModalBottomSheet(
        context: context,
        isScrollControlled: true,
        backgroundColor: Colors.transparent,
        builder: (_) => const NexaPermissionSheet(),
      );
      await prefs.setBool('nexa_permissions_prompted', true);
    }
  }

  @override
  State<NexaPermissionSheet> createState() => _NexaPermissionSheetState();
}

class _NexaPermissionSheetState extends State<NexaPermissionSheet>
    with WidgetsBindingObserver {
  bool _micGranted = false;
  bool _overlayGranted = false;
  bool _batteryIgnored = false;
  bool _notificationGranted = false;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _checkPermissions();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // When returning from Android Settings, re-check permissions immediately
    if (state == AppLifecycleState.resumed) {
      _checkPermissions();
    }
  }

  Future<void> _checkPermissions() async {
    final mic = await Permission.microphone.isGranted;
    final overlay = await FlutterForegroundTask.canDrawOverlays;
    final battery = (await FlutterForegroundTask.isIgnoringBatteryOptimizations) ||
        (await Permission.ignoreBatteryOptimizations.isGranted);
    final notif = await Permission.notification.isGranted;

    if (mounted) {
      setState(() {
        _micGranted = mic;
        _overlayGranted = overlay;
        _batteryIgnored = battery;
        _notificationGranted = notif;
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final bgColor = isDark ? AuraTheme.bgSecondaryDark : Colors.white;
    final textColor = isDark ? Colors.white : AuraTheme.textPrimary;
    final subtextColor = isDark ? const Color(0xFF94A3B8) : AuraTheme.textSecondary;
    final allGranted = _micGranted && _overlayGranted && _batteryIgnored;

    return Container(
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.25),
            blurRadius: 20,
            offset: const Offset(0, -4),
          ),
        ],
      ),
      padding: EdgeInsets.only(
        top: 20,
        left: 24,
        right: 24,
        bottom: MediaQuery.of(context).padding.bottom + 24,
      ),
      child: _loading
          ? const Center(child: Padding(padding: EdgeInsets.all(32), child: CircularProgressIndicator()))
          : Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Top handle
                Center(
                  child: Container(
                    width: 44,
                    height: 4,
                    margin: const EdgeInsets.only(bottom: 20),
                    decoration: BoxDecoration(
                      color: isDark ? const Color(0xFF334155) : const Color(0xFFE2E8F0),
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),
                ),

                // Header
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: AuraTheme.accentCyan.withValues(alpha: 0.15),
                        borderRadius: BorderRadius.circular(14),
                      ),
                      child: const Icon(Icons.mic, color: AuraTheme.accentCyan, size: 26),
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Enable "Hey Nexa"',
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.w700,
                              color: textColor,
                            ),
                          ),
                          const SizedBox(height: 2),
                          Text(
                            'Open Nexa like Google Assistant anywhere',
                            style: TextStyle(
                              fontSize: 13,
                              color: subtextColor,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 20),

                // 1. Microphone Permission
                _PermissionTile(
                  icon: Icons.mic_rounded,
                  title: 'Microphone Access',
                  description: 'Required to hear "Hey Nexa" in real-time.',
                  isGranted: _micGranted,
                  onTap: () async {
                    final status = await Permission.microphone.request();
                    if (status.isPermanentlyDenied) {
                      await openAppSettings();
                    }
                    await _checkPermissions();
                  },
                ),
                const SizedBox(height: 12),

                // 2. Display Over Other Apps (System Alert Window)
                _PermissionTile(
                  icon: Icons.layers_outlined,
                  title: 'Display Over Other Apps',
                  description: 'Allows Nexa to pop open over any screen or app.',
                  isGranted: _overlayGranted,
                  actionText: 'Open Settings',
                  onTap: () async {
                    bool opened = false;
                    try {
                      opened = await FlutterForegroundTask.openSystemAlertWindowSettings();
                    } catch (_) {}
                    if (!opened) {
                      await openAppSettings();
                    }
                    await _checkPermissions();
                  },
                ),
                const SizedBox(height: 12),

                // 3. Ignore Battery Optimizations
                _PermissionTile(
                  icon: Icons.battery_charging_full_rounded,
                  title: 'Unrestricted Battery',
                  description: 'Prevents Android from putting the background listener to sleep.',
                  isGranted: _batteryIgnored,
                  actionText: 'Allow',
                  onTap: () async {
                    bool granted = false;
                    try {
                      // Attempt 1: Standard Android direct prompt
                      granted = await FlutterForegroundTask.requestIgnoreBatteryOptimization();
                    } catch (_) {}

                    if (!granted) {
                      try {
                        // Attempt 2: permission_handler request
                        final status = await Permission.ignoreBatteryOptimizations.request();
                        granted = status.isGranted;
                      } catch (_) {}
                    }

                    if (!granted) {
                      // Attempt 3: Open Battery Optimization Settings list
                      try {
                        granted = await FlutterForegroundTask.openIgnoreBatteryOptimizationSettings();
                      } catch (_) {}
                    }

                    if (!granted) {
                      // Attempt 4: Fallback to App Info page (User selects Battery -> Unrestricted)
                      try {
                        await openAppSettings();
                      } catch (_) {}
                    }

                    if (context.mounted) {
                      ScaffoldMessenger.maybeOf(context)?.showSnackBar(
                        const SnackBar(
                          content: Text('Please select "Unrestricted" or "Don\'t optimize" for AURA in Battery settings.'),
                          duration: Duration(seconds: 4),
                        ),
                      );
                    }

                    await _checkPermissions();
                  },
                ),
                const SizedBox(height: 12),

                // 4. Notification Permission
                _PermissionTile(
                  icon: Icons.notifications_active_outlined,
                  title: 'Foreground Notification',
                  description: 'Shows active listening status in notification bar.',
                  isGranted: _notificationGranted,
                  onTap: () async {
                    await Permission.notification.request();
                    await _checkPermissions();
                  },
                ),
                const SizedBox(height: 24),

                // Action Button
                ElevatedButton(
                  onPressed: () => Navigator.of(context).pop(),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: allGranted ? AuraTheme.accentEmerald : AuraTheme.accentCyan,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                    elevation: 0,
                  ),
                  child: Text(
                    allGranted ? 'All Set! Continue to Nexa' : 'Done for Now',
                    style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                  ),
                ),
              ],
            ),
    );
  }
}

class _PermissionTile extends StatelessWidget {
  final IconData icon;
  final String title;
  final String description;
  final bool isGranted;
  final String actionText;
  final VoidCallback onTap;

  const _PermissionTile({
    required this.icon,
    required this.title,
    required this.description,
    required this.isGranted,
    this.actionText = 'Grant',
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final cardBg = isDark ? AuraTheme.surfaceGlassDark : const Color(0xFFF8FAFC);
    final borderColor = isDark ? AuraTheme.surfaceBorderDark : const Color(0xFFE2E8F0);
    final textColor = isDark ? Colors.white : AuraTheme.textPrimary;
    final subtextColor = isDark ? const Color(0xFF94A3B8) : AuraTheme.textSecondary;

    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: isGranted ? null : onTap,
        borderRadius: BorderRadius.circular(16),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
          decoration: BoxDecoration(
            color: cardBg,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: isGranted ? AuraTheme.accentEmerald.withValues(alpha: 0.4) : borderColor,
              width: 1,
            ),
          ),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: isGranted
                      ? AuraTheme.accentEmerald.withValues(alpha: 0.15)
                      : AuraTheme.accentCyan.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Icon(
                  icon,
                  size: 20,
                  color: isGranted ? AuraTheme.accentEmerald : AuraTheme.accentCyan,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                        color: textColor,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      description,
                      style: TextStyle(fontSize: 11, color: subtextColor),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              if (isGranted)
                const Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.check_circle_rounded, color: AuraTheme.accentEmerald, size: 20),
                    SizedBox(width: 4),
                    Text(
                      'Enabled',
                      style: TextStyle(
                        color: AuraTheme.accentEmerald,
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                )
              else
                TextButton(
                  onPressed: onTap,
                  style: TextButton.styleFrom(
                    backgroundColor: AuraTheme.accentCyan.withValues(alpha: 0.12),
                    foregroundColor: AuraTheme.accentCyan,
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                    minimumSize: Size.zero,
                    tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                  ),
                  child: Text(
                    actionText,
                    style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w700),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

