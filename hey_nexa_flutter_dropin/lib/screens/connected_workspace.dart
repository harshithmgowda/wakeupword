import '../services/reply_voice.dart';
import 'package:http/http.dart' as http;
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:web_socket_channel/io.dart';
import '../services/api_service.dart';
import '../services/endpoint_check.dart';
import '../config.dart';
import '../widgets/nexa_visuals.dart';
import '../services/nexa_wake.dart';
import '../widgets/nexa_permission_sheet.dart';
import '../widgets/nexa_cosmic_overlay.dart';
import '../theme/aura_theme.dart';
part 'nexa_workspace_ui.dart';
part 'nexa_reference_ui.dart';

const secureSession = FlutterSecureStorage();
GoogleSignInAccount? googleAccount;
bool googleInitialized = false;

Future<void> initializeGoogle(ApiService api) async {
  if (googleInitialized) return;
  final config = await api.request('/configuration');
  final client = config['google_client_id'] as String? ?? '';
  if (client.isEmpty) {
    throw Exception('Configure GOOGLE_CLIENT_ID on the backend first.');
  }
  await GoogleSignIn.instance.initialize(serverClientId: client);
  googleInitialized = true;
}

class SessionGate extends StatefulWidget {
  const SessionGate({super.key});
  @override
  State<SessionGate> createState() => _SessionGateState();
}

class _SessionGateState extends State<SessionGate> {
  void clearError() {
    if (mounted) setState(() => error = null);
  }

  final endpoint = TextEditingController(text: AuraConfig.apiUrl);
  final email = TextEditingController(), password = TextEditingController();
  bool busy = false;
  String? error;
  @override
  void initState() {
    super.initState();
    restore();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) NexaPermissionSheet.showIfRequired(context);
    });
  }

  @override
  void dispose() {
    endpoint.dispose();
    email.dispose();
    password.dispose();
    super.dispose();
  }

  Future<void> restore() async {
    try {
      final host = await secureSession.read(key: 'endpoint');
      final current = AuraConfig.preferredEndpoint(host);
      if (host != null && current != host) {
        // A company APK can replace an expired quick-tunnel URL. Authenticate
        // afresh instead of forwarding saved credentials to a changed origin.
        endpoint.text = current;
        await secureSession.write(key: 'endpoint', value: current);
        await secureSession.delete(key: 'access');
        await secureSession.delete(key: 'refresh');
        return;
      }
      final access = await secureSession.read(key: 'access');
      if (host == null || access == null || !mounted) return;
      endpoint.text = host;
      final api = ApiService(baseUrl: '$host/api/v1', token: access);
      api.refreshToken = await secureSession.read(key: 'refresh') ?? '';
      await api.request('/auth/me');
      if (mounted) open(api);
    } catch (_) {
      /* Expired or unreachable sessions require explicit sign-in. */
    }
  }

  void open(ApiService api) {
    Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => ConnectedWorkspace(api: api)));
  }

  Future<void> login({required bool google}) async {
    setState(() {
      busy = true;
      error = null;
    });
    try {
      final host = endpoint.text.trim().replaceAll(RegExp(r'/+$'), '');
      final uri = Uri.parse(host);
      if (!uri.hasAuthority || !['http', 'https'].contains(uri.scheme)) {
        throw Exception('Enter the full HTTPS backend URL.');
      }

      final api = ApiService(baseUrl: '$host/api/v1');
      dynamic result;
      if (google) {
        await initializeGoogle(api);
        googleAccount = await GoogleSignIn.instance.authenticate();
        final id = googleAccount!.authentication.idToken;
        if (id == null) {
          throw Exception(
              'Google did not return an ID token. Check the web client ID and Android SHA-1.');
        }
        result = await api.request('/auth/google/token',
            method: 'POST', body: {'id_token': id});
      } else {
        result = await api.request('/auth/login',
            method: 'POST',
            body: {'email': email.text.trim(), 'password': password.text});
      }
      api.token = result['access_token'];
      api.refreshToken = result['refresh_token'];
      await secureSession.write(key: 'endpoint', value: host);
      await secureSession.write(key: 'access', value: api.token);
      await secureSession.write(key: 'refresh', value: api.refreshToken);
      if (mounted) open(api);
    } catch (e) {
      if (mounted) {
        setState(() =>
            error = e.toString().replaceFirst(RegExp(r'^Exception: '), ''));
      }
    } finally {
      if (mounted) setState(() => busy = false);
    }
  }

  @override
  Widget build(BuildContext context) => nexaSignIn(context);
}

class ConnectedWorkspace extends StatefulWidget {
  final ApiService api;
  final bool connectLive;
  const ConnectedWorkspace(
      {super.key, required this.api, this.connectLive = true});
  @override
  State<ConnectedWorkspace> createState() => _ConnectedWorkspaceState();
}

class _ConnectedWorkspaceState extends State<ConnectedWorkspace>
    with WidgetsBindingObserver {
  late ApiService api;
  late final NexaWake wake;
  Timer? heartbeat;
  bool serverReachable = false;
  int connectionRetries = 0;
  bool checkingConnection = false;
  String busyLabel = 'Working…';
  void redraw(VoidCallback callback) {
    if (mounted) setState(callback);
  }

  List<dynamic> spaces = [],
      members = [],
      workflows = [],
      tasks = [],
      messages = [],
      notifications = [],
      negotiations = [],
      calendars = [];
  Map<String, dynamic> me = {}, cache = {}, voiceProfile = {}, ai = {};
  String workspace = '', panel = 'Assistant', connection = 'Disconnected';
  String meetingFilter = 'My meetings', calendarFilter = 'Timeline';
  String? error;
  List<dynamic> invitations = [];
  bool get needsVoice =>
      voiceProfile['voice_enabled'] == true &&
      voiceProfile['enrolled'] == false;
  bool get isAdmin => members.any((m) =>
      m['user_id'] == me['id'] && ['OWNER', 'ADMIN'].contains(m['role']));
  Timer? invitationPoll;
  DateTime? recordingStarted;
  bool checkingVoice = false;
  Map<String, dynamic>? voiceCheck;
  bool busy = false,
      recording = false,
      speakReplies = true,
      enrollmentMode = false,
      disposed = false;
  final message = TextEditingController();
  final recorder = AudioRecorder();
  final player = AudioPlayer();
  final tts = FlutterTts();
  final replyVoice = ReplyVoice();
  bool offlineReplyVoice = false;
  DateTime? naturalVoiceRetryAfter;
  bool speaking = false;
  bool voiceSheetOpen = false;
  double speechRate = .38;
  Completer<void>? speechStopped;
  bool wakeRemembered = false;
  bool voicePreferencesLoaded = false;
  final List<Uint8List> samples = [];
  String? audioPath;
  Timer? recordLimit, reconnect, poll;
  WebSocketChannel? channel;
  int sequence = 0;
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    api = widget.api;
    wake = NexaWake(
      onCommand: (text) {
        if (!mounted || busy || speaking) return;
        message.text = text;
        if (voiceSheetOpen) {
          voiceSheetOpen = false;
          Navigator.of(context).pop();
        }
        unawaited(send());
      },
      onDraft: (text) {
        if (mounted) {
          message.text = text;
          message.selection = TextSelection.collapsed(offset: text.length);
        }
      },
      onAwake: ({bool fromBackground = false}) {
        if (!mounted) return;
        setState(() => panel = 'Assistant');
        if (fromBackground) {
          // Said "Hey Nexa" without opening the app (from background / home screen):
          // Open the Cosmic Gemini overlay!
          Future.delayed(const Duration(milliseconds: 250), () {
            if (mounted) unawaited(openCosmicNexaModal());
          });
        } else {
          // Said "Hey Nexa" in opened mobile app:
          // Use the in-app mobile UI directly!
        }
      },
    );
    wake.addListener(() {
      if (mounted) setState(() {});
    });
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) NexaPermissionSheet.showIfRequired(context);
    });
    unawaited(action(initialize));
  }

  @override
  void dispose() {
    disposed = true;
    WidgetsBinding.instance.removeObserver(this);
    heartbeat?.cancel();
    wake.dispose();
    invitationPoll?.cancel();
    recordLimit?.cancel();
    reconnect?.cancel();
    poll?.cancel();
    channel?.sink.close();
    recorder.dispose();
    player.dispose();
    tts.stop();
    final stopped = speechStopped;
    if (stopped != null && !stopped.isCompleted) stopped.complete();
    message.dispose();
    deleteAudio();
    super.dispose();
  }

  Future<void> deleteAudio() async {
    final path = audioPath;
    audioPath = null;
    if (path != null) {
      final file = File(path);
      if (await file.exists()) await file.delete();
    }
  }

  Future<void> action(Future<void> Function() callback) async {
    if (mounted) {
      setState(() {
        busy = true;
        error = null;
      });
    }
    try {
      await callback();
    } catch (e) {
      if (mounted) setState(() => error = e.toString());
    } finally {
      if (mounted) {
        setState(() {
          busy = false;
          busyLabel = 'Working…';
        });
      }
    }
  }

  Future<void> initialize() async {
    if (!voicePreferencesLoaded) {
      // Remove credentials from the retired wake engine during migration.
      await secureSession.delete(key: 'picovoice_access_key');
      await secureSession.delete(key: 'picovoice_keyword_path');
      // Local preferences contain no recordings or recognition transcripts.
      final values = await Future.wait([
        secureSession.read(key: 'nexa_voice_rate'),
        secureSession.read(key: 'nexa_voice_locale'),
        secureSession.read(key: 'nexa_wake_enabled'),
        secureSession.read(key: 'nexa_reply_voice_offline'),
      ]);
      speechRate = (double.tryParse(values[0] ?? '') ?? .38).clamp(.25, .65);
      wake.preferredLocale = values[1] ?? 'en_IN';
      wakeRemembered = values[2] != 'false';
      offlineReplyVoice = values[3] == 'true';
      voicePreferencesLoaded = true;
    }
    invitationPoll?.cancel();
    me = Map<String, dynamic>.from(await api.request('/auth/me'));
    serverReachable = true;
    cache = Map<String, dynamic>.from(await api.request('/calendar/mine'));
    voiceProfile =
        Map<String, dynamic>.from(await api.request('/voice/profile'));
    ai = Map<String, dynamic>.from(await api.request('/ai-settings'));
    await refreshInvitations();
    await loadSpaces();
    if (mounted) {
      setState(() => panel = needsVoice
          ? 'Voice studio'
          : workspace.isEmpty
              ? 'People'
              : (cache['calendars'] as List? ?? []).isEmpty
                  ? 'Calendar'
                  : 'Assistant');
    }
    if (mounted && wakeRemembered && !wake.enabled) {
      await wake.start();
    }
    invitationPoll = Timer.periodic(const Duration(seconds: 15), (_) {
      if (!busy) {
        unawaited(refreshInvitations().catchError((Object e) {
          if (mounted) {
            setState(() => error =
                'Cannot refresh invitations. Check your connection and tap Refresh.');
          }
        }));
      }
    });
  }

  Future<void> refreshInvitations() async {
    final results = await Future.wait(
        [api.request('/invitations/pending'), api.request('/notifications')]);
    if (mounted) {
      setState(() {
        invitations = results[0];
        notifications = results[1];
      });
    }
  }

  Future<void> loadSpaces() async {
    final result = await api.request('/workspaces/') as List;
    if (!mounted) return;
    setState(() {
      spaces = result;
      if (!spaces.any((s) => s['id'] == workspace)) {
        workspace = spaces.isEmpty ? '' : spaces.first['id'];
      }
    });
    if (workspace.isNotEmpty) await selectWorkspace(workspace);
  }

  Future<void> selectWorkspace(String id) async {
    reconnect?.cancel();
    poll?.cancel();
    channel?.sink.close();
    sequence = 0;
    api = api.forWorkspace(id);
    setState(() {
      workspace = id;
      workflows = [];
      tasks = [];
      messages = [];
      members = [];
    });
    connect();
    await refresh();
    final enrollment =
        Map<String, dynamic>.from(await api.request('/voice/profile'));
    if (mounted) setState(() => voiceProfile = enrollment);
    poll = Timer.periodic(const Duration(seconds: 30), (_) {
      if (!busy) unawaited(action(refresh));
    });
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed &&
        workspace.isNotEmpty &&
        widget.connectLive) {
      reconnect?.cancel();
      connectionRetries = 0;
      connect();
      unawaited(refresh());
    }
  }

  void retryConnection(String target) {
    if (disposed ||
        !mounted ||
        workspace != target ||
        !widget.connectLive ||
        checkingConnection ||
        (reconnect?.isActive ?? false)) {
      return;
    }
    heartbeat?.cancel();
    const delays = [1, 2, 4, 8, 15, 30];
    final delay = delays[connectionRetries.clamp(0, 5)];
    connectionRetries++;
    setState(() => connection = 'Reconnecting');
    reconnect = Timer(Duration(seconds: delay), () async {
      checkingConnection = true;
      try {
        await api.request('/auth/me');
        await api.request('/workspaces/$target');
        checkingConnection = false;
        if (!disposed && workspace == target) connect();
      } catch (e) {
        checkingConnection = false;
        if (disposed || !mounted || workspace != target) return;
        final message = e.toString().toLowerCase();
        if (message.contains('session expired') ||
            message.contains('not a member') ||
            message.contains('access forbidden')) {
          setState(() {
            connection = 'Sign-in needs attention';
            error = e.toString();
          });
        } else {
          retryConnection(target);
        }
      }
    });
  }

  void connect() {
    if (disposed || workspace.isEmpty || !widget.connectLive) return;
    heartbeat?.cancel();
    reconnect?.cancel();
    final previous = channel;
    channel = null;
    previous?.sink.close();
    final base = api.baseUrl
        .replaceFirst('https://', 'wss://')
        .replaceFirst('http://', 'ws://')
        .replaceFirst('/api/v1', '/ws');
    final target = workspace;
    final current = IOWebSocketChannel.connect(
        Uri.parse(
            '$base/$target?client=mobile&token=${Uri.encodeComponent(api.token)}'),
        headers: {'Authorization': 'Bearer ${api.token}'});
    channel = current;
    bool active() =>
        !disposed &&
        mounted &&
        workspace == target &&
        identical(channel, current);
    current.ready.timeout(const Duration(seconds: 20)).then((_) {
      if (!active()) return;
      connectionRetries = 0;
      setState(() {
        connection = 'Connected';
        serverReachable = true;
      });
      heartbeat = Timer.periodic(const Duration(seconds: 20), (_) {
        if (active()) current.sink.add(jsonEncode({'type': 'ping'}));
      });
      current.sink
          .add(jsonEncode({'type': 'resync', 'since_sequence': sequence}));
    }).catchError((_) {
      if (active()) retryConnection(target);
    });
    current.stream.listen((raw) {
      if (!active()) return;
      try {
        final event = jsonDecode(raw);
        if (event['type'] == 'pong') return;
        final next = event['sequence'] as int? ?? 0;
        if (next > 0 && next <= sequence) return;
        sequence = next;
        if (!busy) unawaited(action(refresh));
      } catch (_) {}
    }, onError: (_) {
      if (active()) retryConnection(target);
    }, onDone: () {
      if (active()) retryConnection(target);
    });
  }

  Future<void> refresh() async {
    if (workspace.isEmpty) {
      await refreshInvitations();
      return;
    }
    final root = '/workspaces/$workspace';
    final results = await Future.wait([
      api.request('$root/members'),
      api.request('$root/workflows/?mine=true'),
      api.request('$root/tasks/'),
      api.request('$root/agents/history'),
      api.request('/notifications'),
      api.request('$root/negotiations')
    ]);
    if (mounted) {
      setState(() {
        members = results[0];
        workflows = results[1];
        tasks = results[2];
        messages = results[3];
        notifications = results[4];
        negotiations = results[5];
      });
    }
  }

  Future<void> changePanel(String value) async {
    if (needsVoice && value != 'Voice studio' && value != 'Settings') {
      await wake.pause();
      setState(() {
        panel = 'Voice studio';
        error = 'Complete your personal voice setup first.';
      });
      return;
    }
    if (wakeRemembered && !wake.enabled) {
      await wake.start();
    }
    setState(() => panel = value);
    await action(() async {
      if (value == 'Calendar') {
        calendars = await api.request('/calendar/list');
        cache = Map<String, dynamic>.from(await api.request('/calendar/mine'));
      }
      if (value == 'Voice studio') {
        voiceProfile =
            Map<String, dynamic>.from(await api.request('/voice/profile'));
      }
      if (value == 'Settings') {
        ai = Map<String, dynamic>.from(await api.request('/ai-settings'));
      }
    });
  }

  Future<Map<String, String>?> inputs(
      String title, Map<String, String> fields) async {
    final controllers =
        fields.map((key, value) => MapEntry(key, TextEditingController()));
    final values = await showDialog<Map<String, String>>(
        context: context,
        builder: (ctx) => AlertDialog(
                title: Text(title),
                content: SingleChildScrollView(
                    child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: fields.entries
                            .map((entry) => Padding(
                                padding: const EdgeInsets.only(bottom: 14),
                                child: TextField(
                                    controller: controllers[entry.key],
                                    decoration:
                                        InputDecoration(labelText: entry.value),
                                    obscureText: entry.key == 'api_key')))
                            .toList())),
                actions: [
                  TextButton(
                      onPressed: () => Navigator.pop(ctx),
                      child: const Text('Cancel')),
                  ElevatedButton(
                      onPressed: () => Navigator.pop(
                          ctx,
                          controllers.map((key, value) =>
                              MapEntry(key, value.text.trim()))),
                      child: const Text('Continue'))
                ]));
    // Dialog controllers are released after the route's closing animation.
    Future<void>.delayed(const Duration(seconds: 1), () {
      for (final controller in controllers.values) {
        controller.dispose();
      }
    });
    return values;
  }

  Future<void> send() async {
    final text = message.text.trim();
    if (text.isEmpty) return;
    if (ai['configured'] != true) {
      await changePanel('Settings');
      return;
    }
    await wake.pause();
    if (mounted) setState(() => busyLabel = 'Nexa is thinking…');
    await action(() async {
      final result = await api.request('/workspaces/$workspace/agents/chat',
          method: 'POST', body: {'text': text});
      message.clear();
      if (result['ai_source'] == 'personal_fallback' && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
            content: Text(
                'Company Gemini was unavailable. Used your personal backup.')));
      }
      // Show the actual response immediately; unrelated dashboard requests
      // should not delay speech or hide a successful assistant response.
      if (mounted) {
        setState(() => messages = [
              ...messages,
              {'role': 'user', 'text': text},
              {'role': 'assistant', 'text': result['response']},
            ]);
      }
      final update = refresh().catchError((Object _) {
        if (mounted) {
          setState(() => error =
              'Reply received. Team updates could not refresh; check your connection.');
        }
      });
      if (speakReplies) await speakReply(result['response'].toString());
      await update;
      if (mounted && (result['workflow_id']?.toString().isNotEmpty ?? false)) {
        await wake.pause();
        if (!mounted) return;
        setState(() => panel = 'Meetings');
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
            content: Text(
                'Proposal ready. Only invited participants receive approval requests. Check Meetings for its status.')));
      }
    });
    if (mounted && panel == 'Assistant') wake.resume();
  }

  Future<void> stopSpeaking() async {
    final stopped = speechStopped;
    try {
      await tts.stop();
      await player.stop();
    } finally {
      if (stopped != null && !stopped.isCompleted) stopped.complete();
    }
  }

  Future<void> speakReply(String text) async {
    await wake.pause();
    final stopped = Completer<void>();
    speechStopped = stopped;
    try {
      await tts.stop();
      if (!offlineReplyVoice &&
          ai['tts_configured'] == true &&
          (naturalVoiceRetryAfter == null ||
              DateTime.now().isAfter(naturalVoiceRetryAfter!))) {
        try {
          final spoken = ReplyVoice.spokenText(text);
          if (spoken.isEmpty) return;
          // Short chunks reduce the wait for the first spoken sentence.
          final chunks = ReplyVoice.chunks(spoken);
          for (final chunk in chunks) {
            if (!mounted || stopped.isCompleted) return;
            setState(() {
              speaking = false;
              busyLabel = 'Preparing natural voice…';
            });
            final result = await api.request('/voice/speech',
                method: 'POST', body: {'text': chunk});
            if (!mounted || stopped.isCompleted) return;
            final ended = Completer<void>();
            final subscription = player.onPlayerComplete.listen((_) {
              if (!ended.isCompleted) ended.complete();
            });
            try {
              replyVoice.label = 'NVIDIA Chatterbox · English';
              setState(() {
                speaking = true;
                busyLabel = 'Nexa is speaking…';
              });
              await player.play(BytesSource(
                  base64Decode(result['audio_base64']),
                  mimeType: 'audio/wav'));
              await Future.any([ended.future, stopped.future])
                  .timeout(const Duration(minutes: 2));
            } finally {
              await subscription.cancel();
            }
          }
          return;
        } catch (_) {
          if (stopped.isCompleted || !mounted) return;
          await player.stop();
          if (!mounted || stopped.isCompleted) return;
          naturalVoiceRetryAfter =
              DateTime.now().add(const Duration(minutes: 1));
          ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
              content: Text(
                  'Natural voice is unavailable. Using your phone voice.')));
        }
      }
      final available = await replyVoice.configure(tts, wake.preferredLocale,
          offline: offlineReplyVoice);
      if (!available) {
        throw StateError(
            'No downloaded voice is available for this language. Choose Natural voice or download a voice in Android speech settings.');
      }
      await tts.awaitSpeakCompletion(true);
      await tts.setSpeechRate(speechRate);
      await tts.setPitch(1.0);
      if (!mounted || stopped.isCompleted) return;
      setState(() {
        speaking = true;
        busyLabel = 'Nexa is speaking…';
      });
      final spoken = ReplyVoice.spokenText(text);
      if (spoken.isEmpty) return;
      Future<void> play() async {
        final result =
            await Future.any<dynamic>([tts.speak(spoken), stopped.future])
                .timeout(const Duration(minutes: 2), onTimeout: () async {
          await tts.stop();
          throw StateError(
              'Voice playback timed out. Your reply is still in chat.');
        });
        if (result == 0 || result == -1) {
          throw StateError('Voice playback was unavailable.');
        }
      }

      try {
        await play();
      } catch (_) {
        if (stopped.isCompleted || !replyVoice.networkVoice) rethrow;
        await tts.stop();
        if (!await replyVoice.configure(tts, wake.preferredLocale,
            offline: true)) {
          rethrow;
        }
        if (!mounted || stopped.isCompleted) return;
        await tts.awaitSpeakCompletion(true);
        await tts.setSpeechRate(speechRate);
        await tts.setPitch(1.0);
        await play();
      }
    } finally {
      if (identical(speechStopped, stopped)) speechStopped = null;
      if (mounted) {
        setState(() => speaking = false);
        if (panel == 'Assistant') wake.resume();
      }
    }
  }

  Future<void> record(bool enrollment, {bool check = false}) async {
    if (recording) {
      await stopRecording();
      return;
    }
    await wake.pause();
    await deleteAudio();
    if (!await recorder.hasPermission()) {
      throw Exception('Microphone permission is required.');
    }
    final directory = await getTemporaryDirectory();
    audioPath =
        '${directory.path}/aura-${DateTime.now().microsecondsSinceEpoch}.wav';
    await recorder.start(
        const RecordConfig(
            encoder: AudioEncoder.wav, sampleRate: 16000, numChannels: 1),
        path: audioPath!);
    recordingStarted = DateTime.now();
    setState(() {
      recording = true;
      enrollmentMode = enrollment;
      checkingVoice = check;
    });
    recordLimit = Timer(Duration(seconds: enrollment ? 10 : 20),
        () => unawaited(action(stopRecording)));
  }

  Future<void> stopRecording() async {
    recordLimit?.cancel();
    final path = await recorder.stop();
    setState(() => recording = false);
    if (path == null) throw Exception('No recording was captured.');
    final raw = await File(path).readAsBytes();
    if (enrollmentMode) {
      if (recordingStarted == null ||
          DateTime.now().difference(recordingStarted!).inMilliseconds < 1500) {
        await deleteAudio();
        throw Exception(
            'Speak clearly for at least 1.5 seconds. Tap stop when finished.');
      }
      setState(() => samples.add(raw));
    } else if (checkingVoice) {
      try {
        voiceCheck = Map<String, dynamic>.from(await api.request(
            '/voice/verify',
            method: 'POST',
            body: {'audio_base64': base64Encode(raw)}));
      } finally {
        await deleteAudio();
      }
    } else {
      try {
        final result = await api.request('/workspaces/$workspace/agents/voice',
            method: 'POST', body: {'audio_base64': base64Encode(raw)});
        if (result['verification']['status'] != 'MATCH') {
          throw Exception(result['response']);
        }
        final update = refresh();
        if (speakReplies) await speakReply(result['response'].toString());
        await update;
      } finally {
        await deleteAudio();
      }
    }
  }

  Future<void> enroll() async {
    final consent = await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
                title: const Text('Store your voice profile?'),
                content: const Text(
                    'AURA will store an encrypted speaker template linked to your account. Raw recordings are discarded. Voice matching does not replace account authentication and can be fooled by replay or synthetic speech.'),
                actions: [
                  TextButton(
                      onPressed: () => Navigator.pop(ctx, false),
                      child: const Text('Cancel')),
                  ElevatedButton(
                      onPressed: () => Navigator.pop(ctx, true),
                      child: const Text('I consent'))
                ]));
    if (consent != true) return;
    await api.request('/voice/enroll', method: 'POST', body: {
      'audio_base64_samples': samples.map(base64Encode).toList(),
      'consent': true
    });
    samples.clear();
    await deleteAudio();
    voiceProfile =
        Map<String, dynamic>.from(await api.request('/voice/profile'));
    if (mounted) {
      setState(() => panel = workspace.isEmpty
          ? 'People'
          : (cache['calendars'] as List? ?? []).isEmpty
              ? 'Calendar'
              : 'Assistant');
    }
  }

  Future<void> connectCalendar() async {
    // Refresh AURA authentication before obtaining a short-lived, single-use
    // Google authorization code. Check the selected Google account explicitly.
    final currentUser = await api.request('/auth/me');
    await initializeGoogle(api);
    googleAccount = await GoogleSignIn.instance.authenticate();
    if (googleAccount!.email.toLowerCase() !=
        currentUser['email'].toString().toLowerCase()) {
      throw Exception(
          'Choose the same Google account you used to sign in to AURA.');
    }
    final authorization =
        await googleAccount!.authorizationClient.authorizeServer([
      'https://www.googleapis.com/auth/calendar.readonly',
      'https://www.googleapis.com/auth/calendar.events'
    ]);
    if (authorization == null) {
      throw Exception(
          'Google did not return a new server code. Use the browser Calendar reconnect flow, or revoke this app in Google permissions and sign in again.');
    }
    await api.request('/integrations/google/server-code',
        method: 'POST', body: {'code': authorization.serverAuthCode});
    calendars = await api.request('/calendar/list');
    if (!calendars.any((c) => c['selected'] == true)) {
      for (final calendar in calendars) {
        if (calendar['primary'] == true) calendar['selected'] = true;
      }
    }
  }

  Future<void> decide(dynamic workflow, String decision) async {
    await api.request(
        '/workspaces/$workspace/workflows/${workflow['id']}/approvals',
        method: 'POST',
        body: {
          'proposal_revision_id': workflow['current_revision']['id'],
          'decision': decision
        });
    await refresh();
  }

  Widget card(List<Widget> children) => Container(
      margin: const EdgeInsets.only(bottom: 18),
      padding: const EdgeInsets.all(20),
      decoration: AuraTheme.glassDecoration(context: context, radius: 26),
      child: Material(
          color: Colors.transparent,
          child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: children)));
  Widget title(String value) => Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Text(value,
          style: const TextStyle(
              fontSize: 20, fontWeight: FontWeight.w600, letterSpacing: -.5)));
  Widget muted(String value) => Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Text(value,
          style: const TextStyle(
              color: AuraTheme.textSecondary, height: 1.65, fontSize: 12)));
  Widget button(String label, Future<void> Function() callback,
          {bool enabled = true}) =>
      Padding(
          padding: const EdgeInsets.only(top: 10),
          child: OutlinedButton(
              onPressed: busy || !enabled ? null : () => action(callback),
              child: Text(label)));
  DateTime toIST(dynamic value) {
    if (value == null) {
      return DateTime.now().toUtc().add(const Duration(hours: 5, minutes: 30));
    }
    if (value is DateTime) {
      final utc = value.isUtc ? value : value.toUtc();
      return utc.add(const Duration(hours: 5, minutes: 30));
    }
    final str = value.toString().trim();
    if (str.isEmpty) {
      return DateTime.now().toUtc().add(const Duration(hours: 5, minutes: 30));
    }
    final dt = DateTime.tryParse(str);
    if (dt == null) {
      return DateTime.now().toUtc().add(const Duration(hours: 5, minutes: 30));
    }
    final isUtcString = str.endsWith('Z') ||
        str.contains('+') ||
        (str.length > 10 && str.substring(10).contains('-'));
    final utc = isUtcString
        ? dt.toUtc()
        : DateTime.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second,
            dt.millisecond);
    return utc.add(const Duration(hours: 5, minutes: 30));
  }

  String clockTimeIST(dynamic value) {
    if (value == null) return '--:--';
    final ist = toIST(value);
    final hour = ist.hour;
    final minute = ist.minute.toString().padLeft(2, '0');
    final period = hour >= 12 ? 'PM' : 'AM';
    final h12 = hour % 12 == 0 ? 12 : hour % 12;
    return '${h12.toString().padLeft(2, '0')}:$minute $period';
  }

  String timeIST(dynamic value) {
    if (value == null) return '--:-- IST';
    return '${clockTimeIST(value)} IST';
  }

  String dateTimeIST(dynamic value) {
    if (value == null) return 'Time not set';
    final ist = toIST(value);
    const months = [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ];
    final day = ist.day.toString().padLeft(2, '0');
    final mon = months[ist.month - 1];
    final year = ist.year;
    return '$day $mon $year · ${clockTimeIST(ist)} IST';
  }

  String dateHeaderIST(dynamic value) {
    final ist = toIST(value);
    final nowIST =
        DateTime.now().toUtc().add(const Duration(hours: 5, minutes: 30));
    final today = DateTime(nowIST.year, nowIST.month, nowIST.day);
    final targetDay = DateTime(ist.year, ist.month, ist.day);
    final diffDays = targetDay.difference(today).inDays;

    const weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    const months = [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ];
    final dayName = weekdays[ist.weekday - 1];
    final mon = months[ist.month - 1];
    final dateStr = '${ist.day.toString().padLeft(2, '0')} $mon ${ist.year}';

    if (diffDays == 0) {
      return 'Today · $dayName, $dateStr (IST)';
    } else if (diffDays == 1) {
      return 'Tomorrow · $dayName, $dateStr (IST)';
    } else if (diffDays == -1) {
      return 'Yesterday · $dayName, $dateStr (IST)';
    } else {
      return '$dayName, $dateStr (IST)';
    }
  }

  String time(dynamic value) => dateTimeIST(value);

  List<Widget> content() {
    if ((workspace.isEmpty &&
            panel != 'Voice studio' &&
            panel != 'Settings' &&
            panel != 'Notifications') ||
        panel == 'People') {
      return [
        card([
          title(workspace.isEmpty
              ? 'Your team is waiting.'
              : 'Your company workspace'),
          muted('Signed in as ${me['email'] ?? ''}'),
          if (invitations.isEmpty && workspace.isEmpty) ...[
            const SizedBox(height: 14),
            const Icon(Icons.mark_email_unread_outlined,
                size: 42, color: AuraTheme.accentPurple),
            muted(
                'Ask your company admin to invite this Gmail address from the web dashboard. Your invitation appears here automatically.'),
            button('Check for invitations', refreshInvitations),
          ],
          ...members.map((m) => ListTile(
              contentPadding: EdgeInsets.zero,
              leading: CircleAvatar(
                  child: Text((m['user']?['full_name'] ?? '?')
                      .toString()
                      .substring(0, 1))),
              title: Text(m['user']?['full_name'] ?? m['user_id']),
              subtitle: Text(m['role']))),
          if (workspace.isNotEmpty)
            muted('Your admin manages this workspace from AURA HQ.'),
        ])
      ];
    }
    if (panel == 'Assistant') {
      return [
        ...messages.map((m) => Container(
            margin:
                EdgeInsets.only(bottom: 12, left: m['role'] == 'user' ? 24 : 0),
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
                color: m['role'] == 'user'
                    ? const Color(0xFFEDF2FF)
                    : Colors.white,
                borderRadius: BorderRadius.circular(16)),
            child:
                Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(m['role'] == 'user' ? 'You' : 'AURA',
                  style: const TextStyle(
                      fontSize: 10, color: AuraTheme.textMuted)),
              const SizedBox(height: 8),
              Text(m['text'], style: const TextStyle(height: 1.6))
            ]))),
        if (messages.isEmpty)
          card([
            title('What’s on your mind?'),
            muted(
                'Gemini understands your request and plans an action. AURA checks permissions and requests approvals before changing calendars.'),
            muted(
                'Ask who is in your workspace, check your tasks, or prepare a meeting with a specific date, time and duration.')
          ]),
        TextField(
            controller: message,
            minLines: 2,
            maxLines: 5,
            decoration: const InputDecoration(labelText: 'Message AURA')),
        const SizedBox(height: 14),
        Wrap(spacing: 12, runSpacing: 10, children: [
          ElevatedButton(
              onPressed: busy ? null : send, child: const Text('Send')),
          OutlinedButton.icon(
              onPressed: busy
                  ? null
                  : () => action(() async {
                        if (ai['configured'] != true) {
                          await changePanel('Settings');
                          return;
                        }
                        if (voiceProfile['enrolled'] != true ||
                            voiceProfile['commands_calibrated'] != true) {
                          await changePanel('Voice studio');
                          return;
                        }
                        await record(false);
                      }),
              icon: Icon(recording ? Icons.stop : Icons.mic_none),
              label: Text(recording ? 'Stop recording' : 'Speak'))
        ]),
        SwitchListTile(
            contentPadding: EdgeInsets.zero,
            title: const Text('Read replies aloud',
                style: TextStyle(fontSize: 13)),
            value: speakReplies,
            onChanged: (value) => setState(() => speakReplies = value)),
      ];
    }
    if (panel == 'Voice studio') {
      const sentences = [
        'My voice makes this workspace a little more personal.',
        'A good day starts with time to think and a clear plan.',
        'Nexa, help our team find a little more time together.',
        'I enjoy learning something new and sharing ideas with my team.',
        'Let us make space for the work that matters most today.'
      ];
      final enrolled = voiceProfile['enrolled'] == true;
      return [
        card([
          const NexaMark(size: 38),
          const SizedBox(height: 24),
          title(enrolled
              ? 'Your voice, recognized.'
              : 'Let’s get to know your voice.'),
          muted(enrolled
              ? 'Your personal voice profile is saved securely to your account.'
              : 'Read three short sentences. Take 5–10 seconds for each, in a quiet place.'),
          if (!enrolled || samples.isNotEmpty) ...[
            const SizedBox(height: 16),
            Row(
                children: List.generate(
                    5,
                    (index) => Expanded(
                        child: Container(
                            height: 7,
                            margin: const EdgeInsets.only(right: 6),
                            decoration: BoxDecoration(
                                color: index < samples.length
                                    ? AuraTheme.accentPurple
                                    : AuraTheme.surfaceBorder,
                                borderRadius: BorderRadius.circular(8)))))),
            muted(
                '${samples.length} of 3 required samples · up to 5 for a better profile'),
            const SizedBox(height: 12),
            Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                    color: const Color(0xFF24283F),
                    borderRadius: BorderRadius.circular(18)),
                child: Text(sentences[samples.length.clamp(0, 4)],
                    style: const TextStyle(height: 1.8, fontSize: 16))),
            const SizedBox(height: 22),
            SizedBox(
                width: double.infinity,
                child: FilledButton.icon(
                    onPressed: busy || samples.length >= 5 && !recording
                        ? null
                        : () => action(() => record(true)),
                    icon: Icon(recording
                        ? Icons.stop_rounded
                        : Icons.mic_none_rounded),
                    label: Text(recording
                        ? 'Recording · tap to stop'
                        : 'Record sample ${samples.length + 1}'))),
            if (audioPath != null)
              TextButton.icon(
                  onPressed: () => player.play(DeviceFileSource(audioPath!)),
                  icon: const Icon(Icons.play_arrow_rounded),
                  label: const Text('Listen to your recording')),
            if (samples.isNotEmpty)
              TextButton(
                  onPressed: recording
                      ? null
                      : () => action(() async {
                            samples.removeLast();
                            await deleteAudio();
                          }),
                  child: const Text('Retake last sample')),
            if (samples.length >= 3)
              SizedBox(
                  width: double.infinity,
                  child: FilledButton(
                      onPressed:
                          busy || recording ? null : () => action(enroll),
                      child: const Text('Save my voice & continue'))),
            muted(
                'You choose whether to save your encrypted voice template. Recordings are discarded after processing.'),
          ],
          if (enrolled) ...[
            const SizedBox(height: 16),
            button(
                recording && checkingVoice
                    ? 'Stop voice check'
                    : 'Check my voice',
                () => record(false, check: true)),
            if (voiceCheck != null)
              muted(
                  '${voiceCheck!['status']} · similarity ${voiceCheck!['similarity_score']} · ${voiceCheck!['detail']}'),
            button('Continue to my team', () async {
              await changePanel(workspace.isEmpty
                  ? 'People'
                  : (cache['calendars'] as List? ?? []).isEmpty
                      ? 'Calendar'
                      : 'Assistant');
            }),
            ExpansionTile(
                tilePadding: EdgeInsets.zero,
                title: const Text('Manage voice profile',
                    style: TextStyle(fontSize: 12)),
                children: [
                  button('Delete saved profile', () async {
                    await api.request('/voice/profile', method: 'DELETE');
                    voiceProfile = {...voiceProfile, 'enrolled': false};
                    voiceCheck = null;
                  })
                ]),
          ],
        ])
      ];
    }
    if (panel == 'Calendar') {
      final selected = cache['calendars'] as List? ?? [];
      final connected = selected.isNotEmpty;
      return [
        card([
          Row(children: [
            const Icon(Icons.calendar_month_rounded, color: _violet),
            const SizedBox(width: 12),
            Expanded(
                child: Text(
                    connected ? 'Google Calendar' : 'Connect Google Calendar',
                    style: const TextStyle(
                        fontSize: 17, fontWeight: FontWeight.w600))),
            Icon(connected ? Icons.check_circle : Icons.link,
                color: connected ? AuraTheme.accentEmerald : _violet)
          ]),
          const SizedBox(height: 12),
          if (!connected) button('Connect Google Calendar', connectCalendar),
          if (connected) ...[
            const SizedBox(height: 18),
            Row(
              children: [
                const Expanded(
                  child: Text(
                    'Your calendar timeline',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w600,
                      letterSpacing: -.5,
                    ),
                  ),
                ),
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: _violet.withValues(alpha: 0.12),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.schedule_rounded, size: 12, color: _violet),
                      SizedBox(width: 4),
                      Text('IST (UTC+05:30)',
                          style: TextStyle(
                              fontSize: 10,
                              fontWeight: FontWeight.w600,
                              color: _violet)),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            referenceTabs(['Timeline', 'Week', 'Month'], calendarFilter,
                (value) => redraw(() => calendarFilter = value)),
            const SizedBox(height: 16),
            ...calendarTimeline(),
          ],
          if (calendars.isNotEmpty) ...[
            const SizedBox(height: 18),
            ...calendars.map((calendar) => CheckboxListTile(
                contentPadding: EdgeInsets.zero,
                title: Text(calendar['name'],
                    style: const TextStyle(fontSize: 13)),
                subtitle: Text(calendar['timezone'],
                    style: const TextStyle(fontSize: 11)),
                value: calendar['selected'] == true,
                onChanged: (value) =>
                    setState(() => calendar['selected'] = value))),
            button('Share selected availability & continue', () async {
              await api.request('/calendar/selection', method: 'PUT', body: {
                'calendar_ids': calendars
                    .where((c) => c['selected'] == true)
                    .map((c) => c['id'])
                    .toList()
              });
              cache = Map<String, dynamic>.from(
                  await api.request('/calendar/mine'));
              me = Map<String, dynamic>.from(await api.request('/auth/me'));
              if (mounted) setState(() => panel = 'Assistant');
            }),
          ],
          if (connected) ...[
            const SizedBox(height: 12),
            ...selected.map((calendar) => muted(calendar['error'] != null
                ? 'Calendar needs attention. Reconnect to refresh permission.'
                : 'Last sync: ${time(calendar['synced_at'])}')),
            button('Open Nexa', () async {
              await changePanel('Assistant');
            }),
            ExpansionTile(
                tilePadding: EdgeInsets.zero,
                title: const Text('Calendar settings',
                    style: TextStyle(fontSize: 12)),
                children: [
                  button('Reconnect or change calendars', connectCalendar),
                  button('Sync now', () async {
                    await api.request('/calendar/sync', method: 'POST');
                    cache = Map<String, dynamic>.from(
                        await api.request('/calendar/mine'));
                  })
                ]),
          ],
          muted(
              'Sign-in and calendar permission are separate. Calendar changes require the team’s approval.'),
        ])
      ];
    }
    if (panel == 'Meetings') {
      return [
        card([
          title('Prepare a meeting'),
          muted(
              'Ask your assistant with the title, teammate names, exact time and duration. Every required participant approves before the event is created.'),
          button('Open assistant', () => changePanel('Assistant'))
        ]),
        referenceTabs(['My meetings', 'Pending', 'Past'], meetingFilter,
            (value) => redraw(() => meetingFilter = value)),
        const SizedBox(height: 18),
        ...filteredMeetings().map((w) {
          final revision = w['current_revision'];
          return card([
            title(w['title']),
            Chip(
                label: Text(w['state'].toString().replaceAll('_', ' '),
                    style: const TextStyle(fontSize: 10)),
                avatar: Icon(
                    w['state'] == 'COMPLETED'
                        ? Icons.check_circle_outline
                        : Icons.hourglass_empty_rounded,
                    size: 16,
                    color: _violet),
                backgroundColor: Theme.of(context).brightness == Brightness.dark
                    ? const Color(0xFF263957)
                    : const Color(0xFFEDF2FF),
                side: BorderSide.none),
            if (w['calendar_event']?['html_link'] != null)
              button('Open in Google Calendar', () async {
                await launchUrl(Uri.parse(w['calendar_event']['html_link']),
                    mode: LaunchMode.externalApplication);
              }),
            if (revision != null) ...[
              muted(
                  '${time(revision['start_time'])} · ${revision['duration_minutes']} min · Revision ${revision['revision_number']}'),
              for (final vote in revision['approvals'])
                ListTile(
                    contentPadding: EdgeInsets.zero,
                    title: Text(vote['user_name'] ?? vote['user_id']),
                    trailing: Text(vote['decision'],
                        style: const TextStyle(fontSize: 10))),
              if (['AWAITING_APPROVALS', 'FAILED'].contains(w['state']) &&
                  (revision['approvals'] as List)
                      .any((a) => a['user_id'] == me['id'])) ...[
                button(
                    w['state'] == 'FAILED'
                        ? 'Retry & reconcile'
                        : 'Approve this revision',
                    () => decide(w, 'APPROVED')),
                if (w['state'] == 'AWAITING_APPROVALS')
                  button('Reject', () => decide(w, 'REJECTED')),
                if (w['state'] == 'AWAITING_APPROVALS')
                  button('Suggest another time', () async {
                    final values = await inputs('Counteroffer',
                        {'start': 'Start (YYYY-MM-DDTHH:MM+05:30)'});
                    if (values == null) return;
                    final start = DateTime.parse(values['start']!);
                    await api.request(
                        '/workspaces/$workspace/workflows/${w['id']}/counter',
                        method: 'POST',
                        body: {
                          'proposal_revision_id': revision['id'],
                          'preferred_start': start.toUtc().toIso8601String(),
                          'preferred_end': start
                              .add(Duration(
                                  minutes: revision['duration_minutes']))
                              .toUtc()
                              .toIso8601String()
                        });
                    await refresh();
                  }),
              ],
              if (w['state'] == 'COMPLETED')
                button('Propose rescheduling', () async {
                  final values = await inputs('Proposed new time',
                      {'start': 'Start (YYYY-MM-DDTHH:MM+05:30)'});
                  if (values == null) return;
                  final start = DateTime.parse(values['start']!);
                  await api.request(
                      '/workspaces/$workspace/workflows/${w['id']}/reschedule',
                      method: 'POST',
                      body: {
                        'start': start.toUtc().toIso8601String(),
                        'end': start
                            .add(
                                Duration(minutes: revision['duration_minutes']))
                            .toUtc()
                            .toIso8601String()
                      });
                  await refresh();
                }),
            ],
          ]);
        }),
        if (filteredMeetings().isEmpty) muted('No meetings in this view.'),
      ];
    }
    if (panel == 'Negotiations') {
      return [
        card([
          title('When plans overlap.'),
          muted(
              'Ask permission to explore moving an app-managed conflict. The original participants approve the move, then the new meeting needs its own approval.'),
          button('Request exploration', () async {
            final values = await inputs('Request rescheduling exploration', {
              'title': 'New meeting title',
              'teammate': 'Exact teammate name',
              'start': 'Start with timezone, e.g. 2026-11-05T15:00+05:30',
              'duration': 'Duration in minutes'
            });
            if (values == null) return;
            final matches = members
                .where((m) =>
                    (m['user']?['full_name'] ?? '').toString().toLowerCase() ==
                    values['teammate']!.toLowerCase())
                .toList();
            if (matches.length != 1) {
              throw Exception('Choose an unambiguous workspace teammate.');
            }
            final start = DateTime.parse(values['start']!);
            final duration = int.parse(values['duration']!);
            await api.request('/workspaces/$workspace/negotiations',
                method: 'POST',
                body: {
                  'title': values['title'],
                  'target_user_id': matches.first['user_id'],
                  'start': start.toUtc().toIso8601String(),
                  'end': start
                      .add(Duration(minutes: duration))
                      .toUtc()
                      .toIso8601String()
                });
            await refresh();
          }),
          ...negotiations.map((n) =>
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                const Divider(),
                title(n['title']),
                muted('${n['state']} · ${time(n['start'])}'),
                if (n['state'] == 'AWAITING_PERMISSION' &&
                    n['target_user_id'] == me['id']) ...[
                  button('Allow exploration', () async {
                    await api.request(
                        '/workspaces/$workspace/negotiations/${n['id']}/permission',
                        method: 'POST',
                        body: {'allow': true});
                    await refresh();
                  }),
                  button('Decline', () async {
                    await api.request(
                        '/workspaces/$workspace/negotiations/${n['id']}/permission',
                        method: 'POST',
                        body: {'allow': false});
                    await refresh();
                  }),
                ]
              ])),
          if (negotiations.isEmpty) muted('No negotiations to review.'),
        ])
      ];
    }
    if (panel == 'Tasks') {
      return [
        card([
          title('Make progress, together.'),
          button('Add task', () async {
            final values = await inputs('New task', {'title': 'Task title'});
            if (values == null) return;
            await api.request('/workspaces/$workspace/tasks/',
                method: 'POST', body: values);
            await refresh();
          }),
          ...tasks.map((task) => CheckboxListTile(
              contentPadding: EdgeInsets.zero,
              title: Text(task['title']),
              subtitle: Text(task['status']),
              value: task['status'] == 'DONE',
              onChanged: busy
                  ? null
                  : (value) => action(() async {
                        await api.request(
                            '/workspaces/$workspace/tasks/${task['id']}/status',
                            method: 'PATCH',
                            body: {'status': value == true ? 'DONE' : 'TODO'});
                        await refresh();
                      }))),
          if (tasks.isEmpty) muted('No tasks yet.')
        ])
      ];
    }
    if (panel == 'Notifications') {
      return [
        card([
          title('Your inbox'),
          ...notifications.map((n) => ListTile(
              contentPadding: EdgeInsets.zero,
              title: Text(n['payload']['message'] ?? n['kind']),
              subtitle: Text(time(n['created_at'])),
              trailing: n['read'] == true
                  ? null
                  : IconButton(
                      icon: const Icon(Icons.check),
                      onPressed: () => action(() async {
                            await api.request('/notifications/${n['id']}/read',
                                method: 'POST');
                            await refresh();
                          })))),
          if (notifications.isEmpty)
            muted(
                'No notifications yet. Updates arrive while connected; missed notifications remain saved.')
        ])
      ];
    }
    if (panel == 'Settings') {
      return [
        card([
          title('Connect your AI assistant'),
          DropdownButtonFormField<String>(
            initialValue: ai['provider']?.toString() ?? 'gemini',
            decoration: const InputDecoration(labelText: 'AI provider'),
            items: const [
              DropdownMenuItem(value: 'gemini', child: Text('Gemini')),
              DropdownMenuItem(
                  value: 'groq', child: Text('Groq · fast responses')),
              DropdownMenuItem(
                  value: 'nvidia', child: Text('NVIDIA · Nemotron')),
            ],
            onChanged: busy
                ? null
                : (value) => action(() async {
                      if (value == null) return;
                      await api.request('/ai-settings',
                          method: 'PUT', body: {'provider': value});
                      ai = Map<String, dynamic>.from(
                          await api.request('/ai-settings'));
                    }),
          ),
          muted(
              'Your provider choice is saved to your account. All providers follow the same meeting approvals.'),
          muted(
              'Mode: ${ai['mode'] ?? 'company'}. Model: ${ai['model'] ?? 'not configured'}.'),
          muted(ai['personal_fallback_available'] == true
              ? 'Company AI is the default. Your personal backup is ready.'
              : ai['company_key_configured'] == true
                  ? 'Company AI is the default. Adding your own backup key is optional.'
                  : 'Connect your own Gemini key to use Nexa.'),
          button('Get a Gemini API key', () async {
            await launchUrl(Uri.parse('https://aistudio.google.com/api-keys'),
                mode: LaunchMode.externalApplication);
          }),
          button(
              ai['company_key_configured'] == true
                  ? 'Add my personal backup key'
                  : 'Save personal Gemini key', () async {
            final values = await inputs('Personal Gemini key',
                {'api_key': 'API key (sent securely, then encrypted)'});
            if (values == null) return;
            if (!api.baseUrl.startsWith('https://')) {
              throw Exception('Personal keys require HTTPS.');
            }
            if (mounted) {
              setState(() => busyLabel =
                  'Checking your Gemini key with Google · up to 25 seconds…');
            }
            await api.request('/ai-settings', method: 'PUT', body: {
              'mode':
                  ai['company_key_configured'] == true ? 'company' : 'personal',
              'api_key': values['api_key']
            });
            ai = Map<String, dynamic>.from(await api.request('/ai-settings'));
            if (mounted) {
              ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
                  content: Text('Your personal Nexa connection is ready.')));
              if (workspace.isNotEmpty && !needsVoice) {
                setState(() => panel = 'Assistant');
              }
            }
          }),
          if (ai['company_key_configured'] == true)
            button('Use company default (keep my backup)', () async {
              await api.request('/ai-settings',
                  method: 'PUT',
                  body: {'mode': 'company', 'keep_personal': true});
              ai = Map<String, dynamic>.from(await api.request('/ai-settings'));
            }),
          if (ai['personal_key_saved'] == true)
            button('Remove my personal backup', () async {
              await api.request('/ai-settings',
                  method: 'PUT', body: {'mode': 'company'});
              ai = Map<String, dynamic>.from(await api.request('/ai-settings'));
            }),
          button('Test selected connection', () async {
            await api.request('/ai-settings/test', method: 'POST');
            if (mounted) {
              ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
                  content: Text('Selected AI connection succeeded.')));
            }
          }),
          button('Update timezone & working hours', () async {
            final values = await inputs('Scheduling preferences', {
              'timezone': 'IANA timezone, e.g. Asia/Kolkata',
              'working_hours_start': 'Workday starts, HH:MM',
              'working_hours_end': 'Workday ends, HH:MM'
            });
            if (values == null) return;
            await api.request('/profile/preferences',
                method: 'PUT', body: values);
          }),
          muted(
              'Company Gemini is the default. If it fails, only your own saved backup key can be used. Your key stays private to your account.'),
        ]),
        card([
          Row(
            children: [
              const Icon(Icons.mic_rounded, color: _violet, size: 24),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    title('Hey Nexa voice activation'),
                    Text(
                      wake.enabled
                          ? 'Active · Listening for “Hey Nexa”'
                          : 'Off · Voice activation paused',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        color: wake.enabled
                            ? AuraTheme.accentEmerald
                            : AuraTheme.textSecondary,
                      ),
                    ),
                  ],
                ),
              ),
              Switch(
                value: wake.enabled,
                onChanged: (val) async {
                  if (val) {
                    await secureSession.write(
                        key: 'nexa_wake_enabled', value: 'true');
                    wakeRemembered = true;
                    await wake.start();
                  } else {
                    await secureSession.write(
                        key: 'nexa_wake_enabled', value: 'false');
                    wakeRemembered = false;
                    await wake.stop();
                  }
                  if (mounted) setState(() {});
                },
              ),
            ],
          ),
          const SizedBox(height: 8),
          muted(
              'Saying “Hey Nexa” wakes your assistant hands-free. Outside the app, it opens the Cosmic Assistant. Inside the app, it uses the workspace directly.'),
          button('Configure permissions & battery settings', () async {
            await NexaPermissionSheet.showIfRequired(context, force: true);
            if (mounted && wakeRemembered && !wake.enabled) {
              await wake.start();
            }
          }),
          button('Preview Cosmic Assistant overlay', openCosmicNexaModal),
        ]),
      ];
    }
    return [];
  }

  @override
  Widget build(BuildContext context) => nexaWorkspace(context);
}
