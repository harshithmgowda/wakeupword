part of 'connected_workspace.dart';

extension _NexaSignInUI on _SessionGateState {
  Future<void> openConnectionSettings(BuildContext context) async {
    String? testResult;
    bool testing = false;
    await showModalBottomSheet<void>(
        context: context,
        isScrollControlled: true,
        showDragHandle: true,
        builder: (ctx) => StatefulBuilder(
            builder: (ctx, setSheetState) => Padding(
                padding: EdgeInsets.fromLTRB(
                    24, 8, 24, MediaQuery.viewInsetsOf(ctx).bottom + 28),
                child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('Company Connection',
                          style: TextStyle(
                              fontSize: 22, fontWeight: FontWeight.w600)),
                      const SizedBox(height: 8),
                      const Text(
                          'Paste the current address from AURA_CURRENT_URL.txt on your PC. The bundled address may expire; this button does not restart Cloudflare.',
                          style: TextStyle(color: AuraTheme.textSecondary)),
                      const SizedBox(height: 14),
                      TextField(
                          controller: endpoint,
                          decoration: const InputDecoration(
                              labelText: 'Backend URL',
                              border: OutlineInputBorder()),
                          keyboardType: TextInputType.url),
                      const SizedBox(height: 12),
                      Wrap(spacing: 8, runSpacing: 8, children: [
                        ActionChip(
                          avatar: const Icon(Icons.cloud_outlined, size: 16),
                          label: const Text('Use bundled address'),
                          onPressed: () => setSheetState(() {
                            endpoint.text = AuraConfig.apiUrl;
                            testResult = null;
                          }),
                        ),
                        ActionChip(
                          avatar: const Icon(Icons.wifi, size: 16),
                          label: const Text('Local Wi-Fi (10.24.125.111)'),
                          onPressed: () => setSheetState(() {
                            endpoint.text = 'http://10.24.125.111:8000';
                            testResult = null;
                          }),
                        ),
                      ]),
                      if (testing)
                        const Padding(
                          padding: EdgeInsets.symmetric(vertical: 8),
                          child: Row(children: [
                            SizedBox(
                                width: 16,
                                height: 16,
                                child:
                                    CircularProgressIndicator(strokeWidth: 2)),
                            SizedBox(width: 10),
                            Text('Testing server health...',
                                style: TextStyle(fontSize: 13))
                          ]),
                        )
                      else if (testResult != null)
                        Padding(
                          padding: const EdgeInsets.symmetric(vertical: 8),
                          child: Text(
                            testResult!,
                            style: TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.w500,
                              color: testResult!.startsWith('✓')
                                  ? AuraTheme.accentEmerald
                                  : AuraTheme.accentRose,
                            ),
                          ),
                        ),
                      const SizedBox(height: 12),
                      Row(
                        children: [
                          Expanded(
                            child: OutlinedButton.icon(
                              icon: const Icon(Icons.network_check_rounded,
                                  size: 16),
                              label: const Text('Test Health',
                                  style: TextStyle(fontSize: 12)),
                              onPressed: testing
                                  ? null
                                  : () async {
                                      setSheetState(() => testing = true);
                                      try {
                                        final raw = endpoint.text
                                            .trim()
                                            .replaceAll(RegExp(r'/+$'), '');
                                        final res = await http
                                            .get(Uri.parse('$raw/health'))
                                            .timeout(
                                                const Duration(seconds: 5));
                                        setSheetState(() {
                                          testing = false;
                                          testResult = res.statusCode == 200
                                              ? '✓ Online & Healthy'
                                              : '✗ HTTP ${res.statusCode}';
                                        });
                                      } catch (_) {
                                        setSheetState(() {
                                          testing = false;
                                          testResult = '✗ Cannot reach server';
                                        });
                                      }
                                    },
                            ),
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: FilledButton(
                                onPressed: () async {
                                  String raw;
                                  try {
                                    raw = await checkCompanyEndpoint(
                                        endpoint.text);
                                  } catch (e) {
                                    if (ctx.mounted) {
                                      setSheetState(() => testResult = e
                                          .toString()
                                          .replaceFirst(
                                              'FormatException: ', ''));
                                    }
                                    return;
                                  }
                                  await secureSession.write(
                                      key: 'endpoint', value: raw);
                                  if (mounted) {
                                    clearError();
                                    if (ctx.mounted) Navigator.pop(ctx);
                                  }
                                },
                                child: const Text('Save & Done',
                                    style: TextStyle(fontSize: 12))),
                          ),
                        ],
                      )
                    ]))));
  }

  Widget nexaSignIn(BuildContext context) => Scaffold(
          body: SafeArea(
              child: ListView(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 28, vertical: 24),
                  children: [
            Row(children: [
              const NexaMark(size: 27),
              const SizedBox(width: 10),
              const Text('AURA',
                  style:
                      TextStyle(fontWeight: FontWeight.w700, letterSpacing: 2)),
              const Spacer(),
              IconButton(
                  tooltip: 'Connection settings',
                  icon: const Icon(Icons.tune_rounded),
                  onPressed: () => openConnectionSettings(context))
            ]),
            const SizedBox(height: 58),
            const Center(child: NexaOrb(size: 186)),
            const SizedBox(height: 44),
            const NexaGradientText('Meet Nexa.', size: 44),
            const SizedBox(height: 12),
            const Text('A little space.\nFor your biggest ideas.',
                style:
                    TextStyle(fontSize: 30, height: 1.25, letterSpacing: -.9)),
            const SizedBox(height: 20),
            const Text(
                'Your personal AI, connected to your team. Sign in with the Gmail address your company invited.',
                style: TextStyle(
                    color: AuraTheme.textSecondary, height: 1.7, fontSize: 15)),
            const SizedBox(height: 32),
            if (error != null)
              Padding(
                  padding: const EdgeInsets.only(bottom: 16),
                  child: Column(
                    children: [
                      Text(error!,
                          textAlign: TextAlign.center,
                          style: const TextStyle(color: AuraTheme.accentRose)),
                      const SizedBox(height: 10),
                      OutlinedButton.icon(
                        icon: const Icon(Icons.tune_rounded, size: 16),
                        label: const Text('Update Server / Tunnel URL'),
                        onPressed: () => openConnectionSettings(context),
                      ),
                    ],
                  )),
            SizedBox(
                height: 56,
                child: FilledButton.icon(
                    onPressed: busy ? null : () => login(google: true),
                    icon: const Icon(Icons.account_circle_outlined),
                    label:
                        Text(busy ? 'Connecting…' : 'Continue with Google'))),
            const SizedBox(height: 18),
            const Text('Voice setup · Your team · Your calendar',
                textAlign: TextAlign.center,
                style: TextStyle(
                    color: AuraTheme.textMuted, fontSize: 11, height: 1.8)),
            const SizedBox(height: 28),
            ExpansionTile(
                title: const Text('Other sign-in options',
                    style: TextStyle(fontSize: 13)),
                children: [
                  TextField(
                      controller: email,
                      decoration: const InputDecoration(labelText: 'Email'),
                      keyboardType: TextInputType.emailAddress),
                  const SizedBox(height: 12),
                  TextField(
                      controller: password,
                      decoration: const InputDecoration(labelText: 'Password'),
                      obscureText: true),
                  TextButton(
                      onPressed: busy ? null : () => login(google: false),
                      child: const Text('Sign in'))
                ]),
          ])));
}

extension _NexaWorkspaceUI on _ConnectedWorkspaceState {
  Future<void> toggleWake() async {
    if (wake.enabled) {
      await wake.stop();
      wakeRemembered = false;
      await secureSession.write(key: 'nexa_wake_enabled', value: 'false');
      return;
    }
    final consent = await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
                title: const Text('Enable “Nexa” activation?'),
                content: const Text(
                    'While the Assistant screen is open, The bundled sherpa-onnx model detects “Hey Nexa” phrase on your phone. After detection, your phone’s speech service transcribes the request and may process that audio. This preference is remembered when you reopen the app. Listening pauses in the background and while Nexa speaks. Finished voice requests are sent automatically. Meeting changes still require their normal approvals.'),
                actions: [
                  TextButton(
                      onPressed: () => Navigator.pop(ctx, false),
                      child: const Text('Not now')),
                  FilledButton(
                      onPressed: () => Navigator.pop(ctx, true),
                      child: const Text('Enable microphone'))
                ]));
    if (consent == true) {
      wakeRemembered = true;
      await secureSession.write(key: 'nexa_wake_enabled', value: 'true');
      await wake.start();
      if (panel != 'Assistant') await wake.pause();
    }
  }

  Future<void> talkNow() async {
    if (busy || speaking) return;
    await openCosmicNexaModal();
  }

  Future<void> openCosmicNexaModal() async {
    if (busy || speaking || voiceSheetOpen) return;
    voiceSheetOpen = true;
    if (!wake.dictating) unawaited(wake.dictate());
    if (!mounted) return;
    try {
      await NexaCosmicOverlay.show(
        context: context,
        wake: wake,
        messageController: message,
        isBusy: busy,
        isSpeaking: speaking,
        onSend: () {
          if (message.text.trim().isNotEmpty) {
            voiceSheetOpen = false;
            Navigator.of(context).pop();
            unawaited(send());
          }
        },
      );
    } finally {
      if (voiceSheetOpen) {
        voiceSheetOpen = false;
        await wake.pause();
        message.clear();
        if (mounted) wake.resume();
      }
    }
  }



  Widget animatedChatText(Map<String, dynamic> item) {
    final user = item['role'] == 'user';
    Widget text = SelectableText(item['text'].toString(),
        style: TextStyle(
            height: 1.7,
            fontSize: 14,
            color: user ? Colors.white : referenceInk));
    if (user) {
      text = ShaderMask(
          blendMode: BlendMode.srcIn,
          shaderCallback: (bounds) => LinearGradient(
                  colors: referenceDark
                      ? const [Color(0xFF9FCBFF), Color(0xFFF0B4ED)]
                      : const [Color(0xFF2459A5), Color(0xFF8C358E)])
              .createShader(bounds),
          child: text);
    }
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0, end: 1),
      duration: MediaQuery.disableAnimationsOf(context)
          ? Duration.zero
          : const Duration(milliseconds: 400),
      curve: Curves.easeOutCubic,
      child: text,
      builder: (_, value, child) => Opacity(
          opacity: value,
          child: Transform.translate(
              offset: Offset(0, 10 * (1 - value)), child: child)),
    );
  }

  Future<void> connectionDetails() async {
    final editController =
        TextEditingController(text: api.baseUrl.replaceAll('/api/v1', ''));
    String? testStatus;
    bool isTesting = false;

    await showModalBottomSheet<void>(
        context: context,
        isScrollControlled: true,
        showDragHandle: true,
        builder: (ctx) => StatefulBuilder(
            builder: (ctx, setModalState) => Padding(
                padding: EdgeInsets.fromLTRB(
                    24, 8, 24, MediaQuery.viewInsetsOf(ctx).bottom + 28),
                child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      title('Company connection'),
                      muted(
                          'Bundled addresses can expire. Paste the current address from AURA_CURRENT_URL.txt on your PC. Only a healthy address can be saved.'),
                      muted(workspace.isEmpty
                          ? 'Sign-in connected. Join your company invitation to appear online in AURA HQ.'
                          : 'Workspace status: $connection'),
                      const SizedBox(height: 12),
                      TextField(
                        controller: editController,
                        decoration: const InputDecoration(
                          labelText: 'Server URL',
                          border: OutlineInputBorder(),
                        ),
                      ),
                      const SizedBox(height: 10),
                      Wrap(spacing: 8, runSpacing: 8, children: [
                        ActionChip(
                          avatar: const Icon(Icons.cloud_outlined, size: 16),
                          label: const Text('Use bundled address'),
                          onPressed: () => setModalState(() {
                            editController.text = AuraConfig.apiUrl;
                            testStatus = null;
                          }),
                        ),
                        ActionChip(
                          avatar: const Icon(Icons.wifi, size: 16),
                          label: const Text('Local Wi-Fi (10.24.125.111)'),
                          onPressed: () => setModalState(() {
                            editController.text = 'http://10.24.125.111:8000';
                            testStatus = null;
                          }),
                        ),
                      ]),
                      if (testStatus != null)
                        Padding(
                          padding: const EdgeInsets.symmetric(vertical: 8),
                          child: Text(
                            testStatus!,
                            style: TextStyle(
                              fontSize: 12,
                              fontWeight: FontWeight.w500,
                              color: testStatus!.startsWith('✓')
                                  ? AuraTheme.accentEmerald
                                  : AuraTheme.accentRose,
                            ),
                          ),
                        ),
                      const SizedBox(height: 14),
                      Row(
                        children: [
                          OutlinedButton.icon(
                            icon: const Icon(Icons.network_check_rounded,
                                size: 16),
                            label: const Text('Test'),
                            onPressed: isTesting
                                ? null
                                : () async {
                                    setModalState(() => isTesting = true);
                                    try {
                                      final raw = editController.text
                                          .trim()
                                          .replaceAll(RegExp(r'/+$'), '');
                                      final res = await http
                                          .get(Uri.parse('$raw/health'))
                                          .timeout(const Duration(seconds: 4));
                                      setModalState(() {
                                        isTesting = false;
                                        testStatus = res.statusCode == 200
                                            ? '✓ Server is Online & Healthy'
                                            : '✗ Server returned HTTP ${res.statusCode}';
                                      });
                                    } catch (_) {
                                      setModalState(() {
                                        isTesting = false;
                                        testStatus =
                                            '✗ Unreachable: check URL & network';
                                      });
                                    }
                                  },
                          ),
                          const Spacer(),
                          FilledButton(
                            onPressed: () async {
                              String newHost;
                              try {
                                newHost = await checkCompanyEndpoint(
                                    editController.text);
                              } catch (e) {
                                if (ctx.mounted) {
                                  setModalState(() => testStatus = e
                                      .toString()
                                      .replaceFirst('FormatException: ', ''));
                                }
                                return;
                              }
                              api.baseUrl = '$newHost/api/v1';
                              await secureSession.write(
                                  key: 'endpoint', value: newHost);
                              if (ctx.mounted) Navigator.pop(ctx);
                              connect();
                              await initialize();
                            },
                            child: const Text('Save & Reconnect'),
                          ),
                        ],
                      ),
                    ]))));
  }

  Widget assistantHome() {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      referenceHome(),
      if (ai['configured'] != true)
        Padding(
            padding: const EdgeInsets.only(top: 24),
            child: InkWell(
                onTap: () => changePanel('Settings'),
                borderRadius: BorderRadius.circular(18),
                child: Container(
                    padding: const EdgeInsets.all(18),
                    decoration: BoxDecoration(
                        color: referenceDark
                            ? const Color(0xFF25304B)
                            : const Color(0xFFEDF2FF),
                        borderRadius: BorderRadius.circular(18)),
                    child: const Row(children: [
                      NexaMark(size: 24),
                      SizedBox(width: 12),
                      Expanded(
                          child: Text('Choose an AI provider in Settings',
                              style: TextStyle(fontSize: 13))),
                      Icon(Icons.arrow_forward_rounded, size: 18)
                    ])))),
      const SizedBox(height: 22),
      if (messages.isNotEmpty)
        Align(
            alignment: Alignment.centerRight,
            child: TextButton.icon(
              icon: const Icon(Icons.delete_outline_rounded, size: 18),
              label: const Text('Clear my chat'),
              onPressed: busy
                  ? null
                  : () => action(() async {
                        final confirmed = await showDialog<bool>(
                            context: context,
                            builder: (ctx) => AlertDialog(
                                    title:
                                        const Text('Clear this conversation?'),
                                    content: const Text(
                                        'Your chat in this workspace will be deleted. Meetings and other people chats are kept.'),
                                    actions: [
                                      TextButton(
                                          onPressed: () =>
                                              Navigator.pop(ctx, false),
                                          child: const Text('Keep chat')),
                                      TextButton(
                                          onPressed: () =>
                                              Navigator.pop(ctx, true),
                                          child: const Text('Clear chat'))
                                    ]));
                        if (confirmed != true) return;
                        await api.request(
                            '/workspaces/$workspace/agents/history',
                            method: 'DELETE');
                        if (mounted) redraw(() => messages = []);
                      }),
            )),
      for (final item in messages)
        Padding(
            padding: EdgeInsets.only(
                bottom: 16,
                left: item['role'] == 'user' ? 28 : 0,
                right: item['role'] == 'user' ? 0 : 12),
            child: Container(
                padding: const EdgeInsets.all(18),
                decoration: BoxDecoration(
                    gradient: LinearGradient(
                        colors: referenceDark
                            ? [const Color(0xFF243450), const Color(0xFF1D2941)]
                            : item['role'] == 'user'
                                ? [
                                    const Color(0xFFF0E3FC),
                                    const Color(0xFFECE3FC)
                                  ]
                                : [
                                    const Color(0xFFFFFFFF),
                                    const Color(0xFFF9F5FF)
                                  ]),
                    border: Border.all(color: const Color(0xFFE9E0F5)),
                    borderRadius: BorderRadius.circular(22)),
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(children: [
                        if (item['role'] != 'user') ...[
                          const NexaMark(size: 17),
                          const SizedBox(width: 8)
                        ],
                        Text(item['role'] == 'user' ? 'You' : 'Nexa ✨',
                            style: const TextStyle(
                                color: AuraTheme.textMuted, fontSize: 11))
                      ]),
                      const SizedBox(height: 10),
                      animatedChatText(Map<String, dynamic>.from(item)),
                    ]))),
    ]);
  }

  Widget composer() => Container(
      margin: const EdgeInsets.fromLTRB(16, 4, 16, 12),
      padding: const EdgeInsets.fromLTRB(18, 8, 8, 8),
      decoration: BoxDecoration(
          color: referenceSurface,
          borderRadius: BorderRadius.circular(30),
          border: Border.all(color: referenceBorder)),
      child: Row(crossAxisAlignment: CrossAxisAlignment.end, children: [
        Expanded(
            child: TextField(
                controller: message,
                minLines: 1,
                maxLines: 4,
                textInputAction: TextInputAction.newline,
                decoration: const InputDecoration(
                    hintText: 'Ask Nexa about your team…',
                    hintStyle: TextStyle(color: Color(0xFF9B8CA9)),
                    filled: false,
                    border: InputBorder.none,
                    enabledBorder: InputBorder.none,
                    focusedBorder: InputBorder.none,
                    contentPadding: EdgeInsets.symmetric(vertical: 12)),
                style: TextStyle(fontSize: 13, color: referenceInk))),
        IconButton(
            tooltip: wake.dictating ? 'Finish dictation' : 'Tap to speak',
            onPressed: busy ? null : talkNow,
            icon: Icon(
                wake.listening
                    ? Icons.graphic_eq_rounded
                    : Icons.mic_none_rounded,
                color: wake.enabled
                    ? AuraTheme.accentPurple
                    : AuraTheme.textSecondary)),
        IconButton.filled(
            tooltip: 'Send message',
            onPressed: busy ? null : send,
            icon: const Icon(Icons.arrow_upward_rounded)),
      ]));

  Widget nexaWorkspace(BuildContext context) => Scaffold(
        appBar: AppBar(
            toolbarHeight: 70,
            titleSpacing: 2,
            title: const Row(children: [
              NexaMark(size: 27),
              SizedBox(width: 10),
              Text('Nexa',
                  style: TextStyle(fontWeight: FontWeight.w600, fontSize: 23))
            ]),
            actions: [
              IconButton(
                  tooltip: referenceDark
                      ? 'Switch to light mode'
                      : 'Switch to dark mode',
                  onPressed: () => changeTheme(!referenceDark),
                  icon: Icon(
                      referenceDark
                          ? Icons.light_mode_outlined
                          : Icons.dark_mode_outlined,
                      size: 21)),
              if (speaking)
                IconButton(
                  tooltip: 'Stop speaking',
                  onPressed: stopSpeaking,
                  icon: const Icon(Icons.stop_circle_outlined),
                ),
              TextButton.icon(
                  onPressed: connectionDetails,
                  icon: Icon(Icons.circle,
                      size: 7,
                      color: connection == 'Connected'
                          ? AuraTheme.accentEmerald
                          : serverReachable
                              ? AuraTheme.accentAmber
                              : AuraTheme.accentRose),
                  label: Text(
                      workspace.isEmpty
                          ? 'Join team'
                          : connection == 'Connected'
                              ? 'Connected'
                              : 'Reconnect',
                      style: const TextStyle(fontSize: 11))),
              IconButton(
                  tooltip: 'Your settings',
                  onPressed: () => changePanel('Settings'),
                  icon: CircleAvatar(
                      radius: 16,
                      backgroundColor: const Color(0xFF507FEA),
                      child: Text(
                          (me['full_name'] ?? '?').toString().substring(0, 1),
                          style: const TextStyle(fontSize: 12)))),
              const SizedBox(width: 8),
            ]),
        drawer: Drawer(
            backgroundColor: referenceSurface,
            child: SafeArea(
                child: ListView(
                    padding: const EdgeInsets.symmetric(horizontal: 12),
                    children: [
                  const Padding(
                      padding: EdgeInsets.all(20),
                      child: Row(children: [
                        NexaMark(size: 32),
                        SizedBox(width: 12),
                        Text('Your space',
                            style: TextStyle(
                                fontSize: 23, fontWeight: FontWeight.w600))
                      ])),
                  for (final entry in {
                    'Assistant': Icons.auto_awesome_rounded,
                    'People': Icons.workspaces_outline,
                    'Calendar': Icons.calendar_month_outlined,
                    'Meetings': Icons.event_available_outlined,
                    'Tasks': Icons.check_circle_outline,
                    'Voice studio': Icons.graphic_eq_rounded,
                    'Notifications': Icons.notifications_none_rounded,
                    'Settings': Icons.tune_rounded
                  }.entries)
                    ListTile(
                        leading: Icon(entry.value, size: 21),
                        title: Text(
                            entry.key == 'People' ? 'My company' : entry.key,
                            style: const TextStyle(fontSize: 13)),
                        selected: panel == entry.key,
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(16)),
                        onTap: () {
                          Navigator.pop(context);
                          changePanel(entry.key);
                        }),
                  const Divider(height: 30),
                  ListTile(
                      leading: const Icon(Icons.logout_rounded, size: 21),
                      title: const Text('Sign out'),
                      onTap: () => action(() async {
                            await wake.stop();
                            await api.request('/auth/logout', method: 'POST');
                            await secureSession.delete(key: 'access');
                            await secureSession.delete(key: 'refresh');
                            if (mounted) {
                              Navigator.of(this.context).pushAndRemoveUntil(
                                  MaterialPageRoute(
                                      builder: (_) => const SessionGate()),
                                  (_) => false);
                            }
                          })),
                ]))),
        body: SafeArea(
            child: Column(children: [
          if (busy)
            const LinearProgressIndicator(
                minHeight: 2, backgroundColor: Colors.transparent),
          if (busy && busyLabel != 'Working…')
            Padding(
                padding: const EdgeInsets.all(12),
                child: Text(busyLabel,
                    style: const TextStyle(
                        color: AuraTheme.textSecondary, fontSize: 12))),
          Expanded(
              child: RefreshIndicator(
                  onRefresh: () => action(refresh),
                  child: ListView(
                      padding: const EdgeInsets.fromLTRB(24, 8, 24, 24),
                      children: [
                        if (panel != 'Assistant') referenceHeader(),
                        if (error != null)
                          Container(
                              margin: const EdgeInsets.only(bottom: 16),
                              padding: const EdgeInsets.all(16),
                              decoration: BoxDecoration(
                                  color: const Color(0xFFFFEEF2),
                                  borderRadius: BorderRadius.circular(18)),
                              child: Row(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    const Icon(Icons.info_outline_rounded,
                                        size: 19, color: AuraTheme.accentRose),
                                    const SizedBox(width: 10),
                                    Expanded(
                                        child: Text(error!,
                                            style: const TextStyle(
                                                color: AuraTheme.accentRose,
                                                height: 1.5,
                                                fontSize: 12))),
                                    IconButton(
                                        visualDensity: VisualDensity.compact,
                                        onPressed: () =>
                                            redraw(() => error = null),
                                        icon: const Icon(Icons.close, size: 16))
                                  ])),
                        if (invitations.isNotEmpty && !needsVoice)
                          ...invitations.map((invitation) => card([
                                const Icon(Icons.mail_outline_rounded,
                                    color: AuraTheme.accentPurple),
                                const SizedBox(height: 14),
                                title(
                                    'You’re invited to ${invitation['workspace_name']}'),
                                muted(
                                    'Your company invited ${me['email']}. Join to share availability and work with your team.'),
                                button('Join workspace', () async {
                                  final joined = await api.request(
                                      '/invitations/${invitation['id']}/accept',
                                      method: 'POST');
                                  await loadSpaces();
                                  await selectWorkspace(joined['workspace_id']);
                                  await refreshInvitations();
                                  await changePanel(
                                      (cache['calendars'] as List? ?? [])
                                              .isEmpty
                                          ? 'Calendar'
                                          : 'Assistant');
                                }),
                                TextButton(
                                    onPressed: busy
                                        ? null
                                        : () => action(() async {
                                              await api.request(
                                                  '/invitations/${invitation['id']}/decline',
                                                  method: 'POST');
                                              await refreshInvitations();
                                            }),
                                    child: const Text('Decline invitation')),
                              ])),
                        if (panel == 'Assistant' && workspace.isNotEmpty)
                          assistantHome()
                        else
                          ...content(),
                        if (panel == 'Settings') ...[
                          const SizedBox(height: 12),
                          card([
                            title('Appearance'),
                            ListTile(
                              contentPadding: EdgeInsets.zero,
                              leading: Icon(AuraTheme.themeNotifier.value ==
                                      ThemeMode.dark
                                  ? Icons.dark_mode_rounded
                                  : Icons.light_mode_rounded),
                              title: const Text('Theme Mode',
                                  style: TextStyle(fontSize: 14)),
                              subtitle: Text(
                                AuraTheme.themeNotifier.value == ThemeMode.dark
                                    ? 'Dark Mode (Cosmic Slate)'
                                    : 'Light Mode (Modern Crisp)',
                                style: const TextStyle(fontSize: 11),
                              ),
                              trailing: Switch(
                                value: AuraTheme.themeNotifier.value ==
                                    ThemeMode.dark,
                                onChanged: (isDark) {
                                  unawaited(changeTheme(isDark));
                                  redraw(() {});
                                },
                              ),
                            ),
                          ]),
                          const SizedBox(height: 12),
                          card([
                            title('Voice activation'),
                            muted(
                                'Hey Nexa is detected offline using the bundled sherpa-onnx model. No account, key or model download is needed. After the listening indicator appears, speak your request; final speech is cleaned up and sent automatically.'),
                            SwitchListTile(
                                contentPadding: EdgeInsets.zero,
                                title: const Text('Nexa activation',
                                    style: TextStyle(fontSize: 14)),
                                subtitle: Text(wake.status,
                                    style: const TextStyle(fontSize: 11)),
                                value: wake.enabled,
                                onChanged: (_) => toggleWake()),
                            SwitchListTile(
                                contentPadding: EdgeInsets.zero,
                                title: const Text('Read replies aloud',
                                    style: TextStyle(fontSize: 14)),
                                value: speakReplies,
                                onChanged: (value) =>
                                    redraw(() => speakReplies = value)),
                            const SizedBox(height: 12),
                            DropdownButtonFormField<bool>(
                              initialValue: offlineReplyVoice,
                              decoration: const InputDecoration(
                                  labelText: 'Reply voice'),
                              items: const [
                                DropdownMenuItem(
                                    value: false,
                                    child:
                                        Text('Natural · Chatterbox / phone')),
                                DropdownMenuItem(
                                    value: true,
                                    child: Text('Downloaded · offline')),
                              ],
                              onChanged: speaking
                                  ? null
                                  : (value) async {
                                      if (value == null) return;
                                      redraw(() => offlineReplyVoice = value);
                                      await secureSession.write(
                                          key: 'nexa_reply_voice_offline',
                                          value: value.toString());
                                    },
                            ),
                            const SizedBox(height: 8),
                            muted(
                                'Natural uses NVIDIA Chatterbox when configured, with phone voice as backup. Downloaded uses only your phone. Natural speech uses internet.'),
                            muted(replyVoice.label),
                            const SizedBox(height: 16),
                            DropdownButtonFormField<String>(
                              initialValue: wake.preferredLocale,
                              decoration: const InputDecoration(
                                  labelText: 'Recognition language'),
                              items: const [
                                DropdownMenuItem(
                                    value: 'en_IN',
                                    child: Text('English (India)')),
                                DropdownMenuItem(
                                    value: 'en_US',
                                    child: Text('English (US)')),
                                DropdownMenuItem(
                                    value: 'device',
                                    child: Text('Phone default')),
                              ],
                              onChanged: (value) async {
                                if (value == null) return;
                                await wake.pause();
                                wake.preferredLocale = value;
                                await secureSession.write(
                                    key: 'nexa_voice_locale', value: value);
                                redraw(() {});
                              },
                            ),
                            const SizedBox(height: 16),
                            const Text('Reply speed',
                                style: TextStyle(fontSize: 14)),
                            Slider(
                              value: speechRate,
                              min: .25,
                              max: .65,
                              divisions: 8,
                              label: speechRate < .4 ? 'Gentle' : 'Faster',
                              onChanged: (value) =>
                                  redraw(() => speechRate = value),
                              onChangeEnd: (value) => secureSession.write(
                                  key: 'nexa_voice_rate',
                                  value: value.toString()),
                            ),
                            Row(
                                mainAxisAlignment:
                                    MainAxisAlignment.spaceBetween,
                                children: [muted('Slower'), muted('Faster')]),
                            TextButton.icon(
                              onPressed: speaking
                                  ? stopSpeaking
                                  : () => action(() => speakReply(
                                      'Hi, I’m Nexa. What would you like to plan today? I can help you find a time that works for your team.')),
                              icon: Icon(speaking
                                  ? Icons.stop_rounded
                                  : Icons.volume_up_outlined),
                              label: Text(
                                  speaking ? 'Stop preview' : 'Preview voice'),
                            ),
                            const Divider(height: 28),
                            ListTile(
                              contentPadding: EdgeInsets.zero,
                              leading: Container(
                                padding: const EdgeInsets.all(8),
                                decoration: BoxDecoration(
                                  color: AuraTheme.accentCyan.withValues(alpha: 0.15),
                                  borderRadius: BorderRadius.circular(10),
                                ),
                                child: const Icon(Icons.layers_outlined,
                                    color: AuraTheme.accentCyan, size: 22),
                              ),
                              title: const Text('Display Over Apps & Background Permissions',
                                  style: TextStyle(
                                      fontSize: 14, fontWeight: FontWeight.w600)),
                              subtitle: const Text(
                                  'Allow Nexa to pop open over other apps like Google Assistant, ignore battery sleep, and stay active.',
                                  style: TextStyle(fontSize: 11)),
                              trailing: const Icon(Icons.chevron_right_rounded),
                              onTap: () => NexaPermissionSheet.showIfRequired(
                                  context,
                                  force: true),
                            ),
                          ])
                        ],
                      ]))),
          if (panel == 'Assistant' && workspace.isNotEmpty) composer(),
        ])),
        bottomNavigationBar: Padding(
            padding: const EdgeInsets.fromLTRB(8, 0, 8, 8),
            child: ClipRRect(
                borderRadius: BorderRadius.circular(28),
                child: NavigationBar(
                    backgroundColor: referenceSurface,
                    indicatorColor: referenceDark
                        ? const Color(0xFF354C78)
                        : const Color(0xFFD9E7FF),
                    height: 72,
                    labelBehavior:
                        NavigationDestinationLabelBehavior.alwaysShow,
                    selectedIndex: {
                          'Assistant': 0,
                          'Calendar': 1,
                          'Meetings': 2,
                          'People': 3,
                          'Voice studio': 4,
                        }[panel] ??
                        0,
                    onDestinationSelected: (index) => changePanel([
                          'Assistant',
                          'Calendar',
                          'Meetings',
                          'People',
                          'Voice studio'
                        ][index]),
                    destinations: const [
                      NavigationDestination(
                          icon: Icon(Icons.auto_awesome_outlined),
                          selectedIcon: Icon(Icons.auto_awesome_rounded),
                          label: 'Nexa'),
                      NavigationDestination(
                          icon: Icon(Icons.calendar_month_outlined),
                          selectedIcon: Icon(Icons.calendar_month_rounded),
                          label: 'Calendar'),
                      NavigationDestination(
                          icon: Icon(Icons.videocam_outlined),
                          selectedIcon: Icon(Icons.videocam_rounded),
                          label: 'Meetings'),
                      NavigationDestination(
                          icon: Icon(Icons.groups_outlined),
                          selectedIcon: Icon(Icons.groups_rounded),
                          label: 'Team HQ'),
                      NavigationDestination(
                          icon: Icon(Icons.graphic_eq_outlined),
                          selectedIcon: Icon(Icons.graphic_eq_rounded),
                          label: 'Voice Studio'),
                    ]))),
      );
}
