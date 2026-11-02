part of 'connected_workspace.dart';

const _violet = Color(0xFF5488F6);

class _CalendarArtwork extends StatelessWidget {
  final IconData icon;
  const _CalendarArtwork(this.icon);
  @override
  Widget build(BuildContext context) => SizedBox(
      width: 112,
      height: 138,
      child: Stack(alignment: Alignment.center, children: [
        Transform.rotate(
            angle: -.22,
            child: Container(
                width: 108,
                height: 95,
                decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(28),
                    border:
                        Border.all(color: const Color(0xFFD5B4F9), width: 2)))),
        Transform.rotate(
            angle: .16,
            child: Container(
                width: 84,
                height: 106,
                decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(22),
                    gradient: const LinearGradient(
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                        colors: [
                          Color(0xFFF4DEFF),
                          Color(0xFFB08AF5),
                          Color(0xFF7742D9)
                        ]),
                    boxShadow: const [
                      BoxShadow(
                          color: Color(0x338E45D9),
                          blurRadius: 20,
                          offset: Offset(2, 12))
                    ]),
                child: Column(children: [
                  const SizedBox(height: 10),
                  const Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        Icon(Icons.horizontal_rule_rounded, color: _violet),
                        Icon(Icons.horizontal_rule_rounded, color: _violet)
                      ]),
                  Container(
                      margin: const EdgeInsets.all(8),
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                          color: const Color(0xFFFAF3FF),
                          borderRadius: BorderRadius.circular(13)),
                      child: Icon(icon, size: 30, color: _violet))
                ]))),
        const Positioned(
            top: 3,
            right: 2,
            child: Icon(Icons.auto_awesome, size: 18, color: Color(0xFFC572EA)))
      ]));
}

extension _NexaReferenceUI on _ConnectedWorkspaceState {
  bool get referenceDark => Theme.of(context).brightness == Brightness.dark;
  Color get referenceSurface => Theme.of(context).colorScheme.surface;
  Color get referenceInk => Theme.of(context).colorScheme.onSurface;
  Color get referenceBorder =>
      referenceDark ? const Color(0xFF303B57) : const Color(0xFFDDE6FA);
  Future<void> changeTheme(bool dark) async {
    AuraTheme.themeNotifier.value = dark ? ThemeMode.dark : ThemeMode.light;
    await secureSession.write(
        key: 'nexa_theme', value: dark ? 'dark' : 'light');
  }

  Widget referenceTabs(List<String> values, String selected,
          ValueChanged<String> onChange) =>
      Container(
          padding: const EdgeInsets.all(4),
          decoration: BoxDecoration(
              color: referenceDark
                  ? const Color(0xFF212B43)
                  : const Color(0xFFEEF2FC),
              borderRadius: BorderRadius.circular(28)),
          child: Row(children: [
            for (final value in values)
              Expanded(
                  child: Semantics(
                      selected: value == selected,
                      child: InkWell(
                          borderRadius: BorderRadius.circular(24),
                          onTap: () => onChange(value),
                          child: AnimatedContainer(
                              duration: const Duration(milliseconds: 180),
                              padding: const EdgeInsets.symmetric(vertical: 12),
                              decoration: BoxDecoration(
                                  borderRadius: BorderRadius.circular(24),
                                  gradient: value == selected
                                      ? const LinearGradient(colors: [
                                          Color(0xFF7FB7FF),
                                          Color(0xFF486CE3)
                                        ])
                                      : null),
                              child: Text(value,
                                  textAlign: TextAlign.center,
                                  style: TextStyle(
                                      fontSize: 10,
                                      fontWeight: FontWeight.w600,
                                      color: value == selected
                                          ? Colors.white
                                          : AuraTheme.textSecondary))))))
          ]));

  List<dynamic> filteredMeetings() => workflows.where((w) {
        final end = DateTime.tryParse('${w['current_revision']?['end_time']}');
        final past = end != null && end.isBefore(DateTime.now());
        if (meetingFilter == 'Past') return past;
        if (meetingFilter == 'Pending') {
          return !past && ['AWAITING_APPROVALS', 'FAILED'].contains(w['state']);
        }
        return !past;
      }).toList();

  Widget _calendarMeetingCard(Map<String, dynamic> meeting) {
    final isConfirmed = meeting['state'] == 'COMPLETED';
    final isPending = meeting['state'] == 'AWAITING_APPROVALS';
    final accentColor = isConfirmed
        ? AuraTheme.accentEmerald
        : (isPending ? const Color(0xFFE5A93C) : _violet);

    final startTimeStr = clockTimeIST(meeting['start']);
    final endTimeStr = clockTimeIST(meeting['end']);
    final duration = meeting['duration'];
    final htmlLink = meeting['html_link'] as String?;

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: referenceDark ? const Color(0xFF222D45) : Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: referenceDark
              ? const Color(0xFF2B3A55)
              : const Color(0xFFE4E9F2),
          width: 1,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: referenceDark ? 0.2 : 0.04),
            blurRadius: 6,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                width: 3.5,
                height: 36,
                decoration: BoxDecoration(
                  color: accentColor,
                  borderRadius: BorderRadius.circular(4),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      meeting['title']?.toString() ?? 'Scheduled Meeting',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w700,
                        letterSpacing: -0.2,
                      ),
                    ),
                    const SizedBox(height: 3),
                    Row(
                      children: [
                        const Icon(Icons.access_time_rounded,
                            size: 12, color: AuraTheme.textSecondary),
                        const SizedBox(width: 4),
                        Expanded(
                          child: Text(
                            '$startTimeStr – $endTimeStr IST${duration != null && duration > 0 ? " · $duration min" : ""}',
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(
                              fontSize: 11,
                              fontWeight: FontWeight.w600,
                              color: _violet,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 3),
                decoration: BoxDecoration(
                  color: accentColor.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      isConfirmed
                          ? Icons.check_circle_rounded
                          : (isPending
                              ? Icons.hourglass_top_rounded
                              : Icons.event_available_rounded),
                      size: 10,
                      color: accentColor,
                    ),
                    const SizedBox(width: 3),
                    Text(
                      isConfirmed
                          ? 'Confirmed'
                          : (isPending
                              ? 'Pending'
                              : meeting['state']
                                  .toString()
                                  .replaceAll('_', ' ')),
                      style: TextStyle(
                        fontSize: 9,
                        fontWeight: FontWeight.w700,
                        color: accentColor,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          if (htmlLink != null && htmlLink.isNotEmpty) ...[
            const SizedBox(height: 8),
            const Divider(height: 1, color: Color(0x1F7F8C9E)),
            const SizedBox(height: 6),
            InkWell(
              onTap: () => launchUrl(Uri.parse(htmlLink),
                  mode: LaunchMode.externalApplication),
              borderRadius: BorderRadius.circular(8),
              child: const Padding(
                padding: EdgeInsets.symmetric(vertical: 2, horizontal: 2),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.calendar_month_rounded,
                        size: 13, color: _violet),
                    SizedBox(width: 5),
                    Text(
                      'Open in Google Calendar',
                      style: TextStyle(
                        fontSize: 10.5,
                        fontWeight: FontWeight.w600,
                        color: _violet,
                      ),
                    ),
                    SizedBox(width: 4),
                    Icon(Icons.open_in_new_rounded,
                        size: 11, color: _violet),
                  ],
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }

  List<Widget> calendarTimeline() {
    final nowIST = toIST(DateTime.now());
    final todayIST = DateTime(nowIST.year, nowIST.month, nowIST.day);

    DateTime fromDate;
    DateTime untilDate;

    if (calendarFilter == 'Week') {
      fromDate = todayIST.subtract(Duration(days: nowIST.weekday - 1));
      untilDate = fromDate.add(const Duration(days: 7));
    } else if (calendarFilter == 'Month') {
      fromDate = DateTime(nowIST.year, nowIST.month, 1);
      untilDate = DateTime(nowIST.year, nowIST.month + 1, 1);
    } else {
      // Timeline mode: show comprehensive upcoming and recent schedule
      fromDate = todayIST.subtract(const Duration(days: 7));
      untilDate = todayIST.add(const Duration(days: 90));
    }

    // 1. Collect real scheduled meetings from workflows
    final scheduledMeetings = <Map<String, dynamic>>[];
    for (final w in workflows) {
      final rev = w['current_revision'];
      final cal = w['calendar_event'];
      final startStr = rev?['start_time'] ?? cal?['start_time'];
      if (startStr != null) {
        final startIST = toIST(startStr);
        final endStr = rev?['end_time'] ?? cal?['end_time'];
        final durationMinutes = rev?['duration_minutes'] ??
            (endStr != null
                ? toIST(endStr).difference(startIST).inMinutes
                : 30);
        final endIST = endStr != null
            ? toIST(endStr)
            : startIST.add(Duration(minutes: durationMinutes));

        scheduledMeetings.add({
          'id': w['id'] ?? cal?['id'] ?? '${w['title']}_$startStr',
          'title': w['title'] ?? cal?['summary'] ?? 'Scheduled meeting',
          'start': startIST,
          'end': endIST,
          'duration': durationMinutes,
          'state': w['state'] ?? (cal != null ? 'COMPLETED' : 'SCHEDULED'),
          'html_link': cal?['html_link'],
          'is_synced': cal != null,
          'approvals': rev?['approvals'] ?? [],
        });
      }
    }

    // Also include any named external events from cache['events'] (if summary exists)
    for (final e in (cache['events'] as List? ?? [])) {
      final summary = e['summary'] ?? e['title'];
      if (summary != null && summary.toString().trim().isNotEmpty) {
        final startIST = toIST(e['start']);
        final endIST = toIST(e['end']);
        scheduledMeetings.add({
          'id': e['id'] ?? 'gcal_${e['start']}',
          'title': summary.toString(),
          'start': startIST,
          'end': endIST,
          'duration': endIST.difference(startIST).inMinutes,
          'state': 'COMPLETED',
          'html_link': e['html_link'],
          'is_synced': true,
          'approvals': const [],
        });
      }
    }

    // Filter by calendar tab date range
    final filteredMeetings = scheduledMeetings.where((m) {
      final start = m['start'] as DateTime;
      if (calendarFilter == 'Timeline') return true;
      return !start.isBefore(fromDate) && start.isBefore(untilDate);
    }).toList();

    // Deduplicate meetings
    final seen = <String>{};
    final deduplicatedMeetings = <Map<String, dynamic>>[];
    for (final m in filteredMeetings) {
      final key = '${m['title']}_${(m['start'] as DateTime).millisecondsSinceEpoch}';
      if (seen.add(key)) {
        deduplicatedMeetings.add(m);
      }
    }
    deduplicatedMeetings.sort((a, b) =>
        (a['start'] as DateTime).compareTo(b['start'] as DateTime));

    // Group meetings by IST date header
    final groups = <String, List<Map<String, dynamic>>>{};
    for (final m in deduplicatedMeetings) {
      final label = dateHeaderIST(m['start']);
      groups.putIfAbsent(label, () => []).add(m);
    }

    // 2. Collect raw Google Calendar busy blocks (preventing timeline spam)
    final rawBusySlots = (cache['events'] as List? ?? []).where((e) {
      final hasSummary = (e['summary'] ?? e['title'])?.toString().trim().isNotEmpty == true;
      return !hasSummary && e['busy'] == true && e['start'] != null && e['end'] != null;
    }).toList();

    final filteredBusy = rawBusySlots.where((e) {
      final startIST = toIST(e['start']);
      if (calendarFilter == 'Timeline') return true;
      return !startIST.isBefore(fromDate) && startIST.isBefore(untilDate);
    }).toList()
      ..sort((a, b) => toIST(a['start']).compareTo(toIST(b['start'])));

    // Clean, collapsible tile for raw Google Calendar busy blocks
    Widget? busyExpansionTile;
    if (filteredBusy.isNotEmpty) {
      busyExpansionTile = Container(
        margin: const EdgeInsets.only(top: 8, bottom: 12),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
            color: referenceDark
                ? const Color(0xFF2B3A52)
                : const Color(0xFFE2E7F0),
          ),
        ),
        child: Material(
          color: referenceDark
              ? const Color(0xFF1B2333)
              : const Color(0xFFF3F6FC),
          borderRadius: BorderRadius.circular(14),
          child: Theme(
            data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
            child: ExpansionTile(
            tilePadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 0),
            initiallyExpanded: groups.isEmpty,
            leading: const Icon(Icons.sync_outlined, color: _violet, size: 18),
            title: Text(
              'Google Calendar Busy Intervals (${filteredBusy.length})',
              style: const TextStyle(fontSize: 12.5, fontWeight: FontWeight.w600),
            ),
            subtitle: const Text(
              'Synced availability blocks from connected Google account (IST)',
              style: TextStyle(fontSize: 10.5, color: AuraTheme.textSecondary),
            ),
            children: [
              for (final slot in filteredBusy)
                Container(
                  margin: const EdgeInsets.symmetric(horizontal: 10, vertical: 3),
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
                  decoration: BoxDecoration(
                    color: referenceDark ? const Color(0xFF222D45) : Colors.white,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Row(
                    children: [
                      Container(
                        width: 3,
                        height: 24,
                        decoration: BoxDecoration(
                          color: const Color(0xFFB17825),
                          borderRadius: BorderRadius.circular(2),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              '${clockTimeIST(slot['start'])} – ${clockTimeIST(slot['end'])} IST',
                              style: const TextStyle(
                                  fontSize: 11, fontWeight: FontWeight.w600),
                            ),
                            const SizedBox(height: 1),
                            Text(
                              dateHeaderIST(slot['start']),
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(
                                  fontSize: 9.5, color: AuraTheme.textSecondary),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(width: 6),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 6, vertical: 2),
                        decoration: BoxDecoration(
                          color: const Color(0xFFB17825).withValues(alpha: 0.15),
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: const Text('Busy',
                            style: TextStyle(
                                fontSize: 9,
                                fontWeight: FontWeight.w600,
                                color: Color(0xFFB17825))),
                      ),
                    ],
                  ),
                ),
              const SizedBox(height: 6),
            ],
          ),
        ),
        ),
      );
    }

    if (groups.isEmpty) {
      final emptyCard = Container(
        margin: const EdgeInsets.symmetric(vertical: 6),
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color:
              referenceDark ? const Color(0xFF222D45) : const Color(0xFFF7F9FD),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
            color: referenceDark
                ? const Color(0xFF2B3A55)
                : const Color(0xFFE2E7F0),
          ),
        ),
        child: Column(
          children: [
            const Icon(Icons.event_available_outlined,
                size: 32, color: _violet),
            const SizedBox(height: 8),
            const Text(
              'No scheduled meetings in this view',
              style: TextStyle(fontSize: 13.5, fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 4),
            const Text(
              'Your calendar has no active meeting proposals or scheduled events for this filter. All times are displayed in India Standard Time (IST · UTC+05:30).',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 11,
                color: AuraTheme.textSecondary,
                height: 1.4,
              ),
            ),
            const SizedBox(height: 10),
            OutlinedButton.icon(
              onPressed: () => changePanel('Assistant'),
              icon: const Icon(Icons.add_rounded, size: 15),
              label: const Text('Schedule with Nexa',
                  style: TextStyle(fontSize: 11.5)),
            ),
          ],
        ),
      );

      return [
        emptyCard,
        if (busyExpansionTile != null) busyExpansionTile,
      ];
    }

    return [
      for (final group in groups.entries) ...[
        Padding(
          padding: const EdgeInsets.only(top: 6, bottom: 6),
          child: Row(
            children: [
              const Icon(Icons.calendar_today_rounded,
                  size: 13, color: _violet),
              const SizedBox(width: 6),
              Expanded(
                child: Text(
                  group.key,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontSize: 11.5,
                    fontWeight: FontWeight.w700,
                    letterSpacing: -0.2,
                  ),
                ),
              ),
              const SizedBox(width: 6),
              Text(
                '${group.value.length} ${group.value.length == 1 ? "meeting" : "meetings"}',
                style: const TextStyle(
                  fontSize: 10,
                  fontWeight: FontWeight.w600,
                  color: AuraTheme.textSecondary,
                ),
              ),
            ],
          ),
        ),
        for (final meeting in group.value) _calendarMeetingCard(meeting),
      ],
      if (busyExpansionTile != null) busyExpansionTile,
    ];
  }

  Widget pageHero(String eyebrow, String heading, String accent,
          String description, IconData icon) =>
      Padding(
          padding: const EdgeInsets.only(top: 18, bottom: 24),
          child: Row(children: [
            Expanded(
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                  Text(eyebrow.toUpperCase(),
                      style: const TextStyle(
                          fontSize: 10,
                          letterSpacing: 2.5,
                          color: _violet,
                          fontWeight: FontWeight.w600)),
                  const SizedBox(height: 14),
                  Text(heading,
                      style: const TextStyle(
                          fontSize: 27,
                          height: 1.15,
                          letterSpacing: -1,
                          fontWeight: FontWeight.w600)),
                  Text(accent,
                      style: const TextStyle(
                          fontSize: 27,
                          height: 1.2,
                          letterSpacing: -1,
                          fontWeight: FontWeight.w600,
                          color: _violet)),
                  const SizedBox(height: 12),
                  Text(description,
                      style: const TextStyle(
                          color: AuraTheme.textSecondary,
                          fontSize: 12,
                          height: 1.6)),
                ])),
            const SizedBox(width: 6),
            _CalendarArtwork(icon)
          ]));

  Widget referenceHeader() {
    final data = <String, List<dynamic>>{
      'Calendar': [
        'Your calendar,',
        (cache['calendars'] as List? ?? []).isEmpty
            ? 'your choice.'
            : 'connected.',
        'Share your availability and make time for your team.',
        Icons.calendar_month_rounded
      ],
      'Meetings': [
        'Plan, prepare &',
        'meet.',
        'Your meetings, invitations and approvals in one place.',
        Icons.videocam_rounded
      ],
      'People': [
        'Good people.',
        'Great company.',
        'A shared space for your team and their personal assistants.',
        Icons.groups_rounded
      ],
      'Voice studio': [
        'A voice that’s',
        'uniquely yours.',
        'Set up your voice profile in a quiet place.',
        Icons.graphic_eq_rounded
      ],
      'Settings': [
        'Make Nexa',
        'your own.',
        'Your preferences, connections and personal AI.',
        Icons.tune_rounded
      ],
      'Tasks': [
        'A little focus.',
        'A lot of progress.',
        'Keep the work that matters in view.',
        Icons.task_alt_rounded
      ],
      'Notifications': [
        'Stay in',
        'the loop.',
        'Invitations and updates meant for you.',
        Icons.notifications_none_rounded
      ],
      'Negotiations': [
        'Make room',
        'together.',
        'Explore a new time with everyone’s permission.',
        Icons.handshake_outlined
      ],
    };
    final item = data[panel] ??
        [panel, '', 'Your workspace, with you in control.', Icons.auto_awesome];
    return pageHero(panel == 'People' ? 'Team HQ' : panel, item[0], item[1],
        item[2], item[3]);
  }

  Widget referenceHome() {
    final name = (me['full_name'] ?? 'there').toString().split(' ').first;
    final hour = toIST(DateTime.now()).hour;
    final greeting = hour < 12
        ? 'Good morning,'
        : hour < 17
            ? 'Good afternoon,'
            : 'Good evening,';
    final upcoming = workflows.where((w) {
      final start =
          DateTime.tryParse('${w['current_revision']?['start_time']}');
      return w['state'] == 'COMPLETED' &&
          start != null &&
          start.isAfter(DateTime.now());
    }).toList()
      ..sort((a, b) => '${a['current_revision']['start_time']}'
          .compareTo('${b['current_revision']['start_time']}'));
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      const SizedBox(height: 20),
      const Text('YOUR AI WORKSPACE',
          style: TextStyle(
              fontSize: 9, letterSpacing: 2.5, fontWeight: FontWeight.w600)),
      const SizedBox(height: 12),
      Row(children: [
        Expanded(
            flex: 6,
            child:
                Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(greeting,
                  style: const TextStyle(
                      fontSize: 25,
                      fontWeight: FontWeight.w600,
                      letterSpacing: -.8)),
              Text(name,
                  style: const TextStyle(
                      fontSize: 30,
                      color: _violet,
                      fontWeight: FontWeight.w600,
                      letterSpacing: -.8)),
              const SizedBox(height: 12),
              const Text(
                  'Your voice-first assistant for meetings, schedules and teamwork.',
                  style: TextStyle(
                      fontSize: 12,
                      height: 1.6,
                      color: AuraTheme.textSecondary)),
            ])),
        Expanded(
            flex: 5,
            child: Image.asset('assets/illustrations/nexa-companion.png',
                height: 170, fit: BoxFit.contain))
      ]),
      const SizedBox(height: 22),
      Container(
          width: double.infinity,
          padding: const EdgeInsets.fromLTRB(18, 18, 18, 8),
          child: Column(children: [
            Center(
                child: Semantics(
                    button: true,
                    label: 'Talk with Nexa',
                    child: InkWell(
                        onTap: busy || speaking ? null : talkNow,
                        customBorder: const CircleBorder(),
                        child: NexaOrb(
                            size: (MediaQuery.sizeOf(context).width - 90)
                                .clamp(190.0, 300.0),
                            listening: wake.listening,
                            speaking: speaking,
                            thinking: busy && !speaking)))),
            const SizedBox(height: 18),
            Text(
                speaking
                    ? 'Nexa is speaking'
                    : wake.dictating
                        ? 'Listening to you'
                        : 'Tap to talk with Nexa',
                style:
                    const TextStyle(fontWeight: FontWeight.w600, fontSize: 19)),
            const SizedBox(height: 6),
            Text(
                wake.reviewing
                    ? wake.status
                    : busy
                        ? 'Working on your request…'
                        : wake.enabled
                            ? wake.status
                            : 'Voice activation is off',
                textAlign: TextAlign.center,
                style: const TextStyle(
                    fontSize: 10, color: AuraTheme.textSecondary)),
            TextButton.icon(
                onPressed: busy ? null : toggleWake,
                icon: Icon(
                    wake.enabled ? Icons.hearing : Icons.hearing_disabled,
                    size: 14),
                label: Text(
                    wake.enabled ? 'Turn off “Hey Nexa”' : 'Enable “Hey Nexa”',
                    style: const TextStyle(fontSize: 10))),
          ])),
      const SizedBox(height: 22),
      Row(children: [
        for (final item in <(String, String, IconData)>[
          ('Calendar', 'Calendar', Icons.calendar_month_outlined),
          ('Meetings', 'Meetings', Icons.videocam_outlined),
          ('People', 'Team HQ', Icons.groups_outlined),
          ('Voice studio', 'Voice Studio', Icons.graphic_eq)
        ])
          Expanded(
              child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 3),
                  child: Material(
                      color: referenceSurface,
                      borderRadius: BorderRadius.circular(22),
                      child: InkWell(
                          borderRadius: BorderRadius.circular(22),
                          onTap: () => changePanel(item.$1),
                          child: Padding(
                              padding: const EdgeInsets.symmetric(vertical: 18),
                              child: Column(children: [
                                Icon(item.$3, color: _violet, size: 28),
                                const SizedBox(height: 10),
                                Text(item.$2,
                                    style: const TextStyle(
                                        fontSize: 9,
                                        fontWeight: FontWeight.w600))
                              ]))))))
      ]),
      const SizedBox(height: 22),
      card([
        Row(children: [
          const Icon(Icons.event_available_outlined, color: _violet, size: 20),
          const SizedBox(width: 8),
          const Text('Next meeting',
              style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
          const Spacer(),
          TextButton(
              onPressed: () => changePanel('Meetings'),
              child: const Text('See all', style: TextStyle(fontSize: 10)))
        ]),
        title(upcoming.isEmpty
            ? 'Space for something good.'
            : upcoming.first['title']),
        muted(upcoming.isEmpty
            ? 'No confirmed upcoming meetings. Ask Nexa to find a time.'
            : time(upcoming.first['current_revision']['start_time']))
      ]),
      const Text('Try saying…',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
      const SizedBox(height: 12),
      Wrap(spacing: 8, runSpacing: 8, children: [
        for (final text in [
          'Schedule a meeting',
          'Find a team time',
          'Show my tasks'
        ])
          ActionChip(
              label: Text(text, style: const TextStyle(fontSize: 10)),
              avatar: const Icon(Icons.auto_awesome, size: 15, color: _violet),
              onPressed: () {
                message.text = text;
              })
      ]),
    ]);
  }
}
