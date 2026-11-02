#!/usr/bin/env bash
# Play recordings one at a time and label them by ear.
#
#   ./tools/recordings/label_clips.sh                       # all speaker_a clips in train/ + eval/
#   ./tools/recordings/label_clips.sh <some-dir> 'speaker_b_*'  # a different set
#
# Labels append to labels.tsv. Re-running skips anything already labelled,
# so you can stop with q and pick up later.
set -uo pipefail
# every path below is relative to the repo root, so go there regardless of
# where this script was invoked from
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${OWW_DATA_ROOT:-$HOME/openwakeword-train-data}"

# Default pool is the 133 original recordings: train/ + eval/. unused/ is left
# out deliberately -- rec_trimmed/ and rec_trimmed_25/ are re-cuts of clips that
# are already in train/, so including them would queue each one three times over.
if [ -n "${1:-}" ]; then
    DIRS=("$1")
else
    DIRS=()
    for d in "$DATA_ROOT/train" "$DATA_ROOT/eval"; do [ -d "$d" ] && DIRS+=("$d"); done
    [ ${#DIRS[@]} -eq 0 ] && { echo "No train/ or eval/ under $DATA_ROOT"; exit 1; }
fi
PATTERN="${2:-speaker_a_*.wav}"
OUT="$REPO_ROOT/data/labels.tsv"

for p in paplay aplay ffplay; do command -v "$p" >/dev/null && PLAYER=$p && break; done
[ -z "${PLAYER:-}" ] && { echo "No audio player found (tried paplay, aplay, ffplay)"; exit 1; }

play() {
    case "$PLAYER" in
        ffplay) ffplay -nodisp -autoexit -loglevel quiet "$1" ;;
        *)      "$PLAYER" "$1" 2>/dev/null ;;
    esac
}

touch "$OUT"
# Sort on the basename, not the full path, so clips stay in recording order
# rather than being grouped by whichever directory they ended up in.
mapfile -t FILES < <(find "${DIRS[@]}" -maxdepth 1 -name "$PATTERN" -printf '%f\t%p\n' | sort | cut -f2)
[ ${#FILES[@]} -eq 0 ] && { echo "No files matching $PATTERN in ${DIRS[*]}"; exit 1; }

cat <<'EOF'
Keys:  m = music    w = whisper    f = far away    n = normal
       r = replay   s = skip       q = quit and save
EOF
echo "Player: $PLAYER | $OUT"
echo

done_count=0
for f in "${FILES[@]}"; do
    name=$(basename "$f" .wav)
    grep -q "^${name}	" "$OUT" 2>/dev/null && continue

    while true; do
        printf '\r\033[K[%d/%d] %-14s playing...' "$((done_count + 1))" "${#FILES[@]}" "$name"
        play "$f"
        printf '\r\033[K[%d/%d] %-14s [m/w/f/n/r/s/q]: ' "$((done_count + 1))" "${#FILES[@]}" "$name"
        read -r key </dev/tty
        case "$key" in
            m) echo -e "${name}\tmusic"   >> "$OUT"; break ;;
            w) echo -e "${name}\twhisper" >> "$OUT"; break ;;
            f) echo -e "${name}\tfar"     >> "$OUT"; break ;;
            n) echo -e "${name}\tnormal"  >> "$OUT"; break ;;
            s) break ;;
            q) echo; echo "Saved to $OUT"; exit 0 ;;
            r) ;;                       # loop replays
            *) ;;
        esac
    done
    done_count=$((done_count + 1))
done

echo
echo "Done. Tally:"
cut -f2 "$OUT" | sort | uniq -c | sort -rn
echo
echo "Saved to $OUT"
