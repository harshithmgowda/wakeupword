#!/usr/bin/env bash
# Record wake-word clips on the Raspberry Pi, in the format openWakeWord expects.
#
#   ./record.sh speaker_a 25            # 25 clips -> <root>/incoming/speaker_a_01.wav ...
#   ./record.sh speaker_a 25 far        # -> <root>/incoming/speaker_a_far_01.wav ...
#   ./record.sh speaker_a 25 normal plughw:2,0   # explicit capture device
#
# Clips land in incoming/, not train/ or eval/. A fresh recording has no role
# yet, and train/ and eval/ are the record of which clips model v5 was fitted
# on versus held out from -- writing into either would quietly rewrite that.
# Promote by hand once you have decided.
#
# Format is fixed at 16 kHz / mono / 16-bit PCM and 3.0 s per clip, matching the
# existing set so the two are directly comparable. Do not change the duration:
# the detector scores a clip with a sliding 1.28 s window and takes the best
# hit, so a 3 s clip gives it 13 chances to line up and a 2 s clip gives it 1.
set -uo pipefail

SPEAKER="${1:?usage: $0 <speaker> [count] [tag] [device]}"
COUNT="${2:-25}"
TAG="${3:-}"
DEVICE="${4:-}"

SR=16000
DURATION=3
DATA_ROOT="${OWW_DATA_ROOT:-$HOME/openwakeword-train-data}"
OUT="$DATA_ROOT/incoming"
mkdir -p "$OUT"

# Pick a capture device if one wasn't given: first card that reports capture.
if [ -z "$DEVICE" ]; then
    card=$(arecord -l 2>/dev/null | sed -n 's/^card \([0-9]\+\).*device \([0-9]\+\).*/\1,\2/p' | head -1)
    if [ -z "$card" ]; then
        echo "No capture device found. Plug in the mic and check: arecord -l"
        exit 1
    fi
    DEVICE="plughw:${card}"
fi

prefix="$SPEAKER"
[ -n "$TAG" ] && prefix="${SPEAKER}_${TAG}"

echo "device   : $DEVICE"
echo "format   : ${SR} Hz mono 16-bit, ${DURATION}s per clip"
echo "output   : $OUT/${prefix}_NN.wav"
echo "clips    : $COUNT"
echo
echo "Say the wake word ONCE per clip, starting right after the beep-ish prompt."
echo "Leave a moment of silence before and after -- do not rush the start."
echo "Ctrl-C to stop early; already-recorded clips are kept."
echo

# resume: continue numbering after whatever is already there
start=1
while [ -f "$(printf '%s/%s_%02d.wav' "$OUT" "$prefix" "$start")" ]; do
    start=$((start + 1))
done
[ "$start" -gt 1 ] && echo "Existing clips found, continuing from $start."

for i in $(seq "$start" $((start + COUNT - 1))); do
    f=$(printf '%s/%s_%02d.wav' "$OUT" "$prefix" "$i")
    printf '\r\033[K[%d/%d] get ready... ' "$((i - start + 1))" "$COUNT"
    sleep 0.7
    printf '\r\033[K[%d/%d] SPEAK NOW -> %s' "$((i - start + 1))" "$COUNT" "$(basename "$f")"
    arecord -D "$DEVICE" -f S16_LE -r "$SR" -c 1 -d "$DURATION" -q "$f" 2>/dev/null
    if [ ! -s "$f" ]; then
        printf '\r\033[K recording failed -- check the device with: arecord -l\n'
        rm -f "$f"
        exit 1
    fi
    # crude level check so a dead mic or muted input is caught immediately
    peak=$(od -An -td2 -j44 -N8000 "$f" 2>/dev/null | tr -s ' ' '\n' | grep -v '^$' \
           | awk '{v=$1<0?-$1:$1; if(v>m)m=v} END{print m+0}')
    if [ "${peak:-0}" -lt 300 ]; then
        printf '\r\033[K[%d/%d] %s  WARNING: almost silent (peak %s)\n' \
               "$((i - start + 1))" "$COUNT" "$(basename "$f")" "$peak"
    else
        printf '\r\033[K[%d/%d] %s  ok (peak %s)\n' \
               "$((i - start + 1))" "$COUNT" "$(basename "$f")" "$peak"
    fi
    sleep 0.4
done

echo
echo "Done. $(ls "$OUT" | wc -l) clips total in $OUT"
echo "Verify one with:  soxi $OUT/${prefix}_01.wav   (expect 16000 Hz, 1 ch, 3.00s)"
