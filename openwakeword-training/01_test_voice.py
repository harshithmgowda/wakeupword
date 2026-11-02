#!/usr/bin/env python
"""Step 1: generate a single sample of the wake word so you can check pronunciation.

Writes test_generation.wav in this directory. Listen to it before running the
full pipeline -- if it sounds wrong, respell the word phonetically with
underscores between parts ("hey siri" -> "hey_seer_e") and re-run.

    ./.venv/bin/python 01_test_voice.py                  # uses TARGET_WORD below
    ./.venv/bin/python 01_test_voice.py "some_other_werd"  # try a variant
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE / "piper-sample-generator"))

TARGET_WORD = "ouh_ke-i_nahbu"

from generate_samples import generate_samples  # noqa: E402

word = sys.argv[1] if len(sys.argv) > 1 else TARGET_WORD
out = HERE / "test_generation.wav"

generate_samples(
    text=word,
    max_samples=1,
    length_scales=[1.1],
    noise_scales=[0.7],
    noise_scale_ws=[0.7],
    output_dir=str(HERE),
    batch_size=1,
    auto_reduce_batch_size=True,
    file_names=["test_generation.wav"],
)

print(f"\nWrote {out}")
# Colab used IPython.display.Audio; locally just hand it to a player if present.
for player in ("paplay", "aplay", "ffplay"):
    if subprocess.run(["which", player], capture_output=True).returncode == 0:
        cmd = [player, str(out)]
        if player == "ffplay":
            cmd = [player, "-nodisp", "-autoexit", "-loglevel", "quiet", str(out)]
        print(f"Playing with {player}...")
        subprocess.run(cmd)
        break
else:
    print("No audio player found -- open the file manually to listen.")
