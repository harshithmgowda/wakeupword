# data/

Metadata about a corpus of real recordings — **no audio**. The recordings
themselves are of identifiable people, are not redistributable, and live outside
the checkout (see [`repo_paths.py`](../repo_paths.py) and `OWW_DATA_ROOT`).

| | |
|---|---|
| `labels.tsv` | per-clip labels, produced by `tools/recordings/label_clips.sh` |
| `split_train.txt`, `split_test.txt` | the train/held-out split, from `tools/recordings/make_split.py` |
| `conditions.txt` | acoustic condition per clip, from `tools/recordings/classify_conditions.py` |
| `real_music_ground_truth.tsv` | which clips have the wake word spoken over music, and which have the music *louder* than the wake word — verified by ear |
| `v7_music_split.tsv` | a worked example of the `only:` subset format `inject_real_clips.py` accepts |

Speakers are referred to by neutral IDs (`speaker_a_07.wav`,
`speaker_b_34.wav`). If your own audio files carry different prefixes,
`tools/recordings/rename_speakers.py` bridges the two without ever writing a real
name into the repo.

Some comments in these files cite scripts and evaluation JSON that document how a
particular row was decided (`make_v7_split.py`, `pick_threshold.py`,
`eval_results/*.json`). Those are on the
[`experiments` branch](https://github.com/Dethon/openwakeword-training/tree/experiments)
along with the rest of this project's research record; the citations are left
intact so the provenance stays traceable. `git show experiments:<path>` reads any
of them without switching branches.
