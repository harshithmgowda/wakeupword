# Notices and attributions

The code in this repository is [MIT licensed](LICENSE). What that covers, and
what it does not, is worth stating precisely — this project pulls in third-party
code, third-party model weights, and training data with its own restrictions.

## Third-party code

Neither of these is vendored into this repository. `setup_env.sh` clones both at
build time, and they are gitignored. This repo carries only *diffs* against them,
in [`patches/`](patches/README.md).

| project | license | how it is used |
|---|---|---|
| [openWakeWord](https://github.com/dscripka/openWakeWord) (David Scripka) | Apache-2.0 | the training pipeline itself; `patches/openwakeword-*.patch` are modifications to it |
| [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator) (Michael Hansen / Rhasspy) | MIT | TTS generation of synthetic positives; `patches/piper-sample-generator-*.patch` modifies it |

The patch files in `patches/` are derivative works of those projects and remain
under their respective upstream licenses (Apache-2.0 and MIT), not this repo's
MIT grant. Apply them only to a checkout of the corresponding upstream project.

## Pretrained models

`setup_env.sh` downloads openWakeWord's pretrained feature extractors
(`melspectrogram.onnx`, `embedding_model.onnx`, and their tflite equivalents)
from the openWakeWord v0.5.1 release, and a Piper TTS voice
(`en_US-libritts_r-medium`) from the piper-sample-generator v2.0.0 release.
These carry their own upstream terms. Additional Piper voices, if you fetch them,
come from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)
under the terms stated there.

## Training data — the important caveat

`02_download_data.py` downloads:

- **AudioSet** (background noise) — via the Hugging Face `agkphysics/AudioSet`
  mirror; individual clips are sourced from YouTube and carry mixed terms
- **FMA** (`fma_small`) — Free Music Archive, mixed per-track Creative Commons
  licenses
- **MIT room impulse responses**
- **openWakeWord's precomputed ACAV100M negative features**

Taken together this is a mix of licenses and usage restrictions. Following the
guidance in the upstream training notebook: **models trained with this data
should be treated as non-commercial, personal use only.**

That restriction comes from the *data*, not from this repository's code. The MIT
license on the code does not, and cannot, grant rights over the datasets.

## Recordings

The real recordings this project evaluates against are of identifiable people.
They are **not** included in this repository, are not redistributable, and live
outside the checkout (see [`repo_paths.py`](repo_paths.py)). Throughout the
committed metadata and results they appear only as neutral speaker IDs
(`speaker_a`, `speaker_b`).
