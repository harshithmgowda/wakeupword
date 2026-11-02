#!/usr/bin/env python
"""Step 3: write my_model.yaml, the training config consumed by openwakeword's train.py.

Edit the constants below (or pass flags) and re-run to change training settings.

    ./.venv/bin/python 03_make_config.py
    ./.venv/bin/python 03_make_config.py --steps 40000 --penalty 1500
"""
import argparse
from pathlib import Path

import yaml

HERE = Path(__file__).parent.resolve()

# --- the three knobs from the notebook's form ----------------------------
TARGET_WORD = "ouh_ke-i_nahbu"
NUMBER_OF_EXAMPLES = 50000
NUMBER_OF_TRAINING_STEPS = 20000
FALSE_ACTIVATION_PENALTY = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default=TARGET_WORD)
    ap.add_argument("--examples", type=int, default=NUMBER_OF_EXAMPLES)
    ap.add_argument("--steps", type=int, default=NUMBER_OF_TRAINING_STEPS)
    ap.add_argument("--penalty", type=int, default=FALSE_ACTIVATION_PENALTY)
    args = ap.parse_args()

    base = HERE / "openwakeword" / "examples" / "custom_model.yml"
    config = yaml.load(base.read_text(), yaml.Loader)

    config["target_phrase"] = [args.word]
    config["model_name"] = args.word.replace(" ", "_")
    config["n_samples"] = args.examples
    config["n_samples_val"] = max(500, args.examples // 10)
    config["steps"] = args.steps
    config["target_accuracy"] = 0.5
    config["target_recall"] = 0.25
    config["output_dir"] = "./my_custom_model"
    config["max_negative_weight"] = args.penalty

    config["background_paths"] = ["./audioset_16k", "./fma"]
    config["rir_paths"] = ["./mit_rirs"]
    config["false_positive_validation_data_path"] = "./validation_set_features.npy"
    config["feature_data_files"] = {
        "ACAV100M_sample": "./openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    }
    # background_paths_duplication_rate must line up 1:1 with background_paths
    config["background_paths_duplication_rate"] = [1] * len(config["background_paths"])

    # Pre-decode the background pool into one memmap of fixed-length segments, built
    # on first use. AddBackgroundNoise otherwise re-reads a background file per clip,
    # which was ~507 ms of every ~1077 ms augmentation batch. See patches/README.md.
    config["background_cache_path"] = "./background_cache.npy"

    # The notebook runs on a Colab CPU box; on a local GPU we can afford
    # larger TTS and augmentation batches, which dominate wall-clock at 50k
    # examples. Lower these if you hit VRAM limits.
    config["tts_batch_size"] = 100
    config["augmentation_batch_size"] = 32

    out = HERE / "my_model.yaml"
    with out.open("w") as f:
        yaml.dump(config, f)

    print(f"Wrote {out}")
    for k in ("model_name", "target_phrase", "n_samples", "n_samples_val", "steps",
              "max_negative_weight", "tts_batch_size", "augmentation_batch_size"):
        print(f"  {k}: {config[k]}")


if __name__ == "__main__":
    main()
