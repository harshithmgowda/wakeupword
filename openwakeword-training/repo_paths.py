"""Shared path resolution, so nothing in this repo hardcodes a home directory.

Two roots matter:

* ``REPO_ROOT`` -- this checkout. Derived from this file's own location, so it
  is correct no matter where the repo is cloned or which directory a script is
  invoked from.
* ``DATA_ROOT`` -- the corpus of **real recordings**, which deliberately lives
  *outside* the repo (it is audio of real people; it is not redistributable and
  would dwarf the repo besides). Override with the ``OWW_DATA_ROOT``
  environment variable; defaults to ``~/openwakeword-train-data``.

Scripts that live in a subdirectory cannot ``import repo_paths`` until the repo
root is on ``sys.path``, and python puts the *script's* directory there, not the
cwd. So they open with a two-line bootstrap before importing anything local::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # depth-dependent

    from repo_paths import DATA_ROOT  # noqa: E402

``parents[N]`` counts from the script: ``1`` for ``tools/``, ``2`` for
``tools/eval/``, and so on.
"""

import os
from pathlib import Path

#: This checkout. ``repo_paths.py`` sits at the repo root, hence ``parent``.
REPO_ROOT = Path(__file__).resolve().parent

#: The real-recording corpus, outside the repo. See the module docstring.
DATA_ROOT = Path(
    os.environ.get("OWW_DATA_ROOT", Path.home() / "openwakeword-train-data")
).expanduser()


def data_path(*parts: str) -> Path:
    """Join ``parts`` onto :data:`DATA_ROOT`.

    Sugar for the common ``DATA_ROOT / "unused" / "excluded" / "music"`` case,
    and the one place to add a friendlier error if the corpus is missing.
    """
    return DATA_ROOT.joinpath(*parts)


def require_data_root() -> Path:
    """Return :data:`DATA_ROOT`, failing loudly if it is not present.

    Most scripts here read real recordings that are not in the repo. Without
    this the failure surfaces as a confusing empty glob or a ``FileNotFoundError``
    deep inside a worker.
    """
    if not DATA_ROOT.is_dir():
        raise SystemExit(
            f"Recording corpus not found at {DATA_ROOT}.\n"
            "This repo's real recordings live outside the checkout. Set "
            "OWW_DATA_ROOT to point at yours, e.g.\n"
            "    OWW_DATA_ROOT=/path/to/recordings python <script>\n"
            "See docs/data-layout.md for the directory structure expected."
        )
    return DATA_ROOT
