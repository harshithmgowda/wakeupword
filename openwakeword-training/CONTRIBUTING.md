# Contributing

This is a personal research repo rather than a maintained library, so please
open an issue before writing a large patch — the change may already have been
tried and refuted somewhere on the
[`experiments` branch](https://github.com/Dethon/openwakeword-training/tree/experiments),
which holds this project's full evaluation record.

Useful contributions, roughly in order of how much they would help:

- **A negative result.** If you ran one of the experiments here on your own wake
  word and data and got a different answer, that is worth more than a code
  change. Several conclusions here rest on n = 3 to n = 7.
- **Fixes to the local port.** Upstream URLs and dataset layouts rot; if
  `02_download_data.py` or `setup_env.sh` breaks, a fix is welcome.
- **Corrections.** If a number or a statistical claim in `docs/`, or in the
  evaluation record on the `experiments` branch, is wrong, say so. See the
  front-end probe write-up there for the standard this repo tries to hold itself
  to.

## Ground rules for changes that touch the pipeline

The awkward constraint here is that this pipeline exists to back up *recorded
measurements*, and a refactor can silently invalidate them.

1. **Prove equivalence.** Any change to feature extraction, augmentation,
   batching or the eval path must come with a test in `tests/` showing it does
   not alter what is computed — bit-identical where possible, and with the
   measured delta stated where not. `tests/README.md` explains the existing
   guards; follow their pattern.
2. **Never edit the vendored repos in place.** `openwakeword/` and
   `piper-sample-generator/` are gitignored clones. Changes there must be
   exported to `patches/` as `git -C <clone> diff <commit>~1 <commit>`, or they
   vanish on a fresh clone. `patches/README.md` documents the exact procedure
   and why `git diff HEAD~1` is the wrong command.
3. **Do not restate a result more strongly than its sample supports.** If a
   comparison is one discordant pair with McNemar p = 1.0, the write-up says so.
   Keep it that way.
4. **Do not commit audio, `.npy` feature files, or trained models.** They are
   gitignored for good reason; everything in `.gitignore` is regenerable from
   committed scripts.
5. **Keep paths portable.** Nothing should hardcode a home directory. Use
   `repo_paths.REPO_ROOT` for in-repo paths and `repo_paths.DATA_ROOT`
   (overridable via `OWW_DATA_ROOT`) for the external recording corpus.

## Style

Match the surrounding code. The prevailing convention is fairly dense
module-level docstrings that explain *why* a thing is the way it is — usually
because something else broke — and inline comments on the non-obvious lines.
That history is the point; please preserve it rather than tidying it away.

Scripts are run from the repo root.
