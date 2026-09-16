# Contributing to FRPM

Thanks for your interest in contributing! FRPM (Facial identity Reference
Pack Maker) turns a YouTube video into a curated facial identity/expression
reference pack for AI image-generation workflows. This document covers how
to get set up and what we expect from contributions.

## Ground rules

* **Privacy first.** All face/identity processing must remain local -
  never add code paths that send images, embeddings, or video data to a
  remote/cloud API, and never add telemetry.
* **License discipline.** Before adding a new pretrained model or
  dependency (especially anything face-recognition related), check its
  license. We avoid models restricted to "non-commercial" or "research
  only" use (this is why InsightFace/InstantID/PuLID's default weights are
  intentionally not used - see the README's "Models & licensing" section).
  Prefer Apache-2.0/MIT/BSD-licensed models and libraries.
* Don't commit real people's photos/videos, generated output directories,
  or any personal data as part of a PR - `output/` is gitignored for this
  reason.

## Development setup

```bash
git clone https://github.com/Sadik-Dev/frpm.git
cd frpm
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

FFmpeg must also be installed and on `PATH` (see the README's Installation
section for platform-specific instructions).

## Project structure

```
src/frpm/          # all pipeline modules (one clear responsibility each)
tests/unit/        # fast, synthetic-data tests - no models/network required
tests/integration/ # slower tests against real models - marked `slow`
```

See the module docstrings in `src/frpm/` (especially `pipeline.py` and
`identity.py`) for how the stage-based pipeline and resume/caching work.

## Running tests

```bash
python -m pytest tests/unit            # fast - run this before every commit
python -m pytest tests/integration -m slow -v   # slower - real models, opt-in
```

Please add or update unit tests for any behavioral change. Keep new tests
in `tests/unit` synthetic/fast unless they genuinely require a real model
or network access, in which case they belong in `tests/integration` marked
`@pytest.mark.slow`.

## Making changes

1. Fork the repo and create a branch off `main`.
2. Make focused, well-scoped changes - prefer several small PRs over one
   large one.
3. Run the fast unit test suite locally and ensure it passes.
4. Update `README.md` if you change CLI flags, output structure, or setup
   steps.
5. Open a pull request describing what changed and why. Link any related
   issue.

## Reporting bugs / requesting features

Please use the GitHub issue templates. Include:
* Your OS and Python version
* The exact command you ran (redact the video URL if it's private/sensitive)
* The full error message/traceback
* For bugs: what you expected vs. what happened

## Code style

* Follow the existing module layout - one clear responsibility per file
  under `src/frpm/`.
* Use type hints and the existing Pydantic/dataclass models in `models.py`
  for any new structured data.
* Keep comments purposeful (explain *why*, not *what*) - the codebase
  favors self-documenting names over dense comments.

## Questions

Open a [discussion or issue](https://github.com/Sadik-Dev/frpm/issues) -
happy to help you get oriented.
