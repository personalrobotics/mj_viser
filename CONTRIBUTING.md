# Contributing

## Development setup

mj_viser is a standalone package; clone it and use [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/personalrobotics/mj_viser.git
cd mj_viser
uv sync --locked --all-extras
```

uv manages the development environment and the lock file (`uv.lock`); use it for any
change to dependencies. Users install mj_viser with `pip` (or any installer); that path is
what the end-to-end check exercises.

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
UV_PYTHON=3.12 scripts/smoke_wheel.sh dist/smoke locked   # end-to-end check from a clean wheel
```

The end-to-end check builds the wheel, installs it into a fresh virtualenv, and runs
`scripts/e2e/run.py` against the installed package, including the README's quick start as
written. Its report and recordings land in
`dist/smoke/locked/`.

### The ssrobot integration

`mj_viser.integrations.ssrobot` is checked by separate CI jobs against ssrobot and its
reference robots, Geodude and ADA, each pinned to a commit in `.github/workflows/ci.yml`.
To run them locally, install those pins into an environment with the wheel, then run
`scripts/e2e/ssrobot_live.py`, `ssrobot_replay.py`, `ssrobot_preview.py`, and
`ssrobot_contacts.py` as the workflow does. Core mj_viser never imports ssrobot; CI fails if
it does.

### The README

The README is also the PyPI project description. Write its images and links as relative
paths and `#` anchors, which work on GitHub and in editors; the build (the
`fancy-pypi-readme` hook in `pyproject.toml`) points them at the repository's `main`
branch, and `scripts/check_readme.py` renders the built wheel's description as PyPI does and
checks each one (CI runs it on every build). The clips at its top are captured by
`scripts/media/capture.py`; [scripts/media/README.md](scripts/media/README.md) says how.

## Pull requests

- Branch from `main`. Open a PR with a clear summary and how you verified it.
- CI (ruff, mypy, the README check, the clean-wheel end-to-end check on every supported
  Python, and the ssrobot integration jobs) must pass.
- Review is by [@siddhss5](https://github.com/siddhss5) (enforced via CODEOWNERS).

## Releasing

[docs/api.md](docs/api.md) gives the version and tag rules. `.github/workflows/release.yml`
publishes a tag: a release candidate (`vX.Y.ZrcN`) to TestPyPI, a release (`vX.Y.Z`, from
1.0.0 on) to PyPI. It first runs all of CI on the tagged commit, checks the sdist the same
way as the wheel, and writes `release-report.json`; then it uploads the exact files CI
verified, after a maintainer approves the upload; then it installs the release back from
the index into a fresh environment and runs the README's quick start. A run that fails
before the upload publishes nothing. A manual run (Actions → Release → Run workflow) only
builds and verifies, unless `publish_testpypi` is ticked.

### One-time setup

- On pypi.org and test.pypi.org, a trusted publisher for project `ssmjviser`: owner
  `personalrobotics`, repository `mj_viser`, workflow `release.yml`, environment `pypi`
  (on pypi.org) or `testpypi` (on test.pypi.org). No API token is stored anywhere.
- In the repository's settings, environments `pypi` and `testpypi`, each with a required
  reviewer and deployments limited to tags matching `v*`.
- Zenodo's GitHub integration enabled for the repository, before the first release, so
  that publishing a GitHub Release archives it under the project's concept DOI.

### A release candidate

1. In a pull request, set `version` in `pyproject.toml` to `X.Y.ZrcN`, and merge it.
2. Tag the merged commit and check it locally:

   ```bash
   git switch main && git pull
   git tag -a vX.Y.ZrcN -m "mj_viser X.Y.ZrcN"
   scripts/release-precheck.sh vX.Y.ZrcN
   git push origin vX.Y.ZrcN
   ```

3. Approve the `testpypi` deployment when the run asks. The run then installs the
   candidate from TestPyPI and runs the quick start; its `release-report` artifact has
   the evidence. Check the project page on test.pypi.org: the README, its clips, and links.

A failed candidate is fixed in a new pull request and tagged `rcN+1`.

### A release

1. In a pull request, set `version` to `X.Y.Z` (a version distinct from every candidate),
   turn `## Unreleased` in `CHANGELOG.md` into `## X.Y.Z — YYYY-MM-DD`, and merge it. For
   1.0.0, also add the classifier `Development Status :: 5 - Production/Stable`.
2. Tag, check, and push as above, with `vX.Y.Z`. The tag must be on `main`.
3. Approve the `pypi` deployment. The run publishes to PyPI, then creates the GitHub
   Release with the same wheel, sdist, and `SHA256SUMS` and the changelog section as its
   notes; Zenodo archives it. The run then installs `ssmjviser==X.Y.Z` from PyPI and runs the
   quick start.

PyPI never accepts a version twice: a published release cannot be replaced or re-uploaded.
If the upload succeeded and a later job failed, re-run the failed jobs; the upload is not
repeated. If a published release is broken, yank it on PyPI and release `X.Y.Z+1`.

After the first release, record the concept DOI Zenodo minted (the one that resolves to
the latest version) in `CITATION.cff`, and add the PyPI, Python, license, and DOI badges and
a BibTeX entry to the README, with `pip install ssmjviser` as its primary installation.

## The robot-code workspace (optional)

mj_viser is also part of the [robot-code workspace](https://github.com/personalrobotics/robot-code),
which clones sibling repositories into one uv workspace for cross-repository work. It is
not needed to develop mj_viser. When `main` advances and the repository has a
`ROBOT_CODE_DISPATCH_TOKEN` secret, CI asks robot-code to run its integration suite; without
the secret, that step is skipped with a notice.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License (see [LICENSE](LICENSE)).
