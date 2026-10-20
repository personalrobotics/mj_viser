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
written (it serves on port 8080, which must be free). Its report and recordings land in
`dist/smoke/locked/`.

### The ssrobot integration

`mj_viser.integrations.ssrobot` is checked by separate CI jobs against ssrobot and its
reference robots, Geodude and ADA, each pinned to a commit in `.github/workflows/ci.yml`.
To run them locally, install those pins into an environment with the wheel, then run
`scripts/e2e/ssrobot_live.py`, `ssrobot_replay.py`, `ssrobot_preview.py`, and
`ssrobot_contacts.py` as the workflow does. Core mj_viser never imports ssrobot; CI fails if
it does.

### The README

The README is also the PyPI project description. Its images and links are absolute URLs;
`scripts/check_readme.py` renders the built wheel's description as PyPI does and checks
each one (CI runs it on every build). The clips at its top are captured by
`scripts/media/capture.py`; [scripts/media/README.md](scripts/media/README.md) says how.

## Pull requests

- Branch from `main`. Open a PR with a clear summary and how you verified it.
- CI (ruff, mypy, the README check, the clean-wheel end-to-end check on every supported
  Python, and the ssrobot integration jobs) must pass.
- Review is by [@siddhss5](https://github.com/siddhss5) (enforced via CODEOWNERS).

## The robot-code workspace (optional)

mj_viser is also part of the [robot-code workspace](https://github.com/personalrobotics/robot-code),
which clones sibling repositories into one uv workspace for cross-repository work. It is
not needed to develop mj_viser. When `main` advances and the repository has a
`ROBOT_CODE_DISPATCH_TOKEN` secret, CI asks robot-code to run its integration suite; without
the secret, that step is skipped with a notice.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License (see [LICENSE](LICENSE)).
