# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The ssrobot integration's adapters on stand-in records (needs ssrobot installed).

Usage: python -I scripts/e2e/ssrobot_adapters.py OUT_DIR

Runs the trace replay, trace channel, and trace image checks of the integration lane
and writes ssrobot_adapters.json, trace_images.viser, and trace_images_conversion.json.
"""

from __future__ import annotations

import json
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402

from e2e.fixtures import MJCF  # noqa: E402
from e2e.trace_channels import check_trace_channels  # noqa: E402
from e2e.trace_images import check_trace_images  # noqa: E402
from e2e.trace_replay import check_replay  # noqa: E402


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_string(MJCF)
    report = {
        "trace_replay": check_replay(model),
        "trace_channels": check_trace_channels(),
        "trace_images": check_trace_images(model, str(out)),
        "versions": {p: version(p) for p in ("mj-viser", "ssrobot", "mujoco", "viser")},
    }
    (out / "ssrobot_adapters.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
