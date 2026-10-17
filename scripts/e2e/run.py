# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""End-to-end check of an installed mj_viser wheel. Run by scripts/smoke_wheel.sh.

Usage: python -I scripts/e2e/run.py OUT_DIR ENV_LABEL

Writes OUT_DIR/smoke_report.json (results and installed versions), the recordings,
and scene_report.json. With MJ_VISER_ROBOT_XML (os.pathsep-separated model paths),
also checks those robots and writes robot_<model>.{json,viser}.
"""

from __future__ import annotations

import faulthandler
import importlib.metadata
import importlib.util
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

# Isolated mode (-I) leaves the script's directory off sys.path, so nothing from the
# repository checkout can shadow the installed wheel. Add only scripts/, for this package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402

import mj_viser  # noqa: E402
from e2e.clients import panel_free_groups, two_client_selection  # noqa: E402
from e2e.fixtures import MJCF, source_states  # noqa: E402
from e2e.lifecycle import check_lifecycle  # noqa: E402
from e2e.package_checks import check_package  # noqa: E402
from e2e.rendering import two_states  # noqa: E402
from e2e.robots import robot_scenario  # noqa: E402
from e2e.textures import texture_scenario  # noqa: E402

# A startup hang is one of the failures under test; never let it stall CI.
faulthandler.dump_traceback_later(300, exit=True)


def main(out_dir: str, env_label: str) -> None:
    modules = check_package()
    model = mujoco.MjModel.from_xml_string(MJCF)
    sources = source_states(model)
    report: dict[str, object] = {}

    two_states(model, sources, out_dir, report)
    report["two_client_selection"] = two_client_selection(model)
    report["texture_oracle"] = texture_scenario()
    report["panel_free_groups"] = panel_free_groups(model)
    check_lifecycle(model, sources, report)

    if robots := os.environ.get("MJ_VISER_ROBOT_XML"):
        report["robots"] = [robot_scenario(path, out_dir) for path in robots.split(os.pathsep)]
        if not any(r["textured_mesh_nodes"] for r in report["robots"]):  # type: ignore[index]
            raise SystemExit("no robot rendered a textured mesh")

    report.update(
        environment=env_label,
        installed={d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()},
        python=sys.version.split()[0],
        mj_viser=version("mj-viser"),
        mujoco=version("mujoco"),
        viser=version("viser"),
        mj_manipulator_installed=importlib.util.find_spec("mj_manipulator") is not None,
        modules_imported=modules,
        public_api=sorted(mj_viser.__all__),
    )
    with open(os.path.join(out_dir, "smoke_report.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
