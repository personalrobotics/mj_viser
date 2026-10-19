# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Contacts of Geodude and ADA, drawn live, agreeing with ssrobot's own contact channels (#11).

Needs ssrobot (with its MuJoCo and planning extras, at a version with contact channels),
the geodude_assets and ada_assets packages, and ssrobot's references directory.

Usage: python -I scripts/e2e/ssrobot_contacts.py REFERENCES_DIR OUT_DIR

Each robot runs in its reference scene with privileged contact channels between the
cylinder and the bodies that touch it:

- Geodude grasps the cylinder: its right hand plans to the side of the cylinder, 6 cm out
  along its approach, and closes, lifting the cylinder off the worktable in its fingers;
- ADA's cylinder stands on its feeding table.

The run happens twice under a JSONL trace, without and with a LiveView carrying a
ContactOverlay and a browser that never reads. The checks: the traces are byte-identical;
for every declared pair, the overlay's contacts between the two bodies sum to the normal
force ssrobot observed, and touch exactly when ssrobot says they touch; every contact is
named by its bodies, with no robot-specific viewer code. Writes, per robot, the traces, a
recording of the final state with its contacts, and its contact report, and
ssrobot_contacts.json.
"""

from __future__ import annotations

import faulthandler
import json
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import msgspec  # noqa: E402
import numpy as np  # noqa: E402
from ssrobot import (  # noqa: E402
    GripperCommand,
    JsonlTrace,
    ObservationRequest,
    Pose,
    RobotContext,
    Tsr,
    load_installed_package,
)
from ssrobot.mujoco import MujocoRuntime  # noqa: E402
from ssrobot.planning import CBiRRT, SsikKinematics  # noqa: E402
from websockets.sync.client import connect  # noqa: E402

from e2e.channels import CAMERA  # noqa: E402
from mj_viser import ContactOverlay, LiveView, StateRecorder  # noqa: E402
from mj_viser.integrations.ssrobot import render_source  # noqa: E402

faulthandler.dump_traceback_later(900, exit=True)

HAND = "right_ur5e/gripper"
# Scenario data only: which bodies may touch the cylinder in each scene.
ROBOTS = {
    "geodude": {
        "module": "geodude_assets",
        "touching": [
            f"{HAND}/{side}_{part}" for side in ("left", "right") for part in ("pad", "follower", "spring_link")
        ]
        + ["vention_base"],
    },
    "ada": {"module": "ada_assets", "touching": ["ada_table"]},
}
SETTLE_TICKS = 200
TOLERANCE_N = 1e-6


def _grasp(ctx: Any, package: Any, scenario: dict[str, Any]) -> None:
    """Geodude's right hand to the cylinder's side, 6 cm out, then closed on it."""
    description = package.description
    arm = ctx.manipulator(scenario["manipulator"])
    planner = CBiRRT(description, kinematics=SsikKinematics.from_package(package, arm.name))
    snapshot = ctx.snapshot()
    pose = next(o.pose for o in snapshot.objects if o.name == scenario["object"])
    region = Tsr(
        origin=pose,
        offset=Pose(position=(-0.06, 0.0, 0.0), quat_wxyz=tuple(scenario["offset_quat_wxyz"])),
        position_bounds=((0.0, 0.0), (0.0, 0.0), (-0.02, 0.02)),
        rotation_bounds=((0.0, 0.0), (0.0, 0.0), (-np.pi, np.pi)),
    )
    result = arm.plan_to_tsr([region], snapshot=snapshot, planner=planner, frame=scenario["frame"])
    if not result.succeeded:
        raise SystemExit(f"the grasp approach {result.status.value}: {result.diagnostics}")
    ctx.run_until(ctx.execute(result, planner=planner), max_ticks=20_000)
    gripper = description.end_effector(description.manipulator(scenario["manipulator"]).end_effector).gripper
    ctx.run_until(ctx.submit(GripperCommand(gripper=gripper, opening=0.0)), max_ticks=3_000)


def run(name: str, references: Path, out: Path, viewed: bool) -> dict[str, Any]:
    spec = ROBOTS[name]
    package = load_installed_package(spec["module"])
    description = package.description
    scenario = next(r for r in tomllib.loads((references / "robots.toml").read_text())["robots"] if r["name"] == name)[
        "scenario"
    ]
    pairs = [("cylinder", other) for other in spec["touching"]]
    channels = tuple(f"contact.{a}.{b}" for a, b in pairs)
    mode = "viewed" if viewed else "plain"
    result: dict[str, Any] = {}
    with (
        JsonlTrace(out / f"{mode}.trace.jsonl") as trace,
        RobotContext(
            description,
            MujocoRuntime(package, scene=references / "scenes" / f"{name}-cylinder.xml", contacts=pairs),
            sinks=[trace],
        ) as ctx,
    ):
        view = overlay = silent = None
        if viewed:
            overlay = ContactOverlay()
            view = LiveView(render_source(ctx), max_rate_hz=None, port=0, label=name)
            view.viewer.add_panel(overlay)
            url = view.open().replace("http://", "ws://")
            silent = connect(url, subprotocols=[f"viser-v{version('viser')}"], max_size=None)
            silent.send(msgspec.msgpack.encode(CAMERA))
        if name == "geodude":
            _grasp(ctx, package, scenario)
        for _ in range(SETTLE_TICKS):
            ctx.step()
            if view is not None:
                view.refresh()
        observed = {r.channel: r.value for r in ctx.observe(ObservationRequest(channels=channels)).readings}
        result["observed"] = {k: [float(v) for v in value] for k, value in observed.items()}
        if view is not None and overlay is not None and silent is not None:
            view.refresh(force=True)
            contacts = overlay.contacts
            assert contacts is not None
            result["drawn"] = _compare(name, pairs, observed, contacts)
            with StateRecorder(view.viewer.model, contacts=True) as recorder:
                recorder.advance(0)
                recorder.show(view.viewer.data)
                (out / "contacts.viser").write_bytes(recorder.finish())
                (out / "contacts_report.json").write_text(json.dumps(recorder.contact_log, indent=2) + "\n")
            silent.close()
            view.close()
    return result


def _compare(name: str, pairs: list[tuple[str, str]], observed: dict[str, Any], contacts: Any) -> dict[str, Any]:
    """Each declared pair: the overlay's normal forces between the two bodies sum to
    what ssrobot observed, and they touch exactly when ssrobot says so."""
    if contacts.truncated:
        raise SystemExit(f"{name}: {contacts.total} contacts, more than drawn")
    out = {}
    for first, second in pairs:
        between = [c for c in contacts.contacts if {c.first, c.second} == {first, second}]
        normal = float(sum(np.dot(c.force, c.normal) for c in between if c.force is not None))
        touching, force = observed[f"contact.{first}.{second}"]
        if bool(between) != bool(touching) or abs(normal - force) > TOLERANCE_N * max(1.0, force):
            raise SystemExit(
                f"{name}: {first}/{second}: drawn {len(between)} contacts, {normal} N; ssrobot: {touching}, {force} N"
            )
        out[f"{first}/{second}"] = {"contacts": len(between), "normal_force_n": round(normal, 6)}
    return out


def main(references: str, out_dir: str) -> None:
    root = Path(out_dir)
    robots = {}
    for name in ROBOTS:
        out = root / name
        out.mkdir(parents=True, exist_ok=True)
        run(name, Path(references), out, viewed=False)
        viewed = run(name, Path(references), out, viewed=True)
        identical = (out / "plain.trace.jsonl").read_bytes() == (out / "viewed.trace.jsonl").read_bytes()
        if not identical:
            raise SystemExit(f"{name}: drawing contacts changed the trace")
        robots[name] = {**viewed, "traces_identical_with_and_without_viewer": identical}
    held = {k: v for k, v in robots["geodude"]["drawn"].items() if v["contacts"] and "vention_base" not in k}
    if not held or robots["geodude"]["drawn"]["cylinder/vention_base"]["contacts"]:
        raise SystemExit(f"Geodude does not hold the cylinder off the worktable: {robots['geodude']['drawn']}")
    if not robots["ada"]["drawn"]["cylinder/ada_table"]["contacts"]:
        raise SystemExit("ADA's cylinder does not rest on its table")
    report = {"robots": robots, "versions": {p: version(p) for p in ("mj-viser", "ssrobot", "mujoco", "viser")}}
    (root / "ssrobot_contacts.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
