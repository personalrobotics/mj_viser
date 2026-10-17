# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The two-state scenario: render, record, and check the scene against MuJoCo."""

from __future__ import annotations

import hashlib
import json
import os

import mujoco
import numpy as np

import mj_viser
from e2e.fixtures import MOCAP, STATES, expect_raises, kinematics, state_hash
from e2e.oracles import (
    check_against_mujoco_scene,
    check_coverage,
    check_recorded_visibility,
    check_world_poses,
    geom_visibility,
    live_local_poses,
    recorded_local_poses,
    replay_recording,
    static_child_messages,
    tags_in,
)


def check_scene_report(scene: dict, model: mujoco.MjModel, viewer: mj_viser.MujocoViewer) -> None:  # type: ignore[type-arg]
    """Paths are unique and id-based; every element mj_viser skips is reported."""
    paths = [x["path"] for k in ("bodies", "geoms", "sites") for x in scene[k]]
    if len(set(paths)) != len(paths):
        raise SystemExit("scene paths collide")
    for kind, prefix in (("geoms", "g"), ("sites", "s")):
        unnamed = [e for e in scene[kind] if not e["name"]]
        if not unnamed or not all(e["path"].rsplit("/", 1)[-1] == f"{prefix}{e['id']}" for e in unnamed):
            raise SystemExit(f"unnamed {kind} lack id paths: {unnamed}")
    site_status = {e["name"]: e["status"] for e in scene["sites"]}
    if site_status.get("ghost") != "filtered" or site_status.get("mark_ellipsoid") != "rendered":
        raise SystemExit(f"site statuses wrong: {site_status}")
    twins = [b["path"] for b in scene["bodies"] if b["name"] in ("ns/a_b", "ns_a/b")]
    if len(set(twins)) != 2:
        raise SystemExit(f"names that sanitize alike share a path: {twins}")
    reported = {(u["kind"], u["name"]) for u in scene["unsupported"]}
    for expected in (
        ("geom", "terrain"),
        ("texture", "painted"),
        ("material-texture", "layered"),
        ("site-texture", "painted_site"),
        ("light", "lamp"),
        ("camera", "overhead"),
    ):
        if expected not in reported:
            raise SystemExit(f"{expected} missing from unsupported report: {sorted(reported)}")
    check_coverage(scene, model)
    if not any(d["name"] == "floor" and "infinite plane" in d["note"] for d in scene["diagnostics"]):
        raise SystemExit(f"infinite floor missing from diagnostics: {scene['diagnostics']}")
    live = live_local_poses(viewer)
    if missing := [
        p
        for p in paths
        if p not in live
        and not any(e["path"] == p and e["status"] != "rendered" for e in scene["geoms"] + scene["sites"])
    ]:
        raise SystemExit(f"reported paths with no scene node: {missing}")


def two_states(
    model: mujoco.MjModel, sources: dict[str, mujoco.MjData], out_dir: str, report: dict[str, object]
) -> None:
    """Render the two caller states, record them, and check scene, groups, markup, and replay."""
    viewer = mj_viser.MujocoViewer(model, port=0, label="mj_viser e2e")
    url = viewer.open()
    host, port = url.removeprefix("http://").rsplit(":", 1)
    if host != "127.0.0.1" or int(port) == 0:
        raise SystemExit(f"expected loopback URL with a real port, got {url}")
    report["default_url_host"] = host
    report["ephemeral_port_resolved"] = int(port) != 0

    # The textured mesh must take the trimesh + Pillow texture path (a GLB node), not the solid-color fallback.
    tet = viewer._scene_mgr._geom_handles[model.geom("tet").id]  # type: ignore[union-attr]
    if type(tet).__name__ != "GlbHandle":
        raise SystemExit(f"textured mesh rendered as {type(tet).__name__}, not a textured GLB")
    report["textured_mesh_node"] = type(tet).__name__

    scene = viewer.scene_report()
    check_scene_report(scene, model, viewer)
    report["scene"] = {
        "paths": len(scene["bodies"]) + len(scene["geoms"]) + len(scene["sites"]),
        "unsupported": sorted(f"{u['kind']}:{u['name']}" for u in scene["unsupported"]),
    }

    serializer = viewer.server.get_scene_serializer()
    states_report = {}
    child_paths = {g["path"] for g in scene["geoms"]} | {x["path"] for x in scene["sites"]}
    children_before = {p: v for p, v in live_local_poses(viewer).items() if p in child_paths}
    for name, d in sources.items():
        before = state_hash(d)
        viewer.update(d)
        after = state_hash(d)
        if before != after:
            raise SystemExit(f"update() modified source state {name}")
        live = live_local_poses(viewer)
        reference = kinematics(model, STATES[name], MOCAP[name])
        checked = check_world_poses(live, scene, model, reference, f"live state {name}")
        drawn = check_against_mujoco_scene(viewer, model, reference, f"live state {name}")
        if viewer.data is d:
            raise SystemExit("viewer retained the caller's MjData")
        states_report[name] = {
            "qpos": STATES[name],
            "source_sha256_before": before,
            "source_sha256_after": after,
            "world_poses_checked": checked,
            "mujoco_scene_geoms_matched": drawn,
        }
        serializer.insert_sleep(0.5)
    recording = serializer.serialize()
    live_visible = geom_visibility(viewer)

    # --- Group visibility is explicit viewer state ------------------------------------------
    # Showing every geom and site group must match MuJoCo with every group enabled, and
    # restoring MuJoCo's defaults (groups 0-2) must match MuJoCo's default view again.
    at_b = kinematics(model, STATES["B"], MOCAP["B"])
    viewer.set_visible_groups(geoms=range(6), sites=range(6))
    all_groups = check_against_mujoco_scene(viewer, model, at_b, "all groups shown")
    viewer.set_visible_groups(geoms={0, 1, 2}, sites={0, 1, 2})
    default_groups = check_against_mujoco_scene(viewer, model, at_b, "default groups restored")
    if all_groups <= default_groups:
        raise SystemExit("showing all groups revealed nothing; the group-4 site should appear")
    report["group_visibility"] = {"default_groups_drawn": default_groups, "all_groups_drawn": all_groups}

    # Geoms and sites are static in their body frames: updates must not have moved them.
    for path, (pos, quat) in children_before.items():
        if not (np.allclose(live[path][0], pos) and np.allclose(live[path][1], quat)):
            raise SystemExit(f"static child {path} changed between states")
    if below_visible := viewer._scene_mgr._geom_handles[model.geom("below").id].visible:  # type: ignore[union-attr]
        report["geom_below_floor_visible"] = below_visible
    else:
        raise SystemExit("geom below z=-0.5 was hidden")

    # --- Untrusted display strings are shown as text, never rendered as markup ------------
    payload = '<img src=x onerror="alert(1)"><script>alert(2)</script>'
    viewer.set_hud("probe", payload)
    panel = mj_viser.SensorPanel(
        title=payload, channels=[mj_viser.SensorChannel(0, payload, color='red;"><img src=x>')], use_folder=False
    )
    viewer.add_panel(panel)
    markup = {"hud": viewer._huds["probe"].content, "sensor_legend": panel._legend.content}  # type: ignore[union-attr]
    for where, content in markup.items():
        injected = tags_in(content) - {"div", "span"}
        if injected:
            raise SystemExit(f"{where} rendered injected markup {sorted(injected)}: {content}")
    report["markup_escaped"] = sorted(markup)
    report["hud_unknown_position"] = expect_raises(ValueError, lambda: viewer.set_hud("probe", "x", "middle"))

    viewer.close()
    if viewer.is_open:
        raise SystemExit("viewer still open after close()")
    # The report is a function of the model alone: identical open, closed, or elsewhere.
    if viewer.scene_report() != scene or mj_viser.MujocoViewer(model).scene_report() != scene:
        raise SystemExit("scene report differs between viewers of the same model")
    with open(os.path.join(out_dir, "scene_report.json"), "w") as f:
        json.dump(scene, f, indent=2, sort_keys=True)
        f.write("\n")
    rec_path = os.path.join(out_dir, "two_states.viser")
    with open(rec_path, "wb") as f:
        f.write(recording)
    report["states"] = states_report
    report["recording"] = {"file": "two_states.viser", "sha256": hashlib.sha256(recording).hexdigest()}

    # --- The recording itself replays to both states ---------------------------------------
    # Oracle on the artifact users open, independent of the live server: replaying the
    # file's messages up to each state's timestamp yields MuJoCo's kinematics for that state.
    messages = replay_recording(recording)
    for name, window_end in (("A", 0.5), ("B", float("inf"))):
        recorded = recorded_local_poses(messages, window_end)
        check_world_poses(
            recorded, scene, model, kinematics(model, STATES[name], MOCAP[name]), f"recording state {name}"
        )
    if moved := static_child_messages(messages, scene):
        raise SystemExit(f"recording moved static children: {moved[:5]}")
    report["recording"]["replays_to_states"] = ["A", "B"]
    check_recorded_visibility(messages, live_visible, "recording")
    dynamic = {b["name"]: b["dynamic"] for b in scene["bodies"]}
    if not dynamic["target"] or dynamic["cap"]:
        raise SystemExit(f"mocap body must be dynamic and fixed child static: {dynamic}")
    report["dynamic_bodies"] = sorted(name for name, moves in dynamic.items() if moves)
