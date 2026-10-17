# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Real robots (MJ_VISER_ROBOT_XML) at two known states."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import mujoco

import mj_viser
from e2e.fixtures import kinematics
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
)


def robot_scenario(xml_path: str, out_dir: str) -> dict[str, Any]:
    """A real articulated robot at two known states: its home keyframe, then every hinge bent 0.3 rad.

    Writes robot_<model>.viser and robot_<model>.json (scene report plus body and child
    local transforms per state, and which sites show) and checks every body, geom, and
    site, live and in the recording. Sites are then shown in every group, as an
    application would with ``set_visible_groups``, and must all match MuJoCo.
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    home = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, home, 0)
    bent = home.qpos.copy()
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            bent[model.jnt_qposadr[j]] += 0.3
    states = {"A": home.qpos.copy(), "B": bent}

    label = "/".join(xml_path.replace(os.sep, "/").split("/")[-3:])  # machine-independent
    out: dict[str, Any] = {"model": label, "states": {}}
    with mj_viser.MujocoViewer(model, port=0) as viewer:
        scene = viewer.scene_report()
        handles = viewer._scene_mgr._geom_handles  # type: ignore[union-attr]
        out["textured_mesh_nodes"] = sum(type(h).__name__ == "GlbHandle" for h in handles.values())
        child_paths = {g["path"] for g in scene["geoms"]} | {x["path"] for x in scene["sites"]}
        serializer = viewer.server.get_scene_serializer()
        children: dict[str, Any] = {}
        for name, qpos in states.items():
            d = mujoco.MjData(model)
            d.qpos[:] = qpos
            viewer.update(d)
            live = live_local_poses(viewer)
            checked = check_world_poses(live, scene, model, kinematics(model, qpos), f"robot live {name}")
            drawn = check_against_mujoco_scene(viewer, model, kinematics(model, qpos), f"robot live {name}")
            local = {p: {"position": v[0].round(9).tolist(), "wxyz": v[1].round(9).tolist()} for p, v in live.items()}
            state_children = {p: v for p, v in local.items() if p in child_paths}
            if children and state_children != children:
                raise SystemExit("robot geom or site local transforms changed between states")
            children = state_children
            out["states"][name] = {
                "qpos": qpos.round(9).tolist(),
                "world_poses_checked": checked,
                "mujoco_scene_geoms_matched": drawn,
                "body_local_transforms": {b["path"]: local[b["path"]] for b in scene["bodies"]},
            }
            serializer.insert_sleep(0.5)
        recording = serializer.serialize()
        live_visible = geom_visibility(viewer)

        site_handles = viewer._scene_mgr._site_handles  # type: ignore[union-attr]

        def shown_sites() -> list[str]:
            return sorted(
                e["name"] or e["path"]
                for e in scene["sites"]
                if e["id"] in site_handles and site_handles[e["id"]].visible
            )

        by_default = shown_sites()
        viewer.set_visible_groups(sites=range(6))
        check_against_mujoco_scene(viewer, model, kinematics(model, states["B"]), f"{label} with all site groups")
        with_all = shown_sites()
        if hidden := [e["name"] for e in scene["sites"] if e["status"] == "rendered" and e["id"] not in site_handles]:
            raise SystemExit(f"{label}: rendered sites without nodes: {hidden}")
        out["sites"] = {"shown_by_default": by_default, "shown_with_all_site_groups": with_all}
    messages = replay_recording(recording)
    check_coverage(scene, model)
    check_recorded_visibility(messages, live_visible, "robot recording")
    for name, window_end in (("A", 0.5), ("B", float("inf"))):
        check_world_poses(
            recorded_local_poses(messages, window_end),
            scene,
            model,
            kinematics(model, states[name]),
            f"robot recording {name}",
        )
    if moved := static_child_messages(messages, scene):
        raise SystemExit(f"robot recording moved static children: {moved[:5]}")

    slug = label.removesuffix(".xml").replace("/", "_")
    with open(os.path.join(out_dir, f"robot_{slug}.viser"), "wb") as f:
        f.write(recording)
    out["child_local_transforms"] = children
    out["scene"] = scene
    out["recording"] = {"file": f"robot_{slug}.viser", "sha256": hashlib.sha256(recording).hexdigest()}
    with open(os.path.join(out_dir, f"robot_{slug}.json"), "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    return {
        "model": label,
        "bodies": len(scene["bodies"]),
        "world_poses_checked": {k: v["world_poses_checked"] for k, v in out["states"].items()},
        "unsupported": sorted(f"{u['kind']}:{u['name']}" for u in scene["unsupported"]),
        "coverage": scene["coverage"],
        "textured_mesh_nodes": out["textured_mesh_nodes"],
        "sites": {k: len(v) for k, v in out["sites"].items()},
        "recording_sha256": out["recording"]["sha256"],
    }
