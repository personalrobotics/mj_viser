# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Scene manager: mirrors the MuJoCo body tree as Viser frames and keeps it posed."""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np
import viser

from mj_viser.geom_builders import GEOM_BUILDERS

_ROOT = "/mujoco"
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _segment(prefix: str, index: int, name: str) -> str:
    """Path segment for one element: unique by index, readable by name.

    MuJoCo names may be empty or contain "/" (namespaces), and sanitizing can make
    two names equal, so the index alone guarantees uniqueness.
    """
    return f"{prefix}{index}_{_UNSAFE.sub('_', name)}" if name else f"{prefix}{index}"


def _name(model: mujoco.MjModel, obj: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, obj, index) or ""


def body_paths(model: mujoco.MjModel) -> list[str]:
    """Viser path of every body, nested to mirror the MuJoCo body tree."""
    paths: list[str] = []
    for b in range(model.nbody):
        parent = _ROOT if b == 0 else paths[model.body_parentid[b]]
        paths.append(f"{parent}/{_segment('b', b, _name(model, mujoco.mjtObj.mjOBJ_BODY, b))}")
    return paths


def _geom_alpha(model: mujoco.MjModel, geom_id: int) -> float:
    mat_id = model.geom_matid[geom_id]
    return float(model.mat_rgba[mat_id][3] if mat_id >= 0 else model.geom_rgba[geom_id][3])


def build_scene_report(model: mujoco.MjModel) -> dict[str, Any]:
    """Machine-readable account of how every MuJoCo visual element maps to the scene.

    Each body, geom, and site has its Viser path. Geoms are ``rendered``,
    ``filtered`` (with the model state that hides them), or ``unsupported``; every
    element mj_viser does not render appears in ``unsupported`` with a reason.
    """
    bodies = body_paths(model)
    report: dict[str, Any] = {"bodies": [], "geoms": [], "sites": [], "unsupported": []}
    unsupported = report["unsupported"]

    for b, path in enumerate(bodies):
        report["bodies"].append(
            {
                "id": b,
                "name": _name(model, mujoco.mjtObj.mjOBJ_BODY, b),
                "path": path,
                "parent": None if b == 0 else int(model.body_parentid[b]),
                # A body's pose relative to its parent changes only through its own joints or,
                # for a mocap body, through mocap_pos/mocap_quat.
                "dynamic": bool(b != 0 and (model.body_jntnum[b] > 0 or model.body_mocapid[b] >= 0)),
            }
        )

    for g in range(model.ngeom):
        name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        gtype = int(model.geom_type[g])
        type_name = mujoco.mjtGeom(gtype).name.removeprefix("mjGEOM_").lower()
        entry: dict[str, Any] = {
            "id": g,
            "name": name,
            "path": f"{bodies[model.geom_bodyid[g]]}/{_segment('g', g, name)}",
            "body": int(model.geom_bodyid[g]),
            "type": type_name,
            "group": int(model.geom_group[g]),
        }
        if gtype not in GEOM_BUILDERS:
            entry["status"] = "unsupported"
            unsupported.append({"kind": "geom", "id": g, "name": name, "reason": f"{type_name} geoms are not rendered"})
        elif _geom_alpha(model, g) == 0:
            entry["status"] = "filtered"
            entry["reason"] = "rgba alpha is 0"
        else:
            entry["status"] = "rendered"
        report["geoms"].append(entry)

    for s in range(model.nsite):
        name = _name(model, mujoco.mjtObj.mjOBJ_SITE, s)
        report["sites"].append(
            {
                "id": s,
                "name": name,
                "path": f"{bodies[model.site_bodyid[s]]}/{_segment('s', s, name)}",
                "body": int(model.site_bodyid[s]),
            }
        )
        unsupported.append(
            {"kind": "site", "id": s, "name": name, "reason": "site shapes are not rendered; frame only"}
        )

    for kind, count, obj, reason in (
        ("light", model.nlight, mujoco.mjtObj.mjOBJ_LIGHT, "MuJoCo lights are not rendered; Viser default lighting"),
        ("camera", model.ncam, mujoco.mjtObj.mjOBJ_CAMERA, "cameras are not rendered"),
        ("tendon", model.ntendon, mujoco.mjtObj.mjOBJ_TENDON, "tendons are not rendered"),
        ("skin", model.nskin, mujoco.mjtObj.mjOBJ_SKIN, "skins are not rendered"),
        ("flex", model.nflex, mujoco.mjtObj.mjOBJ_FLEX, "flexes are not rendered"),
    ):
        for i in range(count):
            unsupported.append({"kind": kind, "id": i, "name": _name(model, obj, i), "reason": reason})

    for t in range(model.ntex):
        if model.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            name = _name(model, mujoco.mjtObj.mjOBJ_TEXTURE, t)
            unsupported.append({"kind": "skybox", "id": t, "name": name, "reason": "skybox textures are not rendered"})

    return report


class SceneManager:
    """Owns the Viser nodes for one model: a frame per body, geoms and sites beneath.

    Geom and site poses are static in their body's frame. Each update poses only the
    dynamic body frames (those with joints, and mocap bodies), relative to their parent.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        lock: threading.RLock,
    ) -> None:
        self._server = server
        self._lock = lock  # held by the viewer while it writes data; taken here for GUI callbacks
        self._model = model
        self._data = data
        self.report = build_scene_report(model)
        self._body_frames: dict[int, viser.FrameHandle] = {}
        self._dynamic_bodies = [b["id"] for b in self.report["bodies"] if b["dynamic"]]
        self._geom_handles: dict[int, viser.SceneNodeHandle] = {}
        self._site_frames: dict[int, viser.FrameHandle] = {}
        self._hidden_groups: set[int] = set()  # groups toggled off by the user
        # Per-client selection: client id -> (geom id, that client's label).
        self._selection: dict[int, tuple[int, viser.LabelHandle]] = {}
        self._on_select_callbacks: list[Callable[[int, str], None]] = []

    def build_scene(self) -> None:
        """Create the body frames, geoms, and site frames, and pose them from data."""
        scene = self._server.scene
        scene.set_up_direction("+z")  # match MuJoCo, so positions map 1:1

        for body in self.report["bodies"]:
            self._body_frames[body["id"]] = scene.add_frame(body["path"], show_axes=False)

        for geom in self.report["geoms"]:
            if geom["status"] != "rendered":
                continue
            g = geom["id"]
            handle = GEOM_BUILDERS[int(self._model.geom_type[g])](scene, geom["path"], g, self._model)
            handle.position = self._model.geom_pos[g]
            handle.wxyz = self._model.geom_quat[g]
            handle.on_click(lambda event, gid=g: self._handle_click(event, gid))
            self._geom_handles[g] = handle

        for site in self.report["sites"]:
            s = site["id"]
            self._site_frames[s] = scene.add_frame(
                site["path"], show_axes=False, position=self._model.site_pos[s], wxyz=self._model.site_quat[s]
            )

        self._server.on_client_disconnect(lambda client: self._forget_client(client.client_id))
        # Bodies without joints or mocap never move relative to their parent: pose them once, here.
        self._pose_bodies(range(1, self._model.nbody))

    def update_transforms(self) -> None:
        """Pose the dynamic body frames from the current data."""
        self._pose_bodies(self._dynamic_bodies)

    def _pose_bodies(self, bodies: Any) -> None:
        data, model = self._data, self._model
        parent_inv = np.zeros(4)
        local_quat = np.zeros(4)
        local_pos = np.zeros(3)
        with self._server.atomic():
            for b in bodies:
                p = model.body_parentid[b]
                mujoco.mju_negQuat(parent_inv, data.xquat[p])
                mujoco.mju_mulQuat(local_quat, parent_inv, data.xquat[b])
                mujoco.mju_rotVecQuat(local_pos, data.xpos[b] - data.xpos[p], parent_inv)
                frame = self._body_frames[b]
                frame.position = local_pos.copy()
                frame.wxyz = local_quat.copy()

    def update_visibility(self, visible_groups: set[int]) -> None:
        """Show geoms whose MuJoCo group is in ``visible_groups``; hide the rest."""
        with self._lock:
            all_groups = {int(self._model.geom_group[g]) for g in self._geom_handles}
            self._hidden_groups = all_groups - visible_groups
            with self._server.atomic():
                for g, handle in self._geom_handles.items():
                    handle.visible = int(self._model.geom_group[g]) not in self._hidden_groups

    def on_select(self, callback: Callable[[int, str], None]) -> None:
        """Register a callback for geom selection: callback(geom_id, body_name)."""
        self._on_select_callbacks.append(callback)

    def _handle_click(self, event: Any, geom_id: int) -> None:
        """Toggle the clicking client's selection; other clients are unaffected."""
        with self._lock:
            previous = self._selection.pop(event.client_id, None)
            if previous is not None:
                previous[1].remove()
                if previous[0] == geom_id:
                    return  # clicking the selected geom again deselects it

            body_id = int(self._model.geom_bodyid[geom_id])
            body_name = _name(self._model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
            # A child of the geom node, so it follows the geom without per-update work.
            label = event.client.scene.add_label(
                f"{self.report['geoms'][geom_id]['path']}/selection_label", text=body_name
            )
            self._selection[event.client_id] = (geom_id, label)

        for cb in self._on_select_callbacks:
            cb(geom_id, body_name)

    def _forget_client(self, client_id: int) -> None:
        with self._lock:
            self._selection.pop(client_id, None)
