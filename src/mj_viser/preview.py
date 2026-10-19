# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Preview planning results as a ghost robot, without touching execution.

A :class:`PlanPreview` shows ssrobot ``PlanResult`` s in the scene they were planned in:
each browser picks a result and scrubs its exact path waypoints or its timed trajectory,
evaluated exactly, and sees a translucent ghost of the moving robot there. The ghost is
in that browser's own scene; nothing is submitted, cancelled, or changed.

The ghost's state is the planning snapshot, applied through the recorded world's
snapshot mapping, with the result's joints set to the shown sample; held objects move
with their end effectors. mj_viser does not import ssrobot: results, snapshots, worlds,
and validations of the same shape work.
"""

from __future__ import annotations

import html
import math
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from mj_viser.geom_builders import PRIMITIVE_TYPES, _simple_mesh, build_primitive
from mj_viser.mesh_utils import extract_mujoco_mesh
from mj_viser.panels import PanelBase

if TYPE_CHECKING:
    import viser

    from mj_viser.live import LiveView
    from mj_viser.viewer import MujocoViewer

GHOST_RGBA = np.array([0.3, 0.6, 1.0, 0.35])
WITNESS_RGBA = np.array([1.0, 0.2, 0.2, 0.5])
_PATH, _TRAJECTORY, _WITNESS = "path", "trajectory", "first invalid state"


@dataclass
class _Preview:
    result: Any
    snapshot: Any
    validation: Any
    stale: str | None = None


class PlanPreview(PanelBase):
    """Ghost previews of planning results, chosen and scrubbed per browser.

    Add it to a viewer (``viewer.add_panel(preview)``, or ``live.viewer.add_panel``),
    then :meth:`add` results. With a :class:`~mj_viser.LiveView`, a result planned at
    an earlier scene revision than the latest frame's is refused, and a shown one goes
    stale when the scene changes.
    """

    def __init__(self, world: Any, description: Any, *, live: LiveView | None = None) -> None:
        """
        Args:
            world: The world the results were planned in, with ``model`` (an
                ``MjModel``), ``world`` and ``description`` (identities), and
                ``mapping`` (``apply(snapshot, data)``), like ssrobot's
                ``MaterializedWorld``.
            description: The robot description, whose end effectors name the frames
                held objects move with.
            live: The live view it is drawn over, whose world it must be and whose
                scene revision results must match, if any.

        Raises:
            ValueError: If the description or the live view is of another robot or world.
        """
        if description.fingerprint() != world.description:
            raise ValueError("the description is not the one the world was built for")
        if live is not None and live.world != world.world:
            raise ValueError(f"the live view shows world {live.world!r}, not {world.world!r}")
        self._world = world
        self._live = live
        self._end_effectors: dict[str, str] = {str(e.name): str(e.frame) for e in description.end_effectors}
        self._model: mujoco.MjModel = world.model
        self._previews: dict[str, _Preview] = {}
        self._clients: dict[int, _ClientPreview] = {}
        self._lock = threading.Lock()
        self._qpos = {name: address for name, address in world.mapping.joints}
        self._viewer: MujocoViewer | None = None

    def name(self) -> str:
        return "Plan preview"

    def add(self, name: str, result: Any, snapshot: Any, *, validation: Any = None) -> None:
        """Offer a planning result for preview.

        Args:
            name: How browsers choose it.
            result: A successful ``PlanResult`` with a path or trajectory.
            snapshot: The ``SceneSnapshot`` it was planned in.
            validation: Optionally, an invalid ``MotionValidation`` of this plan's
                motion, whose first invalid state is shown as supplied.

        Raises:
            ValueError: If the result has nothing to show, or was planned in another
                snapshot, world, description, or (live) scene revision; or the
                validation is valid or bound to another snapshot.
        """
        request = result.request
        if result.path is None and result.trajectory is None:
            status = getattr(result, "status", None)
            why = f"the plan {getattr(status, 'value', status)}" if status is not None else "the result"
            raise ValueError(f"{name}: {why} has no path or trajectory to show")
        if request.snapshot != snapshot.fingerprint():
            raise ValueError(f"{name}: the result was planned in another snapshot")
        if snapshot.model != self._world.world or snapshot.description != self._world.description:
            raise ValueError(f"{name}: the snapshot is of another world or description")
        if request.description != self._world.description:
            raise ValueError(f"{name}: the result is for another robot description")
        unknown = [j for j in request.joints if j not in self._qpos]
        if unknown:
            raise ValueError(f"{name}: the world has no joints {unknown}")
        if validation is not None:
            if validation.valid or validation.witness is None:
                raise ValueError(f"{name}: only an invalid validation has a state to show")
            validator = validation.validator
            if validator.snapshot != request.snapshot:
                raise ValueError(f"{name}: the validation is bound to another snapshot")
            if validator.group != request.group or tuple(validator.joints) != tuple(request.joints):
                raise ValueError(f"{name}: the validation is of another group or joints")
            if validation.motion not in {v.motion for v in result.validation}:
                raise ValueError(f"{name}: the validation is of another motion than this plan's")
        for attachment in snapshot.attachments:
            if attachment.held and self._held_frame(attachment) is None:
                raise ValueError(
                    f"{name}: no frame for {attachment.object!r}'s end effector {attachment.end_effector!r}"
                )
        stale = self._stale_reason(request)
        if stale is not None:
            raise ValueError(f"{name}: {stale}")
        with self._lock:
            self._previews[name] = _Preview(result, snapshot, validation)
            clients = list(self._clients.values())
        for client in clients:
            client.offer(self)

    def remove(self, name: str) -> None:
        """Withdraw a preview; browsers showing it clear their ghost."""
        with self._lock:
            self._previews.pop(name, None)
            clients = list(self._clients.values())
        for client in clients:
            client.offer(self)

    def names(self) -> list[str]:
        with self._lock:
            return list(self._previews)

    def stale(self, name: str) -> str | None:
        """Why a preview no longer describes the scene, or None."""
        with self._lock:
            return self._previews[name].stale

    # ------------------------------------------------------------------
    # PanelBase
    # ------------------------------------------------------------------

    def setup(self, gui: viser.GuiApi, viewer: MujocoViewer) -> None:
        # The ghost is drawn over this viewer: it must show the world the results are of.
        if self._live is not None and viewer is not self._live.viewer:
            raise ValueError("add the preview to its live view's viewer")
        if self._live is None and viewer.model is not self._model:
            raise ValueError("the viewer does not show this world's model")
        self._viewer = viewer
        server = viewer.server
        server.on_client_connect(self._connect)
        server.on_client_disconnect(lambda client: self._drop(client.client_id))
        for client in server.get_clients().values():
            self._connect(client)

    def on_sync(self, viewer: MujocoViewer) -> None:
        changed = False
        with self._lock:
            for preview in self._previews.values():
                stale = preview.stale or self._stale_reason(preview.result.request)
                changed |= stale != preview.stale
                preview.stale = stale
            clients = list(self._clients.values()) if changed else []
        for client in clients:
            client.render(self)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _stale_reason(self, request: Any) -> str | None:
        frame = None if self._live is None else self._live.frame
        if frame is not None and frame.revision != request.revision:
            return f"planned at scene revision {request.revision}; the scene is at {frame.revision}"
        return None

    def _connect(self, client: Any) -> None:
        panel = _ClientPreview(client, self._model)
        with self._lock:
            self._clients[client.client_id] = panel
        panel.offer(self)

    def _drop(self, client_id: int) -> None:
        with self._lock:
            self._clients.pop(client_id, None)

    def preview(self, name: str) -> _Preview | None:
        with self._lock:
            return self._previews.get(name)

    def ghost_state(self, preview: _Preview, configuration: tuple[float, ...]) -> mujoco.MjData:
        """The planning snapshot with the result's joints at ``configuration``, and held
        objects carried along, with kinematics computed."""
        model, data = self._model, mujoco.MjData(self._model)
        self._world.mapping.apply(preview.snapshot, data)
        for joint, value in zip(preview.result.request.joints, configuration, strict=True):
            data.qpos[self._qpos[joint]] = value
        mujoco.mj_kinematics(model, data)
        objects = dict(self._world.mapping.objects)
        held = [a for a in preview.snapshot.attachments if a.held and a.object in objects]
        for attachment in held:  # as the planning scene does: the object rides its end effector
            frame = self._held_frame(attachment)
            assert frame is not None  # checked when the preview was added
            pose = _frame_pose(model, data, frame) @ _transform(attachment.transform)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, pose[:3, :3].flatten())
            at = objects[attachment.object]
            data.qpos[at : at + 3] = pose[:3, 3]
            data.qpos[at + 3 : at + 7] = quat
        if held:
            mujoco.mj_kinematics(model, data)
        return data

    def ghost_bodies(self, preview: _Preview) -> list[int]:
        """The bodies a preview moves: the result's joints' subtrees and held objects."""
        model = self._model
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in preview.result.request.joints]
        objects = dict(self._world.mapping.objects)
        held = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, a.object)
            for a in preview.snapshot.attachments
            if a.held and a.object in objects
        ]
        return _moving_bodies(model, [j for j in joints if j >= 0], [b for b in held if b >= 0])

    def _held_frame(self, attachment: Any) -> str | None:
        """The MJCF body or site a held object rides: its end effector's frame."""
        frame = self._end_effectors.get(attachment.end_effector)
        if frame is None:
            return None
        for kind in (mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_SITE):
            if mujoco.mj_name2id(self._model, kind, frame) >= 0:
                return frame
        return None


def _transform(pose: Any) -> np.ndarray:
    out = np.eye(4)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(pose.quat_wxyz, float))
    out[:3, :3] = rotation.reshape(3, 3)
    out[:3, 3] = pose.position
    return out


def _frame_pose(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    pose = np.eye(4)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body >= 0:
        pose[:3, :3] = data.xmat[body].reshape(3, 3)
        pose[:3, 3] = data.xpos[body]
        return pose
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site >= 0:
        pose[:3, :3] = data.site_xmat[site].reshape(3, 3)
        pose[:3, 3] = data.site_xpos[site]
        return pose
    raise ValueError(f"the model has no body or site {name!r}")


def _moving_bodies(model: mujoco.MjModel, joints: list[int], held: list[int]) -> list[int]:
    """Bodies a plan moves: those at or below a body owning one of its joints, and the
    held objects' bodies."""
    owners = {int(model.jnt_bodyid[j]) for j in joints}
    out = []
    for b in range(1, model.nbody):
        a = b
        while a > 0 and a not in owners:
            a = int(model.body_parentid[a])
        if a in owners:
            out.append(b)
    return sorted(set(out) | set(held))


class _Ghost:
    """A translucent copy of the bodies a plan moves, in one scene (a browser's own,
    or a recording's)."""

    def __init__(self, scene: Any, model: mujoco.MjModel, root: str = "/plan_preview") -> None:
        self._scene = scene
        self._model = model
        self._root = root
        self._frames: dict[int, Any] = {}
        self._rgba: np.ndarray | None = None

    def show(self, data: mujoco.MjData, bodies: list[int], rgba: np.ndarray) -> None:
        if self._rgba is None or not np.array_equal(self._rgba, rgba) or set(self._frames) != set(bodies):
            self._build(bodies, rgba)
        for b, frame in self._frames.items():
            frame.position = data.xpos[b]
            frame.wxyz = data.xquat[b]
            frame.visible = True

    def hide(self) -> None:
        for frame in self._frames.values():
            frame.visible = False

    def _build(self, bodies: list[int], rgba: np.ndarray) -> None:
        for frame in self._frames.values():
            frame.remove()
        self._frames = {}
        model = self._model
        for b in bodies:
            frame = self._scene.add_frame(f"{self._root}/b{b}", show_axes=False)
            for g in range(model.ngeom):
                if model.geom_bodyid[g] != b or model.geom_group[g] > 2:
                    continue
                path = f"{self._root}/b{b}/g{g}"
                kind = int(model.geom_type[g])
                if kind in PRIMITIVE_TYPES:
                    handle = build_primitive(self._scene, path, kind, model.geom_size[g], rgba)
                elif kind == mujoco.mjtGeom.mjGEOM_MESH:
                    vertices, faces = extract_mujoco_mesh(model, model.geom_dataid[g])
                    handle = _simple_mesh(self._scene, path, rgba, vertices, faces, side="double")
                else:
                    continue
                handle.position = model.geom_pos[g]
                handle.wxyz = model.geom_quat[g]
            self._frames[b] = frame
        self._rgba = rgba


class _ClientPreview:
    """One browser's choice, scrubber, and ghost."""

    def __init__(self, client: Any, model: mujoco.MjModel) -> None:
        self._client = client
        self._model = model
        # GUI callbacks and preview updates arrive on different threads. Setting a widget
        # from here fires its callback synchronously; those re-entries are ignored.
        self._lock = threading.RLock()
        self._rendering = False
        gui = client.gui
        with gui.add_folder("Plan preview", order=20):
            self.choice = gui.add_dropdown("Result", ("(none)",), initial_value="(none)")
            self.mode = gui.add_dropdown("Show", (_PATH,), initial_value=_PATH)
            self.position = gui.add_slider("Sample", min=0, max=1, step=1, initial_value=0)
            self.caption = gui.add_html("")
        self._ghost = _Ghost(client.scene, model)
        self._preview: PlanPreview | None = None
        for handle in (self.choice, self.mode, self.position):
            handle.on_update(lambda _event: self._changed())

    def offer(self, preview: PlanPreview) -> None:
        self._preview = preview
        with self._lock:
            names = ("(none)", *preview.names())
            if self.choice.options != names:
                self._rendering = True
                try:
                    self.choice.options = names
                finally:
                    self._rendering = False
        self.render(preview)

    def _changed(self) -> None:
        if self._preview is not None and not self._rendering:
            self.render(self._preview)

    def render(self, preview: PlanPreview) -> None:
        with self._lock:
            self._rendering = True
            try:
                self._render(preview)
            finally:
                self._rendering = False

    def _render(self, preview: PlanPreview) -> None:
        chosen = preview.preview(self.choice.value) if self.choice.value != "(none)" else None
        if chosen is None:
            self._ghost.hide()
            self.caption.content = ""
            return
        result = chosen.result
        modes = tuple(
            m
            for m, present in (
                (_PATH, result.path is not None),
                (_TRAJECTORY, result.trajectory is not None),
                (_WITNESS, chosen.validation is not None),
            )
            if present
        )
        if self.mode.options != modes:
            self.mode.options = modes
        mode = self.mode.value if self.mode.value in modes else modes[0]
        if chosen.stale is not None:
            self._ghost.hide()
            self.caption.content = _caption(["stale: " + chosen.stale, "nothing shown"])
            return
        configuration, lines = self._sample(chosen, mode)
        rgba = WITNESS_RGBA if mode == _WITNESS else GHOST_RGBA
        self._ghost.show(preview.ghost_state(chosen, configuration), preview.ghost_bodies(chosen), rgba)
        self.caption.content = _caption(lines)

    def _sample(self, chosen: _Preview, mode: str) -> tuple[tuple[float, ...], list[str]]:
        result = chosen.result
        if mode == _PATH:
            waypoints = result.path.waypoints
            self._set_range(len(waypoints) - 1, 1)
            i = min(int(self.position.value), len(waypoints) - 1)
            return tuple(waypoints[i]), [f"path waypoint {i + 1} of {len(waypoints)}", "the planner's own waypoint"]
        if mode == _TRAJECTORY:
            trajectory = result.trajectory
            # Ticks of at most 10 ms; the last tick is exactly the trajectory's end.
            ticks = max(1, math.ceil(trajectory.duration * 100))
            self._set_range(ticks, 1)
            t = trajectory.duration * min(int(self.position.value), ticks) / ticks
            positions, _, _ = trajectory.evaluate(t)
            return tuple(positions), [
                f"trajectory at t = {t:.2f} s of {trajectory.duration:.2f} s",
                "evaluated exactly from the planner's timing law",
            ]
        witness = chosen.validation.witness
        self._set_range(0, 1)
        where = (
            f"at t = {witness.time_s:.3f} s"
            if witness.time_s is not None
            else f"in segment {witness.segment} at {witness.fraction:.3f}"
            if witness.segment is not None
            else "at the state"
        )
        contacts = ", ".join(f"{c.first}/{c.second} {c.distance * 1000:.2f} mm" for c in witness.contacts)
        lines = [f"first invalid state: {_enum(witness.failure)} {where}", "as the validator reported it"]
        if contacts:
            lines.append(f"contacts: {contacts}")
        if witness.detail:
            lines.append(witness.detail)
        return tuple(witness.configuration), lines

    def _set_range(self, top: int, step: int) -> None:
        top = max(top, 0)
        if self.position.max != max(top, 1):
            self.position.max = max(top, 1)
        if self.position.step != step:
            self.position.step = step
        if self.position.value > top:
            self.position.value = top


def record_preview(
    result: Any,
    snapshot: Any,
    world: Any,
    description: Any,
    *,
    period_ns: int = 20_000_000,
    waypoint_s: float = 0.5,
) -> tuple[bytes, dict[str, Any]]:
    """Record a preview as a Viser recording: the planning snapshot, then the ghost at
    each path waypoint (``waypoint_s`` apart), then, ``waypoint_s`` later, at each of
    the trajectory's samples (``result.trajectory.sample(period_ns)``, the planner's
    timing law at each tick) at its own time from the trajectory's start.

    Returns the recording and a report of every sample shown.

    Raises:
        ValueError: As :meth:`PlanPreview.add`.
    """
    from mj_viser.viewer import MujocoViewer

    preview = PlanPreview(world, description)
    preview.add("result", result, snapshot)
    chosen = preview.preview("result")
    assert chosen is not None
    model = world.model
    viewer = MujocoViewer(model, port=0, show_visibility=False)
    viewer.open()
    try:
        base = mujoco.MjData(model)
        world.mapping.apply(snapshot, base)
        viewer.update(base)
        server = viewer.server
        recorder = server.get_scene_serializer()
        ghost = _Ghost(server.scene, model)
        anchor = np.asarray(model.stat.center, float) + [0.0, 0.0, 0.75 * float(model.stat.extent)]
        label = server.scene.add_label("/plan_preview/label", "", position=anchor, anchor="bottom-left")
        report: dict[str, Any] = {
            "snapshot": snapshot.fingerprint(),
            "world": world.world,
            "path": [],
            "trajectory": [],
        }
        bodies = preview.ghost_bodies(chosen)
        clock = 0.0  # recording time, advanced exactly as the recorder's

        def wait(seconds: float) -> None:
            nonlocal clock
            recorder.insert_sleep(seconds)
            clock += seconds

        if result.path is not None:
            waypoints = result.path.waypoints
            for i, waypoint in enumerate(waypoints):
                if i:
                    wait(waypoint_s)
                ghost.show(preview.ghost_state(chosen, tuple(waypoint)), bodies, GHOST_RGBA)
                label.text = f"path waypoint {i + 1} of {len(waypoints)}"
                report["path"].append({"index": i, "recorded_s": clock})
        if result.trajectory is not None:
            if result.path is not None:
                wait(waypoint_s)
            command = result.trajectory.sample(period_ns)
            previous = None
            for t, positions in zip(command.time_from_start_ns, command.positions, strict=True):
                if previous is not None:
                    wait((t - previous) / 1e9)  # the samples' own spacing; none after the last
                ghost.show(preview.ghost_state(chosen, tuple(positions)), bodies, GHOST_RGBA)
                label.text = f"trajectory at t = {t / 1e9:.3f} s"
                report["trajectory"].append({"time_ns": int(t), "recorded_s": clock})
                previous = t
        recording = recorder.serialize()
    finally:
        viewer.close()
    return recording, report


def _caption(lines: list[str]) -> str:
    return "<br>".join(html.escape(line) for line in lines)


def _enum(value: Any) -> str:
    return str(getattr(value, "value", value))
