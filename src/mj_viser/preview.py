# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Preview joint motions as a ghost robot, per browser, without touching anything.

A :class:`MotionPreview` shows named :class:`JointMotion` s over a viewer: each browser
picks one and scrubs its waypoints, or its timed motion at exact times, and sees a
translucent ghost of the bodies it moves in its own scene. A motion may also carry a
highlighted state, such as the first invalid state a validator reported. Nothing is
submitted, cancelled, or changed. :func:`record_motion` writes a motion as a native
Viser recording. ``mj_viser.integrations.ssrobot.PlanPreview`` previews ssrobot plans.
"""

from __future__ import annotations

import html
import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from mj_viser.geom_builders import PRIMITIVE_TYPES, _simple_mesh, build_primitive
from mj_viser.mesh_utils import extract_mujoco_mesh
from mj_viser.panels import PanelBase

if TYPE_CHECKING:
    import viser

    from mj_viser.viewer import MujocoViewer

GHOST_RGBA = np.array([0.3, 0.6, 1.0, 0.35])
HIGHLIGHT_RGBA = np.array([1.0, 0.2, 0.2, 0.5])
_PATH, _TIMED, _HIGHLIGHT, _NONE = "path", "trajectory", "highlight", "(none)"


def _check_configuration(configuration: Sequence[float], width: int, what: str) -> tuple[float, ...]:
    """``configuration`` as floats, if it has ``width`` finite values."""
    values = tuple(float(v) for v in configuration)
    if len(values) != width:
        raise ValueError(f"{what}: {len(values)} values for {width} joints")
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{what}: values must be finite, got {values}")
    return values


@dataclass(frozen=True)
class Highlight:
    """A configuration to show apart from the motion, with what it means."""

    configuration: tuple[float, ...]
    lines: tuple[str, ...]
    """Caption lines, e.g. what failed there and where."""
    label: str = "highlighted state"
    """The name browsers choose it by; not ``"path"`` or ``"trajectory"``."""

    def __post_init__(self) -> None:
        if self.label in (_PATH, _TIMED):
            raise ValueError(f"a highlight cannot be labelled {self.label!r}, a mode of its own")
        _check_configuration(self.configuration, len(self.configuration), "the highlight")


@dataclass(frozen=True)
class JointMotion:
    """A motion of named joints, to preview.

    ``state(configuration, data)`` writes the whole state to show for a configuration of
    ``joints`` into ``data`` (an ``MjData`` of the preview's model): everything that does
    not move, the joints, and whatever moves with them. It may compute kinematics; the
    preview computes them again after.
    """

    joints: tuple[str, ...]
    state: Callable[[Sequence[float], mujoco.MjData], None]
    waypoints: tuple[tuple[float, ...], ...] | None = None
    """The path's own configurations, shown one at a time, never interpolated."""
    duration_s: float | None = None
    """How long the timed motion lasts; ``at`` gives its configuration at any time."""
    at: Callable[[float], Sequence[float]] | None = None
    samples: Callable[[int], Sequence[tuple[int, Sequence[float]]]] | None = None
    """For recordings: the timed motion's own (time_ns, configuration) samples at a period."""
    carried: tuple[str, ...] = ()
    """Bodies that move with the motion besides the joints' subtrees (held objects)."""
    highlight: Highlight | None = None
    notes: dict[str, str] = field(default_factory=dict)
    """A caption line per mode (``"path"``, ``"trajectory"``), saying what is shown."""

    def __post_init__(self) -> None:
        if self.waypoints is not None and len(self.waypoints) == 0:
            object.__setattr__(self, "waypoints", None)  # an empty path is no path
        if self.waypoints is None and self.at is None:
            raise ValueError("a motion needs waypoints or a timed motion to show")
        if (self.at is None) != (self.duration_s is None):
            raise ValueError("a timed motion needs both at and duration_s")
        if self.duration_s is not None and not (math.isfinite(self.duration_s) and self.duration_s >= 0):
            raise ValueError(f"duration_s must be finite and not negative, got {self.duration_s!r}")
        width = len(self.joints)
        for i, q in enumerate(self.waypoints or ()):
            _check_configuration(q, width, f"waypoint {i}")
        if self.highlight is not None:
            _check_configuration(self.highlight.configuration, width, "the highlight")


class MotionPreview(PanelBase):
    """Ghost previews of joint motions, chosen and scrubbed per browser.

    Add it to the viewer it previews over (``viewer.add_panel(preview)``), whose model
    must be ``model``, then :meth:`add` motions.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        self._model = model
        self._motions: dict[str, JointMotion] = {}
        self._stale: dict[str, str] = {}
        self._clients: dict[int, _ClientPreview] = {}
        self._lock = threading.Lock()
        self._viewer: MujocoViewer | None = None

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    def name(self) -> str:
        return "Motion preview"

    def add(self, name: str, motion: JointMotion) -> None:
        """Offer a motion for preview, replacing one of the same name.

        Raises:
            ValueError: If the model has no joint or carried body of that name, or the
                name is ``"(none)"``, which browsers use for no choice.
        """
        if name == _NONE:
            raise ValueError(f"{_NONE!r} is reserved for choosing no motion")
        missing = [j for j in motion.joints if mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, j) < 0]
        missing += [b for b in motion.carried if mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, b) < 0]
        if missing:
            raise ValueError(f"{name}: the model has no joints or bodies {missing}")
        with self._lock:
            self._motions[name] = motion
            self._stale.pop(name, None)
        self._offer_all()

    def remove(self, name: str) -> None:
        """Withdraw a motion; browsers showing it clear their ghost."""
        with self._lock:
            self._motions.pop(name, None)
            self._stale.pop(name, None)
        self._offer_all()

    def set_stale(self, name: str, reason: str) -> None:
        """Mark a motion as no longer describing the scene: browsers hide its ghost."""
        with self._lock:
            if name not in self._motions or self._stale.get(name) == reason:
                return
            self._stale[name] = reason
            clients = list(self._clients.values())
        for client in clients:
            client.render(self)

    def stale(self, name: str) -> str | None:
        """Why a motion no longer describes the scene, or None."""
        with self._lock:
            return self._stale.get(name)

    def names(self) -> list[str]:
        with self._lock:
            return list(self._motions)

    def motion(self, name: str) -> JointMotion | None:
        with self._lock:
            return self._motions.get(name)

    def ghost_state(self, motion: JointMotion, configuration: Sequence[float]) -> mujoco.MjData:
        """The state to show for ``configuration``, with kinematics computed.

        Raises:
            ValueError: If ``configuration`` is not one finite value per joint.
        """
        configuration = _check_configuration(configuration, len(motion.joints), "a configuration")
        data = mujoco.MjData(self._model)
        motion.state(configuration, data)
        mujoco.mj_kinematics(self._model, data)
        return data

    def ghost_bodies(self, motion: JointMotion) -> list[int]:
        """The bodies a motion moves: its joints' subtrees and the carried bodies."""
        model = self._model
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in motion.joints]
        carried = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in motion.carried]
        return _moving_bodies(model, joints, carried)

    # -- PanelBase -------------------------------------------------------------------

    def setup(self, gui: viser.GuiApi, viewer: MujocoViewer) -> None:
        self._accept_viewer(viewer)
        self._viewer = viewer
        server = viewer.server
        server.on_client_connect(self._connect)
        server.on_client_disconnect(lambda client: self._drop(client.client_id))
        for client in server.get_clients().values():
            self._connect(client)

    def _accept_viewer(self, viewer: MujocoViewer) -> None:
        """The ghost is drawn over this viewer: it must show the preview's model."""
        if viewer.model is not self._model:
            raise ValueError("the viewer does not show this preview's model")

    # -- internal --------------------------------------------------------------------

    def _offer_all(self) -> None:
        with self._lock:
            clients = list(self._clients.values())
        for client in clients:
            client.offer(self)

    def _connect(self, client: Any) -> None:
        panel = _ClientPreview(client, self._model, self.name())
        with self._lock:
            self._clients[client.client_id] = panel
        panel.offer(self)

    def _drop(self, client_id: int) -> None:
        with self._lock:
            self._clients.pop(client_id, None)


def _moving_bodies(model: mujoco.MjModel, joints: list[int], carried: list[int]) -> list[int]:
    """Bodies a motion moves: those at or below a body owning one of its joints, or at or
    below a carried body."""
    owners = {int(model.jnt_bodyid[j]) for j in joints} | set(carried)
    out = []
    for b in range(1, model.nbody):
        a = b
        while a > 0 and a not in owners:
            a = int(model.body_parentid[a])
        if a in owners:
            out.append(b)
    return sorted(out)


class _Ghost:
    """A translucent copy of the bodies a plan moves, in one scene (a browser's own,
    or a recording's)."""

    def __init__(self, scene: Any, model: mujoco.MjModel, root: str = "/motion_preview") -> None:
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

    def __init__(self, client: Any, model: mujoco.MjModel, title: str) -> None:
        self._client = client
        # GUI callbacks and preview updates arrive on different threads. Setting a widget
        # from here fires its callback synchronously; those re-entries are ignored.
        self._lock = threading.RLock()
        self._rendering = False
        gui = client.gui
        with gui.add_folder(title, order=20):
            self.choice = gui.add_dropdown("Motion", (_NONE,), initial_value=_NONE)
            self.mode = gui.add_dropdown("Show", (_PATH,), initial_value=_PATH)
            self.position = gui.add_slider("Sample", min=0, max=1, step=1, initial_value=0)
            self.caption = gui.add_html("")
        self._ghost = _Ghost(client.scene, model)
        self._preview: MotionPreview | None = None
        for handle in (self.choice, self.mode, self.position):
            handle.on_update(lambda _event: self._changed())

    def offer(self, preview: MotionPreview) -> None:
        self._preview = preview
        with self._lock:
            names = (_NONE, *preview.names())
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

    def render(self, preview: MotionPreview) -> None:
        with self._lock:
            self._rendering = True
            try:
                self._render(preview)
            finally:
                self._rendering = False

    def _render(self, preview: MotionPreview) -> None:
        name = self.choice.value
        motion = preview.motion(name) if name != _NONE else None
        if motion is None:
            self._ghost.hide()
            self.caption.content = ""
            return
        highlight = motion.highlight
        modes = tuple(
            m
            for m, present in (
                (_PATH, motion.waypoints is not None),
                (_TIMED, motion.at is not None),
                (highlight.label if highlight else _HIGHLIGHT, highlight is not None),
            )
            if present
        )
        if self.mode.options != modes:
            self.mode.options = modes
        mode = self.mode.value if self.mode.value in modes else modes[0]
        stale = preview.stale(name)
        if stale is not None:
            self._ghost.hide()
            self.caption.content = _caption(["stale: " + stale, "nothing shown"])
            return
        configuration, lines = self._sample(motion, mode)
        rgba = GHOST_RGBA if mode in (_PATH, _TIMED) else HIGHLIGHT_RGBA
        self._ghost.show(preview.ghost_state(motion, configuration), preview.ghost_bodies(motion), rgba)
        self.caption.content = _caption(lines)

    def _sample(self, motion: JointMotion, mode: str) -> tuple[tuple[float, ...], list[str]]:
        notes = [motion.notes[mode]] if mode in motion.notes else []
        if mode == _PATH:
            assert motion.waypoints is not None
            waypoints = motion.waypoints
            self._set_range(len(waypoints) - 1)
            i = min(int(self.position.value), len(waypoints) - 1)
            return tuple(waypoints[i]), [f"path waypoint {i + 1} of {len(waypoints)}", *notes]
        if mode == _TIMED:
            assert motion.at is not None and motion.duration_s is not None
            duration = motion.duration_s
            # Ticks of at most 10 ms; the last tick is exactly the motion's end.
            ticks = max(1, math.ceil(duration * 100))
            self._set_range(ticks)
            t = duration * min(int(self.position.value), ticks) / ticks
            return tuple(motion.at(t)), [f"trajectory at t = {t:.2f} s of {duration:.2f} s", *notes]
        assert motion.highlight is not None
        self._set_range(0)
        return motion.highlight.configuration, list(motion.highlight.lines)

    def _set_range(self, top: int) -> None:
        top = max(top, 0)
        if self.position.max != max(top, 1):
            self.position.max = max(top, 1)
        if self.position.value > top:
            self.position.value = top


def record_motion(
    model: mujoco.MjModel,
    motion: JointMotion,
    *,
    period_ns: int = 20_000_000,
    waypoint_s: float = 0.5,
    label: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Record a motion as a Viser recording: the state at its first configuration, then
    the ghost at each waypoint (``waypoint_s`` apart), then, ``waypoint_s`` later, the
    timed motion from its start: each of its own samples (``motion.samples(period_ns)``)
    at its own time from the motion's start, nothing after the last.

    Returns the recording and a report of every sample shown.

    Raises:
        ValueError: Before anything is recorded, if ``period_ns`` is not positive,
            ``waypoint_s`` is negative or not finite, the samples' times are negative or
            not strictly increasing, or a configuration is not one finite value per joint.
    """
    from mj_viser.recording import TEXT
    from mj_viser.viewer import MujocoViewer

    if period_ns <= 0:
        raise ValueError(f"period_ns must be positive, got {period_ns}")
    if not (math.isfinite(waypoint_s) and waypoint_s >= 0):
        raise ValueError(f"waypoint_s must be finite and not negative, got {waypoint_s}")
    width = len(motion.joints)
    timeline: list[tuple[int, tuple[float, ...]]] = []
    if motion.samples is not None:
        for t, q in motion.samples(period_ns):
            if t < 0 or (timeline and t <= timeline[-1][0]):
                raise ValueError(f"sample times must start at or after 0 and increase: {t} ns")
            timeline.append((int(t), _check_configuration(q, width, f"the sample at {t} ns")))
    preview = MotionPreview(model)
    if motion.waypoints:
        first = motion.waypoints[0]
    elif timeline:
        first = timeline[0][1]
    else:
        assert motion.at is not None
        first = _check_configuration(motion.at(0.0), width, "the motion's start")
    viewer = MujocoViewer(model, port=0, label=label, show_visibility=False)
    viewer.open()
    try:
        viewer.update(preview.ghost_state(motion, first))
        server = viewer.server
        recorder = server.get_scene_serializer()
        ghost = _Ghost(server.scene, model)
        anchor = np.asarray(model.stat.center, float) + [0.0, 0.0, 0.75 * float(model.stat.extent)]
        text = server.scene.add_label(TEXT, "", position=anchor, anchor="bottom-left")
        report: dict[str, Any] = {"path": [], "trajectory": []}
        bodies = preview.ghost_bodies(motion)
        clock = 0.0  # recording time, advanced exactly as the recorder's

        def wait(seconds: float) -> None:
            nonlocal clock
            recorder.insert_sleep(seconds)
            clock += seconds

        if motion.waypoints is not None:
            for i, waypoint in enumerate(motion.waypoints):
                if i:
                    wait(waypoint_s)
                ghost.show(preview.ghost_state(motion, waypoint), bodies, GHOST_RGBA)
                text.text = f"path waypoint {i + 1} of {len(motion.waypoints)}"
                report["path"].append({"index": i, "recorded_s": clock})
        if timeline:
            if motion.waypoints is not None:
                wait(waypoint_s)
            previous = 0  # the timed motion starts here; a first sample later than 0 waits
            for t, positions in timeline:
                if t > previous:
                    wait((t - previous) / 1e9)  # the samples' own spacing; none after the last
                ghost.show(preview.ghost_state(motion, positions), bodies, GHOST_RGBA)
                text.text = f"trajectory at t = {t / 1e9:.3f} s"
                report["trajectory"].append({"time_ns": int(t), "recorded_s": clock})
                previous = t
        recording = recorder.serialize()
    finally:
        viewer.close()
    return recording, report


def _caption(lines: list[str]) -> str:
    return "<br>".join(html.escape(line) for line in lines)
