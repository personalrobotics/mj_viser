# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""End-to-end check of an installed mj_viser wheel. Run by scripts/smoke_wheel.sh.

Imports every mj_viser module, then exercises the viewer lifecycle: renders two
deterministic caller-owned states and records them to a .viser file, proves the
source states are unchanged, and checks loopback default, port 0, repeated
open/close, and startup/update failures. Writes a JSON report and the recording
into the directory given as argv[1]; argv[2] names the environment in the report.
"""

from __future__ import annotations

import faulthandler
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import os
import pkgutil
import socket
import sys
import tokenize
import urllib.parse
from importlib.metadata import version

import mujoco
import numpy as np

import mj_viser

# A startup hang is one of the failures under test; never let it stall CI.
faulthandler.dump_traceback_later(60, exit=True)

MJCF = """
<mujoco>
  <asset>
    <texture name="checker" type="2d" builtin="checker" width="8" height="8" rgb1="1 1 1" rgb2="0.2 0.2 0.2"/>
    <material name="checkered" texture="checker"/>
    <mesh name="tet" vertex="0 0 0  0.2 0 0  0 0.2 0  0 0 0.2" texcoord="0 0  1 0  0 1  1 1"
          face="0 2 1  0 1 3  0 3 2  1 2 3"/>
  </asset>
  <worldbody>
    <geom name="tet" type="mesh" mesh="tet" material="checkered" pos="-0.5 0 0"/>
    <body name="arm" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom name="link" type="capsule" size="0.05" fromto="0 0 0 0.3 0 0" rgba="0.8 0.3 0.3 1"/>
      <body name="slider" pos="0.3 0 0">
        <joint name="slide" type="slide" axis="1 0 0"/>
        <geom name="tip" type="box" size="0.04 0.04 0.04" rgba="0.3 0.3 0.8 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

STATES = {"A": [0.3, 0.1], "B": [-0.7, 0.25]}

# Fields an update could plausibly touch: state, controls, derived kinematics, sensors.
HASHED_FIELDS = ("time", "qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "xpos", "geom_xpos", "sensordata")


def state_hash(data: mujoco.MjData) -> str:
    h = hashlib.sha256()
    for field in HASHED_FIELDS:
        h.update(field.encode())
        h.update(np.ascontiguousarray(getattr(data, field)).tobytes())
    return h.hexdigest()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def rendered_positions(viewer: mj_viser.MujocoViewer) -> dict[int, list[float]]:
    # Read back the positions on the Viser scene nodes themselves: the oracle for "what the browser shows".
    handles = viewer._scene_mgr._geom_handles  # type: ignore[union-attr]
    return {gid: [round(float(v), 9) for v in h.position] for gid, h in handles.items()}


def expected_positions(model: mujoco.MjModel, qpos: list[float]) -> dict[int, list[float]]:
    # Independent reference: MuJoCo kinematics on a fresh MjData, mapped MuJoCo z-up == Viser z-up.
    ref = mujoco.MjData(model)
    ref.qpos[:] = qpos
    mujoco.mj_kinematics(model, ref)
    return {gid: [round(float(v), 9) for v in ref.geom_xpos[gid]] for gid in range(model.ngeom)}


def rendered_orientations(viewer: mj_viser.MujocoViewer) -> dict[int, np.ndarray]:
    handles = viewer._scene_mgr._geom_handles  # type: ignore[union-attr]
    return {gid: np.asarray(h.wxyz, dtype=float) for gid, h in handles.items()}


def expected_orientations(model: mujoco.MjModel, qpos: list[float]) -> dict[int, np.ndarray]:
    # Independent reference: MuJoCo's own matrix-to-quaternion on fresh kinematics.
    ref = mujoco.MjData(model)
    ref.qpos[:] = qpos
    mujoco.mj_kinematics(model, ref)
    quats = {}
    for gid in range(model.ngeom):
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, ref.geom_xmat[gid])
        quats[gid] = q
    return quats


def same_rotation(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.allclose(a, b, atol=1e-6) or np.allclose(a, -b, atol=1e-6))


def expect_raises(exc: type[BaseException], fn) -> str:  # type: ignore[no-untyped-def]
    try:
        fn()
    except exc as e:
        return f"{type(e).__name__}: {e}"
    raise SystemExit(f"expected {exc.__name__} from {fn}")


def main(out_dir: str, env_label: str) -> None:
    modules = sorted(m.name for m in pkgutil.walk_packages(mj_viser.__path__, "mj_viser."))
    for name in modules:
        importlib.import_module(name)
    forbidden = [m for m in sys.modules if m.split(".")[0] == "mj_manipulator"]
    if forbidden:
        raise SystemExit(f"mj_viser imported robot-control modules: {forbidden}")

    # The viewer code (not its docstrings) must contain no stepping, reset, or worker thread of its own.
    # Nor may it reach into Viser's private modules or transport (viser._x, ._websock_*).
    names = set()
    private_viser = []
    for m in modules:
        src = io.StringIO(inspect.getsource(sys.modules[m]))
        toks = [t.string for t in tokenize.generate_tokens(src.readline) if t.type in (tokenize.NAME, tokenize.OP)]
        names |= set(toks)
        private_viser += [
            f"{m}: viser.{c}"
            for a, b, c in zip(toks, toks[1:], toks[2:])
            if a == "viser" and b == "." and c.startswith("_")
        ]
        private_viser += [f"{m}: {t}" for t in toks if t.startswith("_websock")]
    banned = sorted(names & {"mj_step", "mj_resetData", "Thread"})
    if banned:
        raise SystemExit(f"viewer lifecycle contains {banned}")
    if private_viser:
        raise SystemExit(f"private Viser access: {private_viser}")
    if any(hasattr(mj_viser.MujocoViewer, a) for a in ("launch", "launch_passive", "sync", "is_running")):
        raise SystemExit("legacy launch/sync API still present")

    model = mujoco.MjModel.from_xml_string(MJCF)
    report: dict[str, object] = {}

    # --- Two caller-owned states, rendered in order and recorded --------------------------
    sources = {}
    for name, qpos in STATES.items():
        d = mujoco.MjData(model)
        d.qpos[:] = qpos
        d.time = {"A": 1.0, "B": 2.0}[name]
        if name == "A":
            mujoco.mj_forward(model, d)  # B deliberately skips kinematics: the viewer must compute them itself.
        sources[name] = d

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

    serializer = viewer.server.get_scene_serializer()
    states_report = {}
    for name, d in sources.items():
        before = state_hash(d)
        viewer.update(d)
        after = state_hash(d)
        if before != after:
            raise SystemExit(f"update() modified source state {name}")
        rendered = rendered_positions(viewer)
        expected = expected_positions(model, STATES[name])
        if set(rendered) != set(expected) or not all(
            np.allclose(rendered[g], expected[g], atol=1e-6) for g in rendered
        ):
            raise SystemExit(f"state {name}: rendered {rendered} != expected {expected}")
        rendered_q = rendered_orientations(viewer)
        expected_q = expected_orientations(model, STATES[name])
        if not all(same_rotation(rendered_q[g], expected_q[g]) for g in rendered_q):
            raise SystemExit(f"state {name}: rendered orientations {rendered_q} != expected {expected_q}")
        if viewer.data is d:
            raise SystemExit("viewer retained the caller's MjData")
        states_report[name] = {
            "qpos": STATES[name],
            "source_sha256_before": before,
            "source_sha256_after": after,
            "rendered_geom_positions": {model.geom(g).name: p for g, p in rendered.items()},
        }
        serializer.insert_sleep(0.5)
    recording = serializer.serialize()
    viewer.close()
    if viewer.is_open:
        raise SystemExit("viewer still open after close()")
    rec_path = os.path.join(out_dir, "two_states.viser")
    with open(rec_path, "wb") as f:
        f.write(recording)
    report["states"] = states_report
    report["recording"] = {"file": "two_states.viser", "sha256": hashlib.sha256(recording).hexdigest()}

    # --- Update failures return to the caller ---------------------------------------------
    # Adversarial models: every array the old size check compared is the same length.
    # "swapped" is the same layout with hinge and slide exchanged (0.5 m would render as 0.5 rad);
    # "camera" adds a camera, changing MjData's internal layout so mj_copyData would raise FatalError.
    adversarial = {
        "swapped": MJCF.replace('type="hinge"', 'type="TMP"')
        .replace('type="slide"', 'type="hinge"')
        .replace('type="TMP"', 'type="slide"'),
        "camera": MJCF.replace("<worldbody>", '<worldbody><camera name="cam" pos="0 0 2"/>'),
    }
    report["update_while_closed"] = expect_raises(RuntimeError, lambda: viewer.update(sources["A"]))
    with mj_viser.MujocoViewer(model, port=0) as v:
        v.update(sources["A"])
        good = rendered_positions(v)
        for name, xml in adversarial.items():
            m = mujoco.MjModel.from_xml_string(xml)
            if (m.nq, m.nv, m.nbody, m.ngeom, m.nsensordata) != (model.nq, model.nv, model.nbody, model.ngeom, 0):
                raise SystemExit(f"adversarial model {name} is not size-identical")
            d = mujoco.MjData(m)
            d.qpos[:] = [0.5, 0.5]
            report[f"update_{name}_model"] = expect_raises(ValueError, lambda: v.update(d))
            if rendered_positions(v) != good:
                raise SystemExit(f"rejected {name} update changed the rendered scene")
        report["open_twice"] = expect_raises(RuntimeError, v.open)

    # --- IPv6 literal hosts produce a bracketed URL ---------------------------------------
    try:
        with socket.socket(socket.AF_INET6) as s6:
            s6.bind(("::1", 0))
    except OSError:
        report["ipv6_url"] = "skipped: ::1 not bindable on this host"
    else:
        with mj_viser.MujocoViewer(model, host="::1", port=0) as v:
            parsed = urllib.parse.urlsplit(v.url)
            if parsed.hostname != "::1" or not parsed.port:
                raise SystemExit(f"invalid IPv6 URL {v.url}")
            report["ipv6_url"] = v.url.replace(str(parsed.port), "<port>")

    # --- Repeated open/close on one fixed port releases it each time ----------------------
    fixed = free_port()
    v = mj_viser.MujocoViewer(model, port=fixed)
    for _ in range(3):
        if v.open() != f"http://127.0.0.1:{fixed}":
            raise SystemExit("reopen did not bind the requested port")
        v.update(sources["B"])
        v.close()
    v.close()  # idempotent
    with socket.socket() as s:
        s.bind(("127.0.0.1", fixed))  # port is free again
    report["open_close_cycles_on_fixed_port"] = 3

    # --- Startup failures return to the caller instead of hanging -------------------------
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        taken = busy.getsockname()[1]
        blocked = mj_viser.MujocoViewer(model, port=taken)
        report["startup_port_in_use"] = expect_raises(OSError, blocked.open).replace(str(taken), "<port>")
        if blocked.is_open:
            raise SystemExit("viewer reports open after failed startup")
    # 192.0.2.1 is TEST-NET-1 (RFC 5737): never assigned to a local interface.
    report["startup_unbindable_host"] = expect_raises(OSError, mj_viser.MujocoViewer(model, host="192.0.2.1").open)

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
