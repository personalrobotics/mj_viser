# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Capture the README's clips from real mj_viser runs in a headless browser.

Usage: python -I scripts/media/capture.py MENAGERIE_DIR REFERENCES_DIR OUT_DIR [CLIP ...]

Each clip is a real viewer session, recorded in headless Chrome as it runs, then trimmed
and encoded by ffmpeg as OUT_DIR/<clip>.webp (animated) and OUT_DIR/<clip>.jpg (a still).
scripts/media/README.md lists the inputs and how to reproduce them.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import mujoco
import numpy as np
from playwright.sync_api import Page, sync_playwright

from mj_viser import MujocoViewer

SIZE = (960, 540)  # browser viewport, CSS pixels
WIDTH = 640  # encoded clip width
FPS = 15


@dataclass
class Clip:
    page: Page
    opened: float
    start: float | None = None
    end: float | None = None

    def begin(self) -> None:
        """What happens from now on is the clip."""
        self.start = time.monotonic()

    def finish(self) -> None:
        self.end = time.monotonic()


@contextmanager
def recorded(
    url: str,
    out: Path,
    name: str,
    *,
    label: str = "",
    panel: bool = True,
    collapse: tuple[str, ...] = (),
    plot: str = "",
    wait_for: str = "",
    size: tuple[int, int] = SIZE,
    width: int = WIDTH,
) -> Iterator[Clip]:
    """A headless Chrome page on ``url``, recorded; between ``begin`` and ``finish`` is
    encoded as ``name``, ``width`` pixels wide. Without ``panel``, the control panel is
    collapsed by a click on its header, ``label``; the ``collapse`` folders are closed, and
    the ``plot`` channel is plotted."""
    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", args=["--enable-gpu", "--ignore-gpu-blocklist"])
        viewport = {"width": size[0], "height": size[1]}
        context = browser.new_context(viewport=viewport, record_video_dir=tmp, record_video_size=viewport)
        page = context.new_page()
        clip = Clip(page, time.monotonic())
        page.goto(url)
        if wait_for:
            page.locator(wait_for).first.wait_for(timeout=30_000)
        else:
            page.get_by_text("Connected" if not label else label, exact=True).wait_for(timeout=30_000)
        for close in page.locator(".mantine-Notification-closeButton").all():
            close.click()
        page.add_style_tag(content="*:focus, *:focus-visible { outline: none !important; }")
        for folder in collapse:
            page.get_by_text(folder, exact=True).first.click()
        if plot:
            page.locator('input[value="(none)"]').first.click()
            page.get_by_role("option", name=plot, exact=True).click()
        if not panel:
            page.get_by_text(label, exact=True).click()
        page.mouse.move(size[0] - 2, size[1] - 2)
        time.sleep(1.0)
        yield clip
        if clip.end is None:
            clip.finish()
        video = page.video
        context.close()
        browser.close()
        assert video is not None and clip.start is not None and clip.end is not None
        _encode(Path(video.path()), clip.start - clip.opened, clip.end - clip.start, out / name, width)


def _encode(video: Path, offset: float, duration: float, out: Path, width: int) -> None:
    scale = f"scale={width}:-2:flags=lanczos"
    common = ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{offset:.3f}", "-t", f"{duration:.3f}", "-i", str(video)]
    subprocess.run(
        [*common, "-vf", f"fps={FPS},{scale}", "-loop", "0", "-c:v", "libwebp_anim", "-quality", "70"]
        + ["-compression_level", "6", f"{out}.webp"],
        check=True,
    )
    still = ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{offset + 0.8 * duration:.3f}", "-i", str(video)]
    subprocess.run([*still, "-frames:v", "1", "-vf", scale, "-q:v", "4", f"{out}.jpg"], check=True)


def aim(server: Any, position: tuple[float, ...], look_at: tuple[float, ...]) -> None:
    """The camera a browser starts with, z up; set before it connects."""
    camera = server.initial_camera
    camera.up = (0.0, 0.0, 1.0)
    camera.position = position
    camera.look_at = look_at


def play(seconds: float, frame: Callable[[float], None], rate_hz: float = 30.0) -> None:
    """Call ``frame(t)`` at about ``rate_hz`` for ``seconds`` of wall time."""
    start = time.monotonic()
    while (t := time.monotonic() - start) < seconds:
        frame(t)
        time.sleep(1.0 / rate_hz)


# Raw MuJoCo: the caller owns the model, its state, and the loop.


def _keyframe_qpos(model: mujoco.MjModel, name: str) -> np.ndarray:
    return np.array(model.key(name).qpos)


def shadow_hand(menagerie: Path, references: Path, out: Path) -> None:
    """Shadow Hand through its own keyframes, rock, paper, scissors, and point."""
    model = mujoco.MjModel.from_xml_path(str(menagerie / "shadow_hand" / "scene_right.xml"))
    data = mujoco.MjData(model)
    # The keyframes ship beside the scene, not in it.
    keys = ElementTree.parse(menagerie / "shadow_hand" / "keyframes.xml").getroot().iter("key")
    qpos = {k.get("name"): np.array(k.get("qpos", "").split(), float) for k in keys}
    poses = [qpos[k] for k in ("open hand", "rock", "paper", "scissors", "point", "open hand")]
    with MujocoViewer(model, port=0, label="Shadow Hand", show_visibility=False) as viewer:
        aim(viewer.server, (0.42, -0.2, 0.24), (0.3, 0.0, 0.07))

        def frame(t: float) -> None:
            u = t / 1.2
            i = min(int(u), len(poses) - 2)
            s = 0.5 - 0.5 * math.cos(math.pi * min(u - i, 1.0))
            data.qpos[: poses[0].size] = (1 - s) * poses[i] + s * poses[i + 1]  # the hand; the cube stays
            viewer.update(data)

        frame(0.0)
        with recorded(viewer.url, out, "shadow_hand", label="Shadow Hand", panel=False) as clip:
            clip.begin()
            play(1.2 * (len(poses) - 1), frame)


def g1(menagerie: Path, references: Path, out: Path) -> None:
    """Unitree G1 waving from its standing keyframe."""
    model = mujoco.MjModel.from_xml_path(str(menagerie / "unitree_g1" / "scene_with_hands.xml"))
    data = mujoco.MjData(model)
    stand = _keyframe_qpos(model, "stand")
    waves = {
        "right_shoulder_roll_joint": (-1.35, 0.0),
        "right_shoulder_yaw_joint": (-1.5, 0.0),
        "right_elbow_joint": (0.0, 0.45),
        "left_shoulder_roll_joint": (0.25, 0.1),
    }
    address = {name: model.jnt_qposadr[model.joint(name).id] for name in waves}
    with MujocoViewer(model, port=0, label="Unitree G1", show_visibility=False) as viewer:
        aim(viewer.server, (1.25, -0.75, 1.15), (0.0, 0.0, 0.85))

        def frame(t: float) -> None:
            data.qpos[:] = stand
            rise = min(t / 0.8, 1.0)
            for name, (target, amplitude) in waves.items():  # from standing to the target, then waving
                q = target + amplitude * math.sin(2 * math.pi * t / 1.0)
                data.qpos[address[name]] = (1 - rise) * stand[address[name]] + rise * q
            viewer.update(data)

        frame(0.0)
        with recorded(viewer.url, out, "g1", label="Unitree G1", panel=False) as clip:
            clip.begin()
            play(4.0, frame)


# ssrobot runs: the runtime owns time and stepping; the view follows it.

HAND = "right_ur5e/gripper"
GEODUDE_FINGERS = [f"{HAND}/{side}_{part}" for side in ("left", "right") for part in ("pad", "follower", "spring_link")]


def _paced(ctx: Any, view: Any, until: Callable[[], bool], observe: Any = None, limit_s: float = 30.0) -> None:
    """Step the run, refreshing the view and observing every 25 ticks, until ``until()``."""
    start, tick = time.monotonic(), 0
    while not until() and time.monotonic() - start < limit_s:
        ctx.step()
        tick += 1
        if observe is not None and tick % 25 == 0:
            ctx.observe(observe)
        view.refresh()
        time.sleep(0.002)


def _all_but_depth(ctx: Any) -> Any:
    from ssrobot import ObservationRequest

    quantity = {c.name: str(c.quantity) for c in (*ctx.description.channels, *ctx.info.scene_channels)}
    return ObservationRequest(channels=tuple(c for c in ctx.info.channels if quantity.get(c) != "depth_image"))


def _scenario(references: Path, name: str) -> dict[str, Any]:
    robots = tomllib.loads((references / "robots.toml").read_text())["robots"]
    return dict(next(r for r in robots if r["name"] == name)["scenario"])


def _grasp_plan(ctx: Any, package: Any, references: Path) -> tuple[Any, Any, Any, Any]:
    """Geodude's right hand to the cylinder's side, 6 cm out: the snapshot, cylinder pose,
    plan, and planner."""
    from ssrobot import Pose, Tsr
    from ssrobot.planning import CBiRRT, SsikKinematics

    scenario = _scenario(references, "geodude")
    arm = ctx.manipulator(scenario["manipulator"])
    planner = CBiRRT(package.description, kinematics=SsikKinematics.from_package(package, arm.name))
    snapshot = ctx.snapshot()
    pose = next(o.pose for o in snapshot.objects if o.name == scenario["object"])
    region = Tsr(
        origin=pose,
        offset=Pose(position=(-0.06, 0.0, 0.0), quat_wxyz=tuple(scenario["offset_quat_wxyz"])),
        position_bounds=((0.0, 0.0), (0.0, 0.0), (-0.02, 0.02)),
        rotation_bounds=((0.0, 0.0), (0.0, 0.0), (-math.pi, math.pi)),
    )
    result = arm.plan_to_tsr([region], snapshot=snapshot, planner=planner, frame=scenario["frame"])
    if not result.succeeded:
        raise SystemExit(f"the grasp approach {result.status.value}: {result.diagnostics}")
    return snapshot, pose, result, planner


def geodude_contacts(menagerie: Path, references: Path, out: Path) -> None:
    """Geodude reaches for the cylinder and closes its hand; contacts appear as it holds."""
    from ssrobot import GripperCommand, RobotContext, load_installed_package
    from ssrobot.mujoco import MujocoRuntime

    from mj_viser import ContactOverlay, LiveView
    from mj_viser.integrations.ssrobot import TraceChannels, render_source

    package = load_installed_package("geodude_assets")
    description = package.description
    scenario = _scenario(references, "geodude")
    pairs = [("cylinder", finger) for finger in GEODUDE_FINGERS]
    channels = TraceChannels(description)
    runtime = MujocoRuntime(package, scene=references / "scenes" / "geodude-cylinder.xml", contacts=pairs)
    with RobotContext(description, runtime, sinks=[channels]) as ctx:
        _, pose, result, planner = _grasp_plan(ctx, package, references)
        gripper = description.end_effector(description.manipulator(scenario["manipulator"]).end_effector).gripper
        view = LiveView(render_source(ctx), channels=channels.board, port=0, label="Geodude")
        view.viewer.add_panel(ContactOverlay())
        url = view.open()
        x, y, z = pose.position
        aim(view.viewer.server, (x + 0.3, y + 0.36, z + 0.17), (x - 0.03, y - 0.02, z + 0.02))
        with recorded(url, out, "geodude_contacts", label="Geodude", panel=False) as clip:
            clip.begin()
            execution = ctx.execute(result, planner=planner)
            _paced(ctx, view, lambda: execution.done)
            closing = ctx.submit(GripperCommand(gripper=gripper, opening=0.0))
            _paced(ctx, view, lambda: closing.done)
            hold = time.monotonic() + 1.5
            _paced(ctx, view, lambda: time.monotonic() > hold)
        view.close()


def ada_live(menagerie: Path, references: Path, out: Path) -> None:
    """ADA moves beside the seated user; its channels, plot, and wrist camera follow live."""
    from ssrobot import JointTrajectory, RobotContext, load_installed_package
    from ssrobot.mujoco import MujocoRuntime

    from mj_viser import LiveView
    from mj_viser.integrations.ssrobot import TraceChannels, render_source

    package = load_installed_package("ada_assets")
    description = package.description
    group = next(g for g in description.groups if g.name == description.manipulator("arm").group)
    channels = TraceChannels(description)
    runtime = MujocoRuntime(package, scene=references / "scenes" / "ada-cylinder.xml")
    with RobotContext(description, runtime, sinks=[channels]) as ctx:
        request = _all_but_depth(ctx)
        snapshot = ctx.snapshot()
        by_name = dict(zip(snapshot.joints, snapshot.positions))
        start = tuple(float(by_name[n]) for n in group.joints)
        view = LiveView(render_source(ctx), channels=channels.board, port=0, label="ADA")
        url = view.open()
        aim(view.viewer.server, (0.6, -1.35, 1.4), (0.05, -0.1, 0.95))
        warm = time.monotonic() + 0.5  # readings first: only channels with readings can be plotted
        _paced(ctx, view, lambda: time.monotonic() > warm, observe=request)
        with recorded(
            url,
            out,
            "ada_live",
            label="ADA",
            collapse=("Visibility", "Run", "Status", "Channels"),
            plot="arm_q",
            size=(1280, 880),
            width=800,
        ) as clip:
            clip.begin()
            # The base and the wrist sweep; the shoulder and elbow stay clear of their limits.
            amplitude = (0.6, 0.0, 0.0, 0.5, 0.4, 0.6) + (0.0,) * (len(start) - 6)
            sway = [tuple(q + a * math.sin(k * math.pi / 2) for q, a in zip(start, amplitude)) for k in range(5)]
            execution = ctx.submit(
                JointTrajectory(
                    group=group.name,
                    joints=group.joints,
                    time_from_start_ns=tuple(k * 1_000_000_000 for k in range(5)),
                    positions=tuple(sway),
                )
            )
            _paced(ctx, view, lambda: execution.done, observe=request)
        view.close()


def geodude_preview(menagerie: Path, references: Path, out: Path) -> None:
    """A browser scrubs Geodude's planned reach as a ghost; the robot itself never moves."""
    from ssrobot import RobotContext, load_installed_package
    from ssrobot.mujoco import MujocoRuntime, materialize_world_from

    from mj_viser import LiveView
    from mj_viser.integrations.ssrobot import PlanPreview, render_source

    package = load_installed_package("geodude_assets")
    description = package.description
    scene = references / "scenes" / "geodude-cylinder.xml"
    with RobotContext(description, runtime := MujocoRuntime(package, scene=scene)) as ctx:
        snapshot, pose, result, _ = _grasp_plan(ctx, package, references)
        world = materialize_world_from(package, world=runtime.world(), scene=scene)
        view = LiveView(render_source(ctx), port=0, label="Geodude")
        preview = PlanPreview(world, description, live=view)
        view.viewer.add_panel(preview)
        url = view.open()
        preview.add_result("reach", result, snapshot)
        x, y, z = pose.position
        aim(view.viewer.server, (x + 0.85, y + 1.05, z + 0.6), (x - 0.12, y - 0.2, z + 0.25))
        with recorded(url, out, "geodude_preview", label="Geodude", collapse=("Visibility",)) as clip:
            page = clip.page
            for current, option in (("(none)", "reach"), ("path", "trajectory")):
                page.locator(f'input[value="{current}"]').first.click()
                page.get_by_role("option", name=option, exact=True).click()
                time.sleep(0.3)
            # Dragged across, as a user scrubs (arrow keys also move Viser's camera).
            thumb = page.get_by_role("slider").first.bounding_box()
            track = page.locator(".mantine-Slider-track").first.bounding_box()
            assert thumb is not None and track is not None
            y0 = thumb["y"] + thumb["height"] / 2
            x0, x1 = thumb["x"] + thumb["width"] / 2, track["x"] + track["width"]
            page.mouse.move(x0, y0)
            page.mouse.down()
            clip.begin()
            time.sleep(0.5)
            steps = 90
            for k in range(steps + 1):
                page.mouse.move(x0 + (x1 - x0) * k / steps, y0)
                view.refresh()
                time.sleep(0.03)
            page.mouse.up()
            time.sleep(1.0)
        view.close()


def _cylinder(scene: Path) -> tuple[float, float, float]:
    body = next(b for b in ElementTree.parse(scene).getroot().iter("body") if b.get("name") == "cylinder")
    x, y, z = (float(v) for v in body.get("pos", "").split())
    return x, y, z


def geodude_replay(menagerie: Path, references: Path, out: Path) -> None:
    """Geodude's grasp, saved as an ssrobot run bundle, converted, and played in Viser's
    own player."""
    import functools
    import http.server
    import threading

    from ssrobot import GripperCommand, JsonlTrace, RobotContext, load_installed_package, write_bundle
    from ssrobot.mujoco import MujocoRuntime

    package = load_installed_package("geodude_assets")
    description = package.description
    scene = references / "scenes" / "geodude-cylinder.xml"
    scenario = _scenario(references, "geodude")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        trace_path = work / "trace.jsonl"
        with (
            JsonlTrace(trace_path) as trace,
            RobotContext(description, runtime := MujocoRuntime(package, scene=scene), sinks=[trace]) as ctx,
        ):
            request = _all_but_depth(ctx)
            _, _, result, planner = _grasp_plan(ctx, package, references)
            gripper = description.end_effector(description.manipulator(scenario["manipulator"]).end_effector).gripper
            ctx.snapshot()
            ctx.observe(request)
            ticks = 0
            for submit in (
                lambda: ctx.execute(result, planner=planner),
                lambda: ctx.submit(GripperCommand(gripper=gripper, opening=0.0)),
                None,  # then holding, for a second
            ):
                handle = submit() if submit is not None else None
                hold = ticks + 500
                while (not handle.done) if handle is not None else ticks < hold:
                    ctx.step()
                    ticks += 1
                    if ticks % 10 == 0:
                        ctx.snapshot()
                    if ticks % 50 == 0:
                        ctx.observe(request)
            model_name = runtime.mapping.model_name
        bundle = work / "bundle"
        write_bundle(bundle, trace=trace_path, package=package, model=model_name, scene=scene)
        site = work / "site"
        subprocess.run(
            [sys.executable, "-m", "mj_viser.integrations.ssrobot.replay", str(bundle), str(site)], check=True
        )
        subprocess.run(["viser-build-client", "--out-dir", str(site / "client")], check=True, capture_output=True)
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            x, y, z = _cylinder(scene)
            camera = {
                "initialCameraPosition": f"{x + 0.95},{y + 1.2},{z + 0.65}",
                "initialCameraLookAt": f"{x - 0.2},{y - 0.2},{z + 0.25}",
                "initialCameraUp": "0,0,1",
            }
            query = "&".join(f"{k}={v}" for k, v in camera.items())
            url = f"{base}/client/?playbackPath={base}/recording.viser&{query}"
            with recorded(url, out, "geodude_replay", label="", wait_for="canvas") as clip:
                time.sleep(1.0)
                clip.begin()
                time.sleep(ticks * 0.002)  # one loop of the recording
        finally:
            server.shutdown()


CLIPS: dict[str, Callable[[Path, Path, Path], None]] = {
    "shadow_hand": shadow_hand,
    "g1": g1,
    "geodude_contacts": geodude_contacts,
    "ada_live": ada_live,
    "geodude_preview": geodude_preview,
    "geodude_replay": geodude_replay,
}


def main(menagerie: str, references: str, out_dir: str, clips: list[str]) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is needed to encode the clips")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in clips or list(CLIPS):
        CLIPS[name](Path(menagerie), Path(references), out)
        print(f"{name}: {out / name}.webp ({(out / f'{name}.webp').stat().st_size // 1024} KiB)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
