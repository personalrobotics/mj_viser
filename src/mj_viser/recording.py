# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Record a MuJoCo model's states over time as a native Viser recording.

A :class:`StateRecorder` plays caller-supplied states, text, and RGB images into a
``.viser`` file at the times the caller gives, for Viser's own player. Nothing is
interpolated: each state shows from its time until the next.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mujoco
import numpy as np

from mj_viser.viewer import MujocoViewer

TEXT = "/recording/text"
"""Scene path of the recorded text label."""
IMAGES = "/recording/images"
"""Scene path under which recorded images are placed, one per name."""

# Images stand upright beside the text: the image plane turned 90° about x.
_UPRIGHT = (np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0)


class StateRecorder:
    """Builds one Viser recording of a model.

    Call :meth:`advance` to move the recording's clock to a time, then :meth:`show`,
    :meth:`text`, or :meth:`image` to change what shows from then on, and
    :meth:`finish` for the recording. With ``hide_until_first_state``, nothing of the
    model shows before the first :meth:`show`: there is no state to show yet.
    """

    def __init__(self, model: mujoco.MjModel, *, label: str | None = None, hide_until_first_state: bool = True) -> None:
        self._model = model
        self._viewer = MujocoViewer(model, port=0, label=label, show_visibility=False)
        self._viewer.open()
        server = self._viewer.server
        assert self._viewer._scene_mgr is not None
        self._root = self._viewer._scene_mgr._body_frames[0]  # the world body: every model node is under it
        self._shown = not hide_until_first_state
        self._root.visible = self._shown
        self._extent = float(model.stat.extent)
        self._anchor = np.asarray(model.stat.center, float) + [0.0, 0.0, 0.75 * self._extent]
        self._text = server.scene.add_label(TEXT, "", position=self._anchor, anchor="bottom-left")
        self._images: dict[str, Any] = {}
        self._serializer = server.get_scene_serializer()
        self._origin: int | None = None
        self._cursor = 0
        self.states_ns: list[int] = []
        """The time of each state shown, from the first :meth:`advance`."""

    def advance(self, time_ns: int) -> None:
        """Move the clock to ``time_ns``. The first call sets the recording's start.

        Raises:
            ValueError: If ``time_ns`` is earlier than the clock.
        """
        if self._origin is None:
            self._origin = self._cursor = int(time_ns)
            return
        if time_ns < self._cursor:
            raise ValueError(f"time goes back: {time_ns} ns after {self._cursor} ns")
        if time_ns > self._cursor:
            self._serializer.insert_sleep((time_ns - self._cursor) / 1e9)
            self._cursor = int(time_ns)

    def show(self, data: mujoco.MjData) -> None:
        """Show ``data``'s state from now on."""
        self._viewer.update(data)
        if not self._shown:
            self._root.visible = self._shown = True
        self.states_ns.append(self._cursor)

    def text(self, lines: Sequence[str]) -> None:
        """Show these lines from now on."""
        text = "\n".join(lines)
        if text != self._text.text:
            self._text.text = text

    def image(self, name: str, image: np.ndarray | None) -> None:
        """Show an H×W×3 ``uint8`` image under ``name`` from now on, or hide it (None).

        Raises:
            ValueError: If ``image`` is not an H×W×3 ``uint8`` array.
        """
        node = self._images.get(name)
        if image is None:
            if node is not None:
                node.visible = False
            return
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{name}: not an H×W×3 uint8 image: {image.dtype} {image.shape}")
        if node is not None:
            node.image = image
            node.visible = True
            return
        width = 0.5 * self._extent
        height = width * image.shape[0] / image.shape[1]
        position = self._anchor + [width * (0.6 + 1.1 * len(self._images)), 0.0, -0.5 * height]
        self._images[name] = self._viewer.server.scene.add_image(
            f"{IMAGES}/{name}", image, width, height, format="jpeg", wxyz=_UPRIGHT, position=position
        )

    def finish(self) -> bytes:
        """The recording; the recorder is closed."""
        try:
            return self._serializer.serialize()
        finally:
            self._viewer.close()

    def close(self) -> None:
        """Release the recorder without a recording. Safe after :meth:`finish`."""
        self._viewer.close()

    def __enter__(self) -> StateRecorder:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
