# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""mj_viser for ssrobot runs: the adapters between ssrobot's types and mj_viser's.

Importing this module imports ssrobot, which must be installed with its MuJoCo extra.
Nothing else in mj_viser does.

- :func:`render_source` adapts ``ssrobot.mujoco.render_handle(ctx)`` to
  :class:`mj_viser.RenderSource` for :class:`mj_viser.LiveView`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    # ssrobot is optional and not installed for type checking.
    import ssrobot  # type: ignore[import-not-found,unused-ignore]
    from ssrobot.mujoco import render_handle  # type: ignore[import-not-found,unused-ignore]
except ModuleNotFoundError as e:  # pragma: no cover - exercised by the standalone lane
    raise ModuleNotFoundError(
        "mj_viser.integrations.ssrobot needs ssrobot, installed with its MuJoCo extra", name=e.name
    ) from e

from mj_viser.live import SourceStale

__all__ = ["render_source"]

# ssrobot's codes for a render handle that can no longer describe its runtime: the
# runtime reopened (make a new handle) or is not open.
_STALE_CODES = frozenset({"stale_render", "not_open"})


@dataclass(frozen=True)
class _Frame:
    time_ns: int
    revision: int
    world: str


class _HandleSource:
    """An ssrobot render handle as an mj_viser render source."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self.model = handle.model
        self.data = handle.data
        self.world: str = handle.world

    def capture(self) -> _Frame:
        try:
            frame = self._handle.capture()
        except ssrobot.SsrobotError as e:
            if e.code in _STALE_CODES:
                raise SourceStale(e.code) from e
            raise
        return _Frame(time_ns=frame.time.time_ns, revision=frame.revision, world=frame.world)


def render_source(ctx: ssrobot.RobotContext) -> _HandleSource:
    """A render source over the context's MuJoCo runtime, for :class:`mj_viser.LiveView`.

    Each source owns its own copies of the model and state. It goes stale
    (``"stale_render"``) when the runtime reopens, or ``"not_open"`` when it is closed.
    """
    return _HandleSource(render_handle(ctx))
