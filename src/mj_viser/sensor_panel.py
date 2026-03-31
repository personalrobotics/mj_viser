"""Generic sensor plot panel for real-time MuJoCo sensor data."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import viser
from viser import uplot

from mj_viser.panels import PanelBase
from mj_viser.viewer import MujocoViewer


@dataclass
class SensorChannel:
    """A single data channel to plot.

    Args:
        index: Index into data.sensordata.
        label: Display name (e.g., "Force X").
        color: CSS color string (e.g., "red", "#ff0000").
    """
    index: int
    label: str
    color: str = "#4a9eff"


class SensorPanel(PanelBase):
    """Real-time scrolling plot of MuJoCo sensor data.

    Reads from ``data.sensordata`` at the given indices on each sync() call
    and plots the values as a time series.

    Example::

        panel = SensorPanel(
            title="Wrist F/T",
            channels=[
                SensorChannel(6, "Fx", "#e74c3c"),
                SensorChannel(7, "Fy", "#2ecc71"),
                SensorChannel(8, "Fz", "#3498db"),
            ],
            window_seconds=10.0,
        )
        viewer.add_panel(panel)
    """

    def __init__(
        self,
        title: str = "Sensors",
        channels: list[SensorChannel] | None = None,
        window_seconds: float = 10.0,
        max_points: int = 500,
        y_label: str = "Value",
    ) -> None:
        self._title = title
        self._channels = channels or []
        self._window_seconds = window_seconds
        self._max_points = max_points
        self._y_label = y_label

        # Ring buffers for time + each channel
        self._times: deque[float] = deque(maxlen=max_points)
        self._data: dict[int, deque[float]] = {
            ch.index: deque(maxlen=max_points) for ch in self._channels
        }

        self._plot: viser.GuiUplotHandle | None = None
        self._start_time: float | None = None

    def name(self) -> str:
        return self._title

    def setup(self, gui: viser.GuiApi, viewer: MujocoViewer) -> None:
        with gui.add_folder(self._title, order=5):
            # Initial empty plot — first series is always the time axis
            series = (
                uplot.Series(label="Time"),
                *(uplot.Series(label=ch.label, stroke=ch.color)
                  for ch in self._channels),
            )
            empty = np.zeros(0, dtype=np.float64)
            data = tuple([empty] * (1 + len(self._channels)))
            self._plot = gui.add_uplot(
                data=data,
                series=series,
                axes=(
                    uplot.Axis(label="Time (s)"),
                    uplot.Axis(label=self._y_label),
                ),
                aspect=2.5,
            )

    def on_sync(self, viewer: MujocoViewer) -> None:
        if self._plot is None or not self._channels:
            return

        # Record current time relative to start
        t = float(viewer.data.time)
        if self._start_time is None:
            self._start_time = t
        elapsed = t - self._start_time

        # Sample sensor data
        self._times.append(elapsed)
        for ch in self._channels:
            val = float(viewer.data.sensordata[ch.index])
            self._data[ch.index].append(val)

        # Build arrays for uplot: (time, ch0, ch1, ...)
        times_arr = np.array(self._times, dtype=np.float64)

        # Trim to window
        if len(times_arr) > 1:
            cutoff = elapsed - self._window_seconds
            mask = times_arr >= cutoff
            times_arr = times_arr[mask]
            channel_arrs = tuple(
                np.array(self._data[ch.index], dtype=np.float64)[mask]
                for ch in self._channels
            )
        else:
            channel_arrs = tuple(
                np.array(self._data[ch.index], dtype=np.float64)
                for ch in self._channels
            )

        self._plot.data = (times_arr, *channel_arrs)

    def reset(self) -> None:
        """Clear all recorded data."""
        self._times.clear()
        for d in self._data.values():
            d.clear()
        self._start_time = None
