"""Generic sensor plot panel for real-time MuJoCo sensor data."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

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
    """Real-time scrolling plot of MuJoCo sensor data using uPlot.

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
        y_label: str = "",
        aspect: float = 1.5,
        use_folder: bool = True,
    ) -> None:
        self._title = title
        self._channels = channels or []
        self._window_seconds = window_seconds
        self._max_points = max_points
        self._y_label = y_label
        self._aspect = aspect
        self._use_folder = use_folder

        # Ring buffers
        self._times: deque[float] = deque(maxlen=max_points)
        self._data: dict[int, deque[float]] = {
            ch.index: deque(maxlen=max_points) for ch in self._channels
        }

        self._plot: viser.GuiUplotHandle | None = None
        self._start_time: float | None = None

    def name(self) -> str:
        return self._title

    def setup(self, gui: viser.GuiApi, viewer: MujocoViewer) -> None:
        import contextlib
        ctx = gui.add_folder(self._title, order=5) if self._use_folder else contextlib.nullcontext()
        with ctx:
            if not self._use_folder:
                gui.add_markdown(f"**{self._title}**")
            # Time axis series + one per channel
            series = (
                uplot.Series(label="Time"),
                *(uplot.Series(
                    label=ch.label,
                    stroke=ch.color,
                    width=1,
                    points=uplot.Series_Points(show=False),
                ) for ch in self._channels),
            )

            empty = np.zeros(0, dtype=np.float64)
            data = tuple([empty] * (1 + len(self._channels)))

            self._plot = gui.add_uplot(
                data=data,
                series=series,
                axes=(
                    uplot.Axis(size=28),
                    uplot.Axis(size=30, label=self._y_label if self._y_label else None),
                ),
                legend=uplot.Legend(show=False),
                aspect=self._aspect,
            )

            # Compact inline legend
            legend_items = " ".join(
                f'<span style="margin-right:6px;">'
                f'<span style="display:inline-block;width:8px;height:8px;'
                f'background:{ch.color};border-radius:1px;margin-right:2px;'
                f'vertical-align:middle;"></span>'
                f'<span style="font-size:10px;color:#666;">{ch.label}</span>'
                f'</span>'
                for ch in self._channels
            )
            gui.add_html(f'<div style="padding:0 2px;">{legend_items}</div>')

    def on_sync(self, viewer: MujocoViewer) -> None:
        if self._plot is None or not self._channels:
            return

        t = float(viewer.data.time)
        if self._start_time is None:
            self._start_time = t
        elapsed = t - self._start_time

        # Record data
        self._times.append(elapsed)
        for ch in self._channels:
            self._data[ch.index].append(float(viewer.data.sensordata[ch.index]))

        # Build windowed arrays
        times_arr = np.array(self._times, dtype=np.float64)
        cutoff = elapsed - self._window_seconds
        mask = times_arr >= cutoff
        t_win = times_arr[mask]

        channel_arrs = tuple(
            np.array(self._data[ch.index], dtype=np.float64)[mask]
            for ch in self._channels
        )

        # uplot data update — just swap arrays, no DOM rebuild
        self._plot.data = (t_win, *channel_arrs)

    def reset(self) -> None:
        """Clear all recorded data."""
        self._times.clear()
        for d in self._data.values():
            d.clear()
        self._start_time = None
