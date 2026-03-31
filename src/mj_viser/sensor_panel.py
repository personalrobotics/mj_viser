"""Generic sensor plot panel for real-time MuJoCo sensor data."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
import viser

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
    """Real-time scrolling plot of MuJoCo sensor data using Plotly.

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
        height: int = 150,
        y_label: str = "",
    ) -> None:
        self._title = title
        self._channels = channels or []
        self._window_seconds = window_seconds
        self._max_points = max_points
        self._height = height
        self._y_label = y_label

        # Ring buffers
        self._times: deque[float] = deque(maxlen=max_points)
        self._data: dict[int, deque[float]] = {
            ch.index: deque(maxlen=max_points) for ch in self._channels
        }

        self._plot: viser.GuiPlotlyHandle | None = None
        self._start_time: float | None = None
        self._update_counter = 0

    def name(self) -> str:
        return self._title

    def setup(self, gui: viser.GuiApi, viewer: MujocoViewer) -> None:
        with gui.add_folder(self._title, order=5):
            fig = self._make_figure()
            self._plot = gui.add_plotly(fig, aspect=1.0)

            # Compact inline legend below the plot
            legend_items = " ".join(
                f'<span style="margin-right:8px;">'
                f'<span style="display:inline-block;width:10px;height:10px;'
                f'background:{ch.color};border-radius:2px;margin-right:3px;'
                f'vertical-align:middle;"></span>'
                f'<span style="font-size:11px;color:#555;">{ch.label}</span>'
                f'</span>'
                for ch in self._channels
            )
            gui.add_html(
                f'<div style="padding:2px 4px;">{legend_items}</div>'
            )

    def on_sync(self, viewer: MujocoViewer) -> None:
        if self._plot is None or not self._channels:
            return

        # Only update every 5th sync to reduce WebSocket traffic
        self._update_counter += 1
        if self._update_counter % 5 != 0:
            # Still record data, just don't send the plot update
            t = float(viewer.data.time)
            if self._start_time is None:
                self._start_time = t
            self._times.append(t - self._start_time)
            for ch in self._channels:
                self._data[ch.index].append(float(viewer.data.sensordata[ch.index]))
            return

        t = float(viewer.data.time)
        if self._start_time is None:
            self._start_time = t
        elapsed = t - self._start_time

        self._times.append(elapsed)
        for ch in self._channels:
            self._data[ch.index].append(float(viewer.data.sensordata[ch.index]))

        self._plot.figure = self._make_figure(populated=True, now=elapsed)

    def _make_figure(self, populated: bool = False, now: float = 0.0) -> go.Figure:
        fig = go.Figure()

        if populated:
            times = np.array(self._times)
            cutoff = now - self._window_seconds
            mask = times >= cutoff
            t = times[mask]

            for ch in self._channels:
                vals = np.array(self._data[ch.index])[mask]
                fig.add_trace(go.Scatter(
                    x=t, y=vals,
                    name=ch.label,
                    line=dict(color=ch.color, width=1.5),
                    hoverinfo="skip",
                ))

        fig.update_layout(
            margin=dict(l=35, r=5, t=5, b=25),
            height=self._height,
            showlegend=False,
            plot_bgcolor="white",
            xaxis=dict(
                showgrid=True, gridcolor="#eee", zeroline=False,
                tickfont=dict(size=9),
            ),
            yaxis=dict(
                title=dict(text=self._y_label, font=dict(size=10)) if self._y_label else None,
                showgrid=True, gridcolor="#eee", zeroline=True, zerolinecolor="#ccc",
                tickfont=dict(size=9),
            ),
        )
        return fig

    def reset(self) -> None:
        """Clear all recorded data."""
        self._times.clear()
        for d in self._data.values():
            d.clear()
        self._start_time = None
