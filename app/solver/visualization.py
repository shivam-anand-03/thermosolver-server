"""Static Matplotlib figures (used in the PDF report)."""

from __future__ import annotations

import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE = "#2563eb"
SLATE = "#334155"
GRID = "#e2e8f0"


def _style(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, color=SLATE, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    return buf.getvalue()


def temperature_profile_png(result: dict[str, Any]) -> bytes:
    """Temperature vs position (final state; several times for transient)."""
    unit = result["summary"]["unit_symbol"]
    x = result["x"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    if result["problem_type"] == "transient":
        times, snaps = result["times"], result["snapshots"]
        picks = sorted({0, len(times) // 4, len(times) // 2, 3 * len(times) // 4, len(times) - 1})
        cmap = plt.get_cmap("viridis")
        for n, i in enumerate(picks):
            ax.plot(x, snaps[i], marker="o", ms=3, lw=1.6, color=cmap(n / max(1, len(picks) - 1)),
                    label=f"t = {times[i]:.4g} s")
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.plot(x, result["solution"], marker="o", ms=5, lw=2, color=BLUE)
    _style(ax, "Position x [m]", f"Temperature [{unit}]", "Temperature vs position")
    return _png(fig)


def temperature_history_png(result: dict[str, Any]) -> bytes:
    """Temperature vs time at a few representative nodes (transient only)."""
    unit = result["summary"]["unit_symbol"]
    times, snaps = result["times"], result["snapshots"]
    n = len(result["x"])
    nodes = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    cmap = plt.get_cmap("tab10")
    for c, j in enumerate(nodes):
        ax.plot(times, [s[j] for s in snaps], lw=1.6, color=cmap(c), label=f"Node {j + 1}")
    ax.legend(fontsize=8, frameon=False)
    _style(ax, "Time t [s]", f"Temperature [{unit}]", "Temperature vs time")
    return _png(fig)
