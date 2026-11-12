"""Optional matplotlib plots. Install with ``pip install "specinsure[plot]"``."""

from __future__ import annotations

from collections.abc import Sequence

from ..risk.curves import ExceedanceCurve


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("plotting needs matplotlib: pip install 'specinsure[plot]'") from exc
    return plt


def plot_exceedance_curves(
    curves: Sequence[ExceedanceCurve],
    labels: Sequence[str] | None = None,
    *,
    ax=None,
    max_return_period: float | None = None,
):
    """Plot loss against return period (log scale) for one or more AEP/OEP curves."""
    plt = _pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    labels = labels or [c.kind for c in curves]
    for curve, label in zip(curves, labels, strict=False):
        ax.plot(curve.return_periods, curve.losses, label=label)
    ax.set_xscale("log")
    if max_return_period:
        ax.set_xlim(1, max_return_period)
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return ax
