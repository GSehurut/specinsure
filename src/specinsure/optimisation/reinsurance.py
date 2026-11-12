"""Compare reinsurance structures on cost against capital released."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Literal

import numpy as np

from ..core.base import ValidationError
from ..core.types import ArrayLike
from ..core.validation import check_level, check_non_negative
from ..reinsurance.programme import Programme, ProgrammeResult
from ..reinsurance.treaties import ExcessOfLoss
from ..risk.measures import tvar, var
from .results import ProgrammeEvaluation

PremiumSource = Mapping[str, float] | Callable[[ProgrammeResult], float]


def evaluate_programmes(
    gross,
    candidates: Mapping[str, Programme],
    premiums: PremiumSource,
    *,
    level: float = 0.995,
    measure: Literal["var", "tvar"] = "var",
    cost_of_capital: float = 0.10,
    other_losses: ArrayLike | None = None,
) -> list[ProgrammeEvaluation]:
    """Evaluate candidate programmes on the same simulated gross losses (common random numbers).

    ``gross`` is the :class:`~specinsure.simulation.YearLossTable` being
    protected. ``premiums`` maps each candidate to its premium, or is a
    function of the programme result (for example a technical price).
    ``other_losses`` are annual losses from the rest of the portfolio, in the
    same simulated years; when given, capital is measured on the whole
    portfolio, so the result reflects the marginal (diversified) capital
    effect of the reinsurance rather than its stand-alone effect.

    Returns one evaluation per candidate plus ``"no reinsurance"``, sorted
    by ``value`` (best first).
    """
    check_level(level)
    check_non_negative("cost_of_capital", cost_of_capital)
    if measure not in ("var", "tvar"):
        raise ValidationError("measure must be 'var' or 'tvar'")
    if "no reinsurance" in candidates:
        raise ValidationError("'no reinsurance' is reserved for the baseline")
    risk = var if measure == "var" else tvar
    background = (
        np.zeros(gross.n_years) if other_losses is None else np.asarray(other_losses, float)
    )
    if background.shape != (gross.n_years,):
        raise ValidationError("other_losses must have one value per simulated year")

    def capital(annual_losses: np.ndarray) -> float:
        total = annual_losses + background
        return risk(total, level) - float(total.mean())

    gross_capital = capital(gross.annual_totals())
    evaluations = [
        ProgrammeEvaluation("no reinsurance", 0.0, 0.0, 0.0, gross_capital, 0.0, 0.0, 0.0)
    ]
    for name, programme in candidates.items():
        result = programme.apply_to(gross)
        premium = premiums(result) if callable(premiums) else float(premiums[name])
        check_non_negative(f"premium for {name!r}", premium)
        recovery = float(result.annual_ceded().mean())
        net_capital = capital(result.annual_net())
        released = gross_capital - net_capital
        saved = cost_of_capital * released
        margin = premium - recovery
        evaluations.append(
            ProgrammeEvaluation(
                name, premium, recovery, margin, net_capital, released, saved, saved - margin
            )
        )
    return sorted(evaluations, key=lambda e: e.value, reverse=True)


def xol_grid(
    attachments: Iterable[float], limits: Iterable[float], **treaty_kwargs
) -> dict[str, Programme]:
    """Single-layer XoL candidates for every (attachment, limit) pair on a market lattice.

    Extra keyword arguments (``reinstatements``, ``aad``, ...) go to every
    :class:`~specinsure.reinsurance.ExcessOfLoss`.
    """
    grid: dict[str, Programme] = {}
    for a in attachments:
        for lim in limits:
            layer = ExcessOfLoss(limit=lim, attachment=a, **treaty_kwargs)
            grid[f"{lim:,.0f} xs {a:,.0f}"] = Programme(layer)
    if not grid:
        raise ValidationError("attachments and limits must not be empty")
    return grid
