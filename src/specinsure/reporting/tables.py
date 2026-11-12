"""Summary tables as plain Python structures, with a dependency-free text formatter.

Rows are lists of dicts so they drop straight into ``pandas.DataFrame(rows)``
when pandas is available, without making it a dependency.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from ..core.types import ArrayLike
from ..reinsurance.programme import Programme, ProgrammeResult
from ..reinsurance.treaties import ExcessOfLoss
from ..risk.measures import aal, standard_deviation, standard_error, tvar, var
from ..risk.return_period import return_period_loss


def risk_summary(
    annual_losses: ArrayLike,
    *,
    levels: Iterable[float] = (0.99, 0.995),
    return_periods: Iterable[float] = (10, 50, 100, 200, 250),
) -> dict[str, float]:
    """AAL with its Monte Carlo standard error, VaR/TVaR at each level and return period losses."""
    x = np.asarray(annual_losses, dtype=float)
    out = {
        "n_years": float(x.size),
        "aal": aal(x),
        "aal_standard_error": standard_error(x),
        "standard_deviation": standard_deviation(x),
    }
    for lvl in levels:
        out[f"var_{lvl:g}"] = var(x, lvl)
        out[f"tvar_{lvl:g}"] = tvar(x, lvl)
    for rp in return_periods:
        out[f"1_in_{rp:g}"] = return_period_loss(x, rp)
    return out


def layer_summary(programme: Programme, result: ProgrammeResult) -> list[dict[str, Any]]:
    """One row per treaty: expected recovery, volatility, recovery and exhaustion probabilities."""
    rows = []
    for treaty in programme.treaties:
        annual = result.annual_ceded(treaty.name)
        row: dict[str, Any] = {
            "treaty": treaty.name,
            "expected_recovery": float(annual.mean()),
            "standard_deviation": standard_deviation(annual),
            "probability_of_recovery": float(np.mean(annual > 0)),
        }
        if isinstance(treaty, ExcessOfLoss):
            row["loss_on_line"] = row["expected_recovery"] / treaty.limit
            cap = treaty.aggregate_limit
            row["probability_of_exhaustion"] = (
                float(np.mean(annual >= cap * (1 - 1e-12))) if np.isfinite(cap) else 0.0
            )
        rows.append(row)
    return rows


def format_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str] | None = None,
    float_format: str = ",.2f",
) -> str:
    """Render rows as an aligned plain-text table."""
    if not rows:
        return ""
    columns = list(columns or rows[0].keys())

    def fmt(value: Any) -> str:
        if isinstance(value, float | np.floating):
            return format(float(value), float_format)
        return str(value)

    cells = [[fmt(r.get(c, "")) for c in columns] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(columns)]
    header = "  ".join(c.ljust(w) for c, w in zip(columns, widths, strict=False))
    rule = "  ".join("-" * w for w in widths)
    body = ["  ".join(v.rjust(w) for v, w in zip(row, widths, strict=False)) for row in cells]
    return "\n".join([header, rule, *body])
