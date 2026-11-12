"""Reinsurance programmes: several treaties applied in inuring order."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..core.base import ValidationError
from ..core.types import ArrayLike, FloatArray, IntArray
from ..core.validation import as_float_array, as_year_index
from .base import Treaty


@dataclass(frozen=True)
class ProgrammeResult:
    """Per-loss gross, ceded (by treaty) and net amounts from applying a programme."""

    gross: FloatArray
    ceded_by_treaty: dict[str, FloatArray]
    year_index: IntArray
    n_years: int

    @property
    def ceded(self) -> FloatArray:
        """Total ceded per loss across all treaties."""
        return sum(self.ceded_by_treaty.values(), np.zeros_like(self.gross))

    @property
    def net(self) -> FloatArray:
        return self.gross - self.ceded

    def annual(self, values: FloatArray) -> FloatArray:
        """Sum ``values`` (one per loss) by simulated year."""
        return np.bincount(self.year_index, weights=values, minlength=self.n_years)

    def annual_ceded(self, treaty: str | None = None) -> FloatArray:
        """Annual recoveries from one treaty, or from the whole programme."""
        return self.annual(self.ceded if treaty is None else self.ceded_by_treaty[treaty])

    def annual_net(self) -> FloatArray:
        return self.annual(self.net)


class Programme:
    """Treaties grouped into stages applied in order (inuring).

    Each stage is a treaty or a sequence of treaties. Every treaty in a stage
    applies to the same subject loss (for example the layers of an XoL tower);
    the loss retained after one stage is the subject loss of the next. So

    ``Programme(QuotaShare(0.3), [xol_5x5, xol_10x10])``

    applies the quota share first, with the XoL tower protecting the retained 70%.
    """

    def __init__(self, *stages: Treaty | Sequence[Treaty]) -> None:
        normalised: list[tuple[Treaty, ...]] = []
        for stage in stages:
            treaties = (stage,) if isinstance(stage, Treaty) else tuple(stage)
            if not treaties or not all(isinstance(t, Treaty) for t in treaties):
                raise ValidationError(
                    "each stage must be a Treaty or a non-empty sequence of Treaty"
                )
            normalised.append(treaties)
        if not normalised:
            raise ValidationError("a programme needs at least one treaty")
        names = [t.name for stage in normalised for t in stage]
        if len(set(names)) != len(names):
            raise ValidationError(f"treaty names must be unique within a programme, got {names}")
        self.stages = tuple(normalised)

    @property
    def treaties(self) -> list[Treaty]:
        return [t for stage in self.stages for t in stage]

    def apply(
        self, losses: ArrayLike, year_index: ArrayLike | None = None, n_years: int | None = None
    ) -> ProgrammeResult:
        """Apply every stage in order to ``losses``.

        Pass ``n_years`` when trailing simulated years may have no losses;
        otherwise it is inferred from the largest year in ``year_index``.
        """
        gross = as_float_array("losses", losses, non_negative=True)
        years = as_year_index(year_index, gross.size)
        if n_years is None:
            n_years = int(years.max()) + 1 if years.size else 1
        subject = gross
        ceded_by_treaty: dict[str, FloatArray] = {}
        for stage in self.stages:
            stage_ceded = np.zeros_like(subject)
            for treaty in stage:
                ceded = treaty.apply(subject, years)
                ceded_by_treaty[treaty.name] = ceded
                stage_ceded += ceded
            if np.any(stage_ceded > subject * (1 + 1e-12) + 1e-9):
                raise ValidationError(
                    "treaties in one stage cede more than the subject loss; check layers overlap"
                )
            subject = subject - stage_ceded
        return ProgrammeResult(gross, ceded_by_treaty, years, n_years)

    def apply_to(self, ylt) -> ProgrammeResult:
        """Apply to a :class:`~specinsure.simulation.YearLossTable`."""
        return self.apply(ylt.losses, ylt.year_index, ylt.n_years)

    def __repr__(self) -> str:
        parts = [" + ".join(t.name for t in stage) for stage in self.stages]
        return "Programme(" + " -> ".join(parts) + ")"
