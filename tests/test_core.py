import numpy as np
import pytest

from specinsure.core import ValidationError, make_rng, spawn_rngs
from specinsure.core.validation import as_float_array, as_year_index, check_fraction, check_level


def test_validation_error_is_value_error():
    assert issubclass(ValidationError, ValueError)


@pytest.mark.parametrize("level", [0, 1, -0.1, 1.5])
def test_check_level_rejects_out_of_range(level):
    with pytest.raises(ValidationError):
        check_level(level)


def test_check_fraction_bounds():
    assert check_fraction("q", 1.0) == 1.0
    with pytest.raises(ValidationError):
        check_fraction("q", 0.0, allow_zero=False)


def test_as_float_array_rejects_nan_and_negative():
    with pytest.raises(ValidationError):
        as_float_array("x", [1.0, np.nan])
    with pytest.raises(ValidationError):
        as_float_array("x", [-1.0], non_negative=True)


def test_year_index_defaults_to_single_year_and_checks_shape():
    np.testing.assert_array_equal(as_year_index(None, 3), [0, 0, 0])
    with pytest.raises(ValidationError):
        as_year_index([0, 1], 3)
    with pytest.raises(ValidationError):
        as_year_index([0.5, 1.0, 2.0], 3)


def test_spawned_streams_are_reproducible_and_independent():
    a = [g.random() for g in spawn_rngs(7, 3)]
    b = [g.random() for g in spawn_rngs(7, 3)]
    assert a == b
    assert len(set(a)) == 3


def test_make_rng_passes_generators_through():
    g = np.random.default_rng(1)
    assert make_rng(g) is g
