"""
this function contains some physics related utility functions.

For python related helper functions, see ``qftlib.helper``.
"""
from __future__ import annotations

import itertools
from typing import Any, Iterator, Tuple


def _validate_mass(value: Any) -> None:
    """
    validator for a positive mass
    """
    if (value is not None) and value < 0:
        raise ValueError(
            f"The mass of a particle needs to be positiv! ({value} given.)"
        )


def _get_spin_combinations(no_of_particles: int) -> Iterator[tuple[int, ...]]:
    return itertools.product(*([0, 1],) * no_of_particles)
