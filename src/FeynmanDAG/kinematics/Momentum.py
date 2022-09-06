"""
Submodule to describe the four momentum of a particle.
"""
from __future__ import annotations

from typing import Any, Optional, Union

import attr
import numpy as np

from .helper import _alias_attribute
from .Lorentz import _LorentzVectorType
from .utils import _validate_mass


__all__ = ["FourMomentum"]


@attr.s(cmp=False)
class _FourMomentumType(_LorentzVectorType):
    """Numeric type to describe the four momentum of a particle."""

    mass: float | None = attr.ib(
        repr=False,
        validator=lambda instance, attribute, value: _validate_mass(value),
    )  # type: ignore
    E = _alias_attribute("x0")
    x = _alias_attribute("x1")
    y = _alias_attribute("x2")
    z = _alias_attribute("x3")

    @mass.default
    def __mass_default(self) -> Any:
        """Mass Default.

        The proposed mass is the max value of `np.sqrt(self@self)`. This proposed mass is retured if all elements in `np.sqrt(self@self)` is equal to the proposed mass. Otherwise `None` is returned.

        """
        proposed_masses = np.sqrt(self @ self)
        proposed_mass = proposed_masses.max()
        if np.isclose(
            proposed_masses, proposed_masses.max() * np.ones(proposed_masses.shape)
        ):
            return proposed_mass
        else:
            return None

    @property
    def isonshell(self) -> Any:
        r"""Indicates the onshellness of this instance.

        True, if the given :class:`_FourMomentumType` is onshell.

        :rtype: bool
        :raise ValueError: if there is no ``mass`` defined for this instance.


        Notes
        -----
        A four-momentum :math:`p^\mu` is called `onshell` w.r.t. a given mass :math:`m`, if and only if :math:`p^\mu p_\mu = m^2`.

        """
        if self.mass is not None:
            return np.allclose(
                np.asarray(self @ self), np.ones(self.shape) * self.mass**2
            )
        else:
            raise ValueError(
                "The onshell property of a FourMomentum can only be checked, if it has a mass attribute."
            )


def _from_LorentzVector(
    mom: _LorentzVectorType, mass: float | None
) -> _FourMomentumType:
    """
    Low-level constructor for :class:`_FourMomentumType` from a :class:`_LorentzVectorType`.
    """
    return _FourMomentumType(mom.x0, mom.x1, mom.x2, mom.x3, mass)



def _from_ndarray(arr: np.ndarray, mass: float | None) -> _FourMomentumType:
    """
    Low-level constructor for :class:`_FourMomentumType` from a :class:`np.ndarray`.
    """
    if arr.shape[0] != 4:
        raise ValueError(
            f"Array with shape {arr.shape} can not be interpreted as a FourMomentum. The first axis needs to have length four!"
        )
    return _FourMomentumType(arr[0], arr[1], arr[2], arr[3], mass)


def _from_tuple(tpl: tuple, mass: float | None) -> _FourMomentumType:
    """
    Low-level constructor for :class:`_FourMomentumType` from a given components.
    """
    return _FourMomentumType(tpl[0], tpl[1], tpl[2], tpl[3], mass)


AVIAL_CONSTRUCTORS = {
    "_LorentzVectorType": _from_LorentzVector,
    "_FourMomentumType": _from_LorentzVector,
    "ndarray": _from_ndarray,
}


def FourMomentum(*args: Any, **kwargs: Any) -> _FourMomentumType:
    """
    Top-level constructor for :class:`_FourMomentumType`.
    """
    mass = kwargs.pop("mass", None)
    if len(args) == 4:
        return _from_tuple(args, mass)
    elif (len(args) == 1) and (type(args[0]).__name__ in AVIAL_CONSTRUCTORS.keys()):
        return AVIAL_CONSTRUCTORS[type(args[0]).__name__](args[0], mass)
    else:
        raise ValueError(
            f"{args} cannot be interpreted as a FourVector. Avialable constructors are the components itself or {list(AVIAL_CONSTRUCTORS.keys())}."
        )
