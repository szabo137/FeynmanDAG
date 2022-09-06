"""
Particle Spinors.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import attr
import numpy as np

from .DiracAlgebra import BiSpinor, DiracMatrix, UnitDiracMatrix
from .GammaMatrix import G0, feynman_slash
from .Momentum import _FourMomentumType
from .utils import _get_spin_combinations

__all__ = [
    "FermionSpinor",
    "FermionBaseSpinorList",
    "IncomingFermion",
    "OutgoingFermion",
    "IncomingAntiFermion",
    "OutgoingAntiFermion",
]


@attr.s(frozen=True, cmp=False, slots=True)
class FermionBaseSpinorList:
    anti_particle: bool = attr.ib(default=False)
    __base_spinors: tuple[BiSpinor, BiSpinor] = attr.ib(init=False, repr=False)

    @__base_spinors.default
    def __base_spinors_constructor(self) -> tuple[BiSpinor, BiSpinor]:
        if self.anti_particle:
            return (BiSpinor([0, 0, 1, 0]), BiSpinor([0, 0, 0, 1]))
        else:
            return (BiSpinor([1, 0, 0, 0]), BiSpinor([0, 1, 0, 0]))

    def __getitem__(self, key: int) -> BiSpinor | None:
        try:
            return self.__base_spinors[key]
        except IndexError:
            raise IndexError(
                "Index out of bounds. The possible indices are 1 (spin up) or 2 (spin down). <{key}> given."
            )


def _get_fermion_boost_matrix(
    mom: _FourMomentumType, anti_particle: bool = False
) -> Any:
    return (
        (-1) ** anti_particle * feynman_slash(mom)
        + mom.mass * UnitDiracMatrix(len(mom.shape))
    ) / np.sqrt(np.abs(mom.E) + mom.mass)


@attr.s(frozen=True, cmp=False, slots=True)
class FermionSpinor:
    mom: _FourMomentumType = attr.ib()
    is_incoming: bool = attr.ib(default=True, kw_only=True)
    is_outgoing: bool = attr.ib(init=False)
    is_anti_particle: bool = attr.ib(default=False, kw_only=True)
    base_spinor: FermionBaseSpinorList = attr.ib(init=False, repr=False)
    boost_matrix: DiracMatrix = attr.ib(init=False, repr=False)

    @mom.validator
    def __mom_onshell_validator(self, attribute: Any, value: Any) -> None:
        assert (
            self.mom.isonshell
        ), "To construct a fermion spinor, the momentum needs to be on-shell."

    @is_incoming.validator
    def __is_incoming_validator(self, attribute: Any, value: Any) -> None:
        assert isinstance(
            value, bool
        ), f"The keyword argument `is_incoming` needs to be a boolian. {type(value)} given."

    @is_anti_particle.validator
    def __is_anti_particle_validator(self, attribute: Any, value: Any) -> None:
        assert isinstance(
            value, bool
        ), f"The keyword argument `is_anti_particle` needs to be a boolian. {type(value)} given."

    @is_outgoing.default
    def __outgoing_default(self) -> bool:
        return not self.is_incoming

    def __attrs_post_init__(self) -> None:
        # we need to build these here, since it used validated mom.
        object.__setattr__(
            self, "base_spinor", FermionBaseSpinorList(self.is_anti_particle)
        )
        object.__setattr__(
            self,
            "boost_matrix",
            _get_fermion_boost_matrix(self.mom, self.is_anti_particle),
        )

    def __call__(self, spin: int) -> Any:
        if self.is_incoming ^ self.is_anti_particle:
            return self.boost_matrix * self.base_spinor[spin]
        else:
            return (self.boost_matrix * self.base_spinor[spin]).adjoint() * G0


def IncomingFermion(mom: _FourMomentumType) -> FermionSpinor:
    return FermionSpinor(mom, is_incoming=True, is_anti_particle=False)


def OutgoingFermion(mom: _FourMomentumType) -> FermionSpinor:
    return FermionSpinor(mom, is_incoming=False, is_anti_particle=False)


def IncomingAntiFermion(mom: _FourMomentumType) -> FermionSpinor:
    return FermionSpinor(mom, is_incoming=True, is_anti_particle=True)


def OutgoingAntiFermion(mom: _FourMomentumType) -> FermionSpinor:
    return FermionSpinor(mom, is_incoming=False, is_anti_particle=True)


def fermion_sandwich(
    aSP: FermionSpinor, dirac_operator: DiracMatrix, SP: FermionSpinor
) -> Callable:
    """
    Return a function which maps the spin combinations ``(i,j)`` onto the product ``aSP(i)*(dirac_operator*SP(j))``.
    """
    return lambda i, j: aSP(i) * (dirac_operator * SP(j))


def fermion_sandwich_spinsummed(
    aSP: FermionSpinor, dirac_operator: DiracMatrix, SP: FermionSpinor
) -> Any:
    """
    Return the result of ``aSP(i)*(dirac_operator*SP(j))`` summed over all spin combinations.
    """
    return np.sum(
        [
            fermion_sandwich(aSP, dirac_operator, SP)(i, j)
            for (i, j) in _get_spin_combinations(2)
        ],
        axis=0,
    )


def fermion_sandwich_spinorthosummed(
    aSP: FermionSpinor, dirac_operator: DiracMatrix, SP: FermionSpinor
) -> Any:
    """
    Return the result of ``aSP(i)*(dirac_operator*SP(i))`` summed over all equal spins.
    """
    return np.sum(
        [
            fermion_sandwich(aSP, dirac_operator, SP)(i, j)
            for (i, j) in _get_spin_combinations(2)
            if i == j
        ],
        axis=0,
    )
