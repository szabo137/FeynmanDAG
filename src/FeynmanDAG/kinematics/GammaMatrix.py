"""
this module contains Dirac's gamma matrices, provided as a LorentzVector of DiracMatrices. For now, we choose the Dirac representation.

TODO:
- check signs (convention for co- and contra-variant indices)
"""

from __future__ import annotations

import numbers
from typing import Any

import attr
import numpy as np

from .DiracAlgebra import DiracMatrix
from .Lorentz import _LorentzVectorType

__all__ = ["Gmu", "G0", "G1", "G2", "G3", "G5", "feynman_slash"]

g0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
G0 = DiracMatrix(g0)
g1 = np.array([[0, 0, 0, -1], [0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
G1 = DiracMatrix(g1)
g2 = np.array([[0, 0, 0, 1j], [0, 0, -1j, 0], [0, -1j, 0, 0], [1j, 0, 0, 0]])
G2 = DiracMatrix(g2)

g3 = np.array([[0, 0, -1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, -1, 0, 0]])
G3 = DiracMatrix(g3)

g5 = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
G5 = DiracMatrix(-g5)


@attr.s(frozen=True)
class GammaMatrix(_LorentzVectorType):
    x0: DiracMatrix = attr.ib(default=G0)
    x1: DiracMatrix = attr.ib(default=G1)
    x2: DiracMatrix = attr.ib(default=G2)
    x3: DiracMatrix = attr.ib(default=G3)


Gmu = GammaMatrix()


def feynman_slash(LV: _LorentzVectorType) -> Any:
    if issubclass(LV.dtype, numbers.Number):
        return Gmu @ LV
    else:
        ndim = len(LV.shape)
        new_shape = (1,) * ndim
        return Gmu.reshape(*new_shape) @ LV
