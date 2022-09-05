"""
Submodule to describe the polarisation of fields.
"""
from __future__ import annotations

from typing import Any

import attr
import numpy as np

from .Lorentz import _LorentzVectorType

__all__ = ["PolarisationVector"]


@attr.s(eq=False, order=False)
class _PolarisationVectorType(_LorentzVectorType):
    """Numeric type to describe the four polarisation of a field."""

    @property
    def isnormed(self) -> Any:
        r"""Indicates the normalisation of this instance.

        True, if the given :class:`_PolarisationVectorType` is normed to -1.

        :rtype: bool

        """
        return np.allclose(np.asarray(self @ self), -1)

    def conj(self) -> None:
        """
        Inplace-conjugate the component of this instance.
        """
        self.x0 = np.conjugate(self.x0)
        self.x1 = np.conjugate(self.x1)
        self.x2 = np.conjugate(self.x2)
        self.x3 = np.conjugate(self.x3)

    def conjugate(self) -> _PolarisationVectorType:
        """
        Return a new instance with conjugated components.
        """
        return self.__class__(
            np.conjugate(self.x0),
            np.conjugate(self.x1),
            np.conjugate(self.x2),
            np.conjugate(self.x3),
        )


def PolarisationVector(*args: Any) -> _PolarisationVectorType:
    """
    Top-level constructor for :class:`_PolarisationVectorType`.
    """
    if len(args) == 4:
        return _PolarisationVectorType(*args)

    elif isinstance(args[0], _PolarisationVectorType):
        return _PolarisationVectorType(args[0].x0, args[0].x1, args[0].x2, args[0].x3)

    elif isinstance(args[0], np.ndarray) and args[0].shape[0] == 4:
        return _PolarisationVectorType(args[0][0], args[0][1], args[0][2], args[0][3])
    else:
        raise ValueError(
            f"""{args} can not be interpreted for initializing a PolarisationVector!\n
                    Hint: try to initialize by\n
                    \t four components: `PolarisationVector(x0,x1,x2,x3)`\n
                    \t a suitable array: `PolarisationVector(arr)`\n
                    \t another PolarisationVector: `PolarisationVector(LV)`."""
        )
