"""
Generic implementation of a Lorentz vector

TODO:
- transform input always to an array_like. This provides consistence to shape
    and reshape.

"""
from __future__ import annotations

from typing import Any

import attr  # type: ignore
import numpy as np  # type: ignore

__all__ = [
    "LorentzVector",
]


@attr.s(cmp=False)
class _LorentzVectorType:
    
    x0: Any = attr.ib(default=None)
    x1: Any = attr.ib(default=None)
    x2: Any = attr.ib(default=None)
    x3: Any = attr.ib(default=None)

    def __eq__(self, other: Any) -> Any:
        if type(self) is not type(other):
            return False
        if (self.dtype is np.ndarray) or (other.dtype is np.ndarray):
            return (
                np.array_equal(self.x0, other.x0)
                & np.array_equal(self.x1, other.x1)
                & np.array_equal(self.x2, other.x2)
                & np.array_equal(self.x3, other.x3)
            )
        else:
            return (
                (self.x0 == other.x0)
                & (self.x1 == other.x1)
                & (self.x2 == other.x2)
                & (self.x3 == other.x3)
            )

    def __ne__(self, other: Any) -> Any:
        return not (self.__eq__(other))

    def __neg__(self) -> Any:
        return self.__class__(-self.x0, -self.x1, -self.x2, -self.x3)

    def __pos__(self) -> Any:
        return self.__class__(self.x0, self.x1, self.x2, self.x3)

    def __add__(self, other: Any) -> Any:
        if isinstance(other, _LorentzVectorType):
            return self.__class__(
                self.x0 + other.x0,
                self.x1 + other.x1,
                self.x2 + other.x2,
                self.x3 + other.x3,
            )
        else:
            raise TypeError(
                f"Operation {self.__class__} + {other.__class__} is not defined."
            )

    def __mul__(self, other: Any) -> Any:
        if not (isinstance(other, _LorentzVectorType)):
            # validation is passed to the components
            return self.__class__(
                self.x0 * other, self.x1 * other, self.x2 * other, self.x3 * other
            )
        else:
            raise TypeError(
                f"""The operation {self.__class__}*{other.__class__} is not defined.\n
                            Hint: for the Minkowski dot product, use {self.__class__.__name__}@{other.__class__.__name__} instead.
                            """
            )

    def __rmul__(self, other: Any) -> Any:
        if not (isinstance(other, _LorentzVectorType)):
            # validation is passed to the components
            return self.__class__(
                other * self.x0, other * self.x1, other * self.x2, other * self.x3
            )

    @property
    def shape(self) -> Any:
        """
        we need to implement a validator for the input components
        """
        if hasattr(self.x0, "shape"):
            return self.x0.shape
        return NotImplemented

    def reshape(self, *shape: int) -> _LorentzVectorType:
        """
        propagates the reshape function to the components
        """
        if hasattr(self.x0, "shape"):
            return self.__class__(
                self.x0.reshape(*shape),
                self.x1.reshape(*shape),
                self.x2.reshape(*shape),
                self.x3.reshape(*shape),
            )
        return NotImplemented

    @property
    def dtype(self) -> type:
        return type(self.x0)

    def __sub__(self, other: Any) -> Any:
        return self + (-other)

    def __rsub__(self, other: Any) -> Any:
        return other + (-self)

    def __matmul__(self, other: Any) -> Any:
        if isinstance(other, _LorentzVectorType):
            return self._dot(other)
        else:
            raise TypeError(
                f"The operation {self.__class__}*{other.__class__} is not defined."
            )

    def _dot(self, otherLV: _LorentzVectorType) -> Any:
        return (
            self.x0 * otherLV.x0
            - self.x1 * otherLV.x1
            - self.x2 * otherLV.x2
            - self.x3 * otherLV.x3
        )

    def __array__(self) -> np.ndarray:
        return np.asarray([self.x0, self.x1, self.x2, self.x3])

    __array_ufunc__ = None


def LorentzVector(*args: tuple) -> _LorentzVectorType:
    """Custom constructors for :class:`LorentzVectors`.

    .. todo:: build and doc single function for each constructor.

    """
    if len(args) == 4:
        return _LorentzVectorType(*args)

    elif isinstance(args[0], _LorentzVectorType):
        return _LorentzVectorType(args[0].x0, args[0].x1, args[0].x2, args[0].x3)

    elif isinstance(args[0], np.ndarray) and args[0].shape[0] == 4:
        return _LorentzVectorType(args[0][0], args[0][1], args[0][2], args[0][3])
    else:
        raise ValueError(
            f"""{args} can not be interpreted for initializing a LorentzVector!\n
                        Hint: try to initialize by\n
                        \t four components: `LorentzVector(x0,x1,x2,x3)`\n
                        \t a suitable array: `LorentzVector(arr)`\n
                        \t another LorentzVector: `LorentzVector(LV)`."""
        )
