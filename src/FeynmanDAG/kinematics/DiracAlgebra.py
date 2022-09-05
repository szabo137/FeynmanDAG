"""
This module contains the three datatypes :class:`qftlib.DiracMatrix`, :class:`qftlib.BiSpinor` and
:class:`qftlib.AdjointBiSpinor`,which form the Dirac algebra. All data types are subclassed from the abstrac base class :class:`_DiracABC` and represent containers
for :class:`numpy.ndarray` of arbitrary shape. Internally, these are extensions of :class:`numpy.ndarray`, where the respective tensor shape is enforced and multiplications are restricted to specific combination of these data types.


.. todo::
     - implement __get_index__ and __set_index__ for all data types
     - input types of the arithmetics shall not be 'object' but a union of
       supported types


"""
from __future__ import annotations

import numbers
from typing import Any

import attr
import numpy as np

from .Lorentz import _LorentzVectorType

__all__ = ["DiracMatrix", "BiSpinor", "AdjointBiSpinor", "UnitDiracMatrix"]


def _check_input_data(self: Any, attribute: Any, value: Any) -> Any:
    tensor_shape = self.TSHAPE
    tensor_dim = len(tensor_shape)
    if not np.array_equal(value.shape[:tensor_dim], tensor_shape):
        raise ValueError(
            f"""Input data has wrong shape:
                 The leading axes need to have shape {tensor_shape}.
                 {value.shape} given."""
        )


@attr.s(slots=True, cmp=False)
class _DiracABC:
    """This is the abstract base class for objects referred to the Dirac Algebra.

    |:warning:| This class is only for subclassing. Do not initialize this class! |:warning:|
    """

    TSHAPE: tuple[int, ...] = tuple()
    ALLOWED_MATMUL_TYPES: tuple[str, ...] = attr.ib(
        init=False, repr=False, default=tuple()
    )
    data = attr.ib(on_setattr=_check_input_data, converter=np.asarray)  # type: ignore

    @data.validator
    def check_data(self, attribute: Any, value: Any) -> Any:
        _check_input_data(self, attribute, value)

    __array_ufunc__ = None

    def __array__(self) -> np.ndarray:
        """Returns the array version of this instance. This is returned if :func:`numpy.asarray` is called on this instance.

        :rtype: numpy.ndarray
        """
        return self.data

    @property
    def shape(self) -> Any:
        """Returns the shape of each component of this instance.

        :rtype: tuple
        """
        tensor_dim = len(self.TSHAPE)
        return self.data.shape[tensor_dim:]

    @property
    def tshape(self) -> tuple[Any, ...]:
        """Returns the tensorshape of this instance.

        :rtype: tuple
        """
        return self.TSHAPE

    def reshape(self, *shape: int) -> _DiracABC:
        """Propagates the reshape function to all components of this instance.

        :param shape: new shape for each component
        :type shape: int or tuple
        :rtype: _DiracABC

        Notes
        -----
        This does not change the tensor shape of this instance.
        """
        new_shape = self.tshape + shape
        return self.__class__(self.data.reshape(*new_shape))

    def __eq__(self, other: object) -> Any:  # check why bool raises an error
        """
        Returns ``True`` if ``other`` has the same type and the same ``data`` as this instance.

        :param other: Object to compare with.
        :rtype: bool
        """
        if not isinstance(other, self.__class__):
            return False
        return np.array_equal(self.data, other.data)

    def __ne__(self, other: Any) -> Any:
        """Returns ``True`` if ``other`` is not equal to this instance.

        :param other: Object to compare with.
        :rtype: bool

        See Also
        --------
        __eq__ : equality function of this instance
        """
        return not (self == other)

    def __neg__(self) -> _DiracABC:
        """Returns a copy of this instance with negated ``data``.

        :rtype: _DiracABC

        Notes
        -----
        This method is called, if ``-()`` is called on this instance.
        """
        return self.__class__(-self.data)

    def __pos__(self) -> _DiracABC:
        """Returns this instance (without copying).

        :rtype: _DiracABC

        Notes
        -----
        This method is called, if ``+()`` is called on this instance.
        """
        return self

    def __add__(self, other: object) -> _DiracABC:
        """Addition of this instance with another object.

        For this instance, addition is only implemented for ``other`` being the same type as this instance as well as for
        ``type(other)`` is subclassed from ``numbers.Number`` or from ``numpy.ndarray``. In the latter case, the addition is propagated to each component of this instance (including the applications of broadcasting ruses if present; see also :np:func:`numpy.add`).

        :param other: other summand added to this instance.
        :type other: object
        :rtype: _DiracABC

        Notes
        -----
        This function is called, if one uses :py:func:`add` on ``self`` and ``other``, i.e. ``self + other``.
        """
        if isinstance(other, numbers.Number):
            return self.__class__(self.data + other)
        if isinstance(other, np.ndarray):
            broadcast_shape = (1,) * len(self.tshape) + other.shape
            return self.__class__(self.data + other.reshape(broadcast_shape))
        if isinstance(other, self.__class__):
            return self.__class__(self.data + other.data)
        raise TypeError(
            f"Operation {self.__class__} + {other.__class__} is not defined."
        )

    def __radd__(self, other: object) -> Any:
        if isinstance(other, numbers.Number) or isinstance(other, np.ndarray):
            return self + other

    def __sub__(self, other: Any) -> Any:
        """Substraction of another object from this instance.

        For this instance, substraction is only implemented for ``other`` being the same type as this instance as well as for
        ``type(other)`` is subclassed from ``numbers.Number`` or from ``numpy.ndarray``.  In the latter case, the substraction is propagated to each component of this instance (including the applications of broadcasting ruses if present; see also :func:`numpy.sub`).

        :param other: munient which is substracted from this instance.
        :type other: object
        :rtype: _DiracABC

        Notes
        -----
        This function is called, if one uses :py:func:`sub` on ``self`` and ``other``, i.e. ``self + other``.
        """
        return self.__add__(-other)

    def __rsub__(self, other: Any) -> Any:
        """Substraction of this instance from another object.

        For this instance, substraction is only implemented for ``other`` being the same type as this instance as well as for
        ``type(other)`` is subclassed from ``numbers.Number`` or from ``numpy.ndarray``.  In the latter case, the substraction is propagated to each component of this instance (including the applications of broadcasting ruses if present; see also :func:`numpy.sub`).

        :param other: subrahend this instance is substracted from.
        :type other: object
        :rtype: _DiracABC

        Notes
        -----
        This function is called, if one uses :py:func:`sub` on ``other`` and ``self``, i.e. ``other - self``.
        """
        return other + (-self)

    def __mul__(self, other: object) -> Any:
        """Multiplication of this instance with another object.

        For this instance, multiplication is only implemented for ``other`` type specified by the attribute ``ALLOWED_MATMUL_TYPES`` of this instance as well as for
        ``type(other)`` is subclassed from ``numbers.Number`` or from ``numpy.ndarray``.  In the latter case, the multiplication is propagated to each component of this instance (including the applications of broadcasting ruses if present; see also :func:`numpy.mul`).

        :param other: factor multiplyed with this instance.
        :type other: object
        :rtype: _DiracABC

        See Also
        --------
        __matmul__ : concrete implementation of the multiplication :class:`_DiracABC`.

        Notes
        -----
        This function is called, if one uses :py:func:`mul` on ``self`` and ``other``, i.e. ``self*other``.


        """
        if isinstance(other, numbers.Number):
            return self.__class__(self.data * other)
        if isinstance(other, np.ndarray):
            broadcast_shape = (1,) * len(self.tshape) + other.shape
            return self.__class__(self.data * other.reshape(broadcast_shape))
        if isinstance(other, _LorentzVectorType):
            return NotImplemented
        if type(other).__name__ in self.ALLOWED_MATMUL_TYPES:
            return self @ other
        raise TypeError(
            f"Operation {self.__class__} * {other.__class__} is not defined."
        )

    def __truediv__(self, other: Any) -> Any:
        if isinstance(other, numbers.Number) or isinstance(other, np.ndarray):
            return self * (1.0 / other)  # type: ignore

    def __matmul__(self, other: Any) -> Any:
        raise NotImplementedError

    def __rmul__(self, other: Any) -> Any:
        if isinstance(other, numbers.Number) or isinstance(other, np.ndarray):
            return self * other


@attr.s(cmp=False, slots=True)
class BiSpinor(_DiracABC):
    """Extension of numpys ndarray to describe bispinors.

    Here, a bispinor is represented as a column vector (in the sense of matrix multiplications) with four complex components. However, it has a different bevahiour under multiplications with other objects of the Dirac algebra.

    :param data: Numpy array which contains the data of the :class:`BiSpinor`. The shape of the leading axes needs to be ``(4,)``.
    :type data: np.ndarray
    :raises ValueError: if the leading axes haven't shape ``(4,)``

    Notes
    -----
    This is sort of an numerical representation of a spinor in the mathematical sense (see also `here <https://en.wikipedia.org/wiki/Spinor>`_). However, this is only reduced to implementations of the spinor arithmetics and the interplay with Dirac matrices, i.e. matrix representations of linear mappings between spinor spaces. This means, there are no transforms of this spinor type implemented yet (but maybe in the future).

    """

    TSHAPE = (4,)
    ALLOWED_MATMUL_TYPES = attr.ib(init=False, repr=False, default=("AdjointBiSpinor"))

    def __matmul__(self, other: Any) -> Any:
        """Multiplication within the Dirac Algebra.

        This function is called, if :class:`BiSpinor` is multiplyed by another instance of :class:`_DiracABC`. If other is a :class:`AdjointBiSpinor`, the result is equivalent to the outer product and results in a :class:`DiracMatrix`. Else, a :class: `TypeError` is raised. It is also called, if the :py:func:`matmul` function is called w.r.t. to a :class:`BiSpinor`, i.e. `self@other`.

        :param other: second factor of the multiplication.
        :type other: _DiracABC
        :rtype: :class:`_DiracABC` or :class:`numpy.ndarray`

        Notes
        -----
        - It is also called, if the :py:func:`matmul` function is called w.r.t. to a :class:`BiSpinor`, i.e. `self@other`.
        - If the ``shape`` of either this instance or ``other`` is not ``None``, the numpy broadcasting rules are applied for each component.
        """
        if isinstance(other, AdjointBiSpinor):
            return DiracMatrix(
                self.data[:, np.newaxis, ...] * other.data[np.newaxis, ...]
            )
        raise TypeError(
            f"Operation {self.__class__} @ {other.__class__} is not defined."
        )

    def vdot(self, other: BiSpinor) -> np.ndarray:
        """Dot product with another :class:`BiSpinor`, where this instance is adjoint.

        Essentially computes ``self.adjoint()@other``.

        :param other: another :class:`BiSpinor`.
        :type other: :class:`BiSpinor`.
        :rtype: numpy.ndarray

        :raises TypeError: If ``other`` is not a :class:`BiSpinor`.
        """
        if isinstance(other, BiSpinor):
            return self.adjoint() @ other
        raise TypeError(
            f"""Operation {self.__class__}.vdot({other.__class__})
             is not defined."""
        )

    def adjoint(self) -> AdjointBiSpinor:
        """Standard adjoint operation, i.e. the respective adjoint bispinor with complex conjugated values.

        :rtype: AdjointBiSpinor

        Notes
        -----
        This is not the Dirac conjugate of a bispinor! See :func:`dirac_adjoined`).
        """
        return AdjointBiSpinor(np.conjugate(self.data))


@attr.s(cmp=False, slots=True)
class AdjointBiSpinor(_DiracABC):
    r"""Extension of numpys ndarray to describe adjoint bispinors.

    This data type is similar to a bispinor (see :class:`BiSpinor`), but represented as a row vector (in the sense of matrix multiplications). This means, here an adjoint bispinor is referred to as the vector representation of a linear functional on a spinor space (see also `Linear form <https://en.wikipedia.org/wiki/Linear_form>`_), there the application of the functional to a bispinor is represented by its (matrix) multiplication (see also `Riesz' representation theorem <https://en.wikipedia.org/wiki/Riesz_representation_theorem>`_).

    :param data: Numpy array which contains the data of the :class:`AdjointBiSpinor`. The shape of the leading axes needs to be ``(4,)``.
    :type data: np.ndarray
    :raises ValueError: if the leading axes haven't shape ``(4,)``

    """

    TSHAPE = (4,)
    ALLOWED_MATMUL_TYPES = attr.ib(
        init=False, repr=False, default=("BiSpinor", "DiracMatrix")
    )

    def __matmul__(self, other: Any) -> Any:
        """Multiplication within the Dirac Algebra.

        This function is called, if :class:`AdjointBiSpinor` is multiplyed by another instance of :class:`_DiracABC`. Depending on the type of ``other`` different multiplication methods are used, where all methods are reduced to concrete matrix multiplications. For instance, if ``other`` is a :class:`BiSpinor`, the result is equivalent to the dot product. Furthermore, if ``other`` is a :class:`DiracMatrix`, the result is the matrix multiplication of ``self`` and ``other``.  If none of the above cases is present, a :class: `TypeError` is raised.

        :param other: second factor of the multiplication.
        :type other: _DiracABC
        :rtype: :class:`_DiracABC` or :class:`numpy.ndarray`

        Notes
        -----
        - It is also called, if the :py:func:`matmul` function is called w.r.t. to a :class:`AdjointBiSpinor`, i.e. `self@other`.
        - If the ``shape`` of either this instance or ``other`` is not ``None``, the numpy broadcasting rules are applied for each component.
        """
        if isinstance(other, BiSpinor):
            return np.einsum("i...,i... -> ...", self.data, other.data)
        elif isinstance(other, DiracMatrix):
            return AdjointBiSpinor(
                np.einsum("i...,ij... -> j...", self.data, other.data)
            )
        raise TypeError(
            f"Operation {self.__class__} @ {other.__class__} is not defined."
        )

    def adjoint(self) -> BiSpinor:
        """Standard adjoint operation, i.e. the respective bispinor with complex conjugated values.

        :rtype: BiSpinor

        Notes
        -----
        This is not the Dirac conjugate of a bispinor! See :func:`dirac_adjoined`).
        """
        return BiSpinor(np.conjugate(self.data))


@attr.s(cmp=False, slots=True)
class DiracMatrix(_DiracABC):
    """Extension of numpys ndarray to describe Dirac matrices.

    This data type represents a Dirac matrix, i.e. the matrix representation of linear mappings between spinor spaces.

    :param data: Numpy array which contains the data of the :class:`DiracMatrix`. The shape of the leading axes needs to be ``(4,4)``.
    :type data: np.ndarray
    :raises ValueError: if the first two axes haven't shape ``(4,4)``

    """

    TSHAPE = (4, 4)
    ALLOWED_MATMUL_TYPES = attr.ib(
        init=False, repr=False, default=("BiSpinor", "DiracMatrix")
    )

    def __matmul__(self, other: object) -> Any:  # returns a Union?!
        """Multiplication within the Dirac Algebra.

        This function is called, if :class:`DiracMatrix` is multiplyed by another instance of :class:`_DiracABC`. Depending on the type of ``other`` different multiplication methods are used, where all methods are reduced to concrete matrix multiplications. For instance, if ``other`` is a :class:`BiSpinor`, the result is equivalent to linear mapping represented by this Dirac Matrix acting on ``other`` resulting in another ``BiSpinor``. Furthermore, if ``other`` is a :class:`DiracMatrix`, the result is the matrix multiplication of ``self`` and ``other``.  If none of the above cases is present, a :class: `TypeError` is raised.

        :param other: second factor of the multiplication.
        :type other: _DiracABC
        :rtype: :class:`_DiracABC` or :class:`numpy.ndarray`

        Notes
        -----
        - It is also called, if the :py:func:`matmul` function is called w.r.t. to a :class:`DiracMatrix`, i.e. `self@other`.
        - If the ``shape`` of either this instance or ``other`` is not ``None``, the numpy broadcasting rules are applied for each component.
        """
        if isinstance(other, DiracMatrix):
            return DiracMatrix(
                np.matmul(self.data, other.data, axes=[(0, 1), (0, 1), (0, 1)])
            )
        if isinstance(other, BiSpinor):
            return BiSpinor(np.einsum("ij...,j... -> i...", self.data, other.data))
        raise TypeError(
            f"Operation {self.__class__} @ {other.__class__} is not defined."
        )


def UnitDiracMatrix(additional_axes: int | None = None) -> DiracMatrix:
    """Unit Dirac matrix with ones in the diagonal and zeros everywhere else.

    :param additional_axes = 0: number of additional axes with length 1. This can be used for broadcasting with other Dirac matrices.
    :type additional_axes: int
    :rtype: DiracMatrix

    """
    if additional_axes is None:
        additional_axes = 0
    return DiracMatrix(np.eye(4).reshape((4, 4) + (1,) * additional_axes))
