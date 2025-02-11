import abc
from typing import Union, overload

import numpy as np
from typing_extensions import Self


class MatrixLieGroup(abc.ABC):
    """Interface definition for matrix Lie groups.

    Attributes:
        matrix_dim: Dimension of the square matrix output.
        parameters_dim: Dimension of the underlying parameters.
        tangent_dim: Dimension of the tangent space.
        space_dim: Dimension of the coordinates that can be transformed.
    """

    matrix_dim: int
    parameters_dim: int
    tangent_dim: int
    space_dim: int

    @overload
    def __matmul__(self, other: Self) -> Self: ...

    @overload
    def __matmul__(self, other: np.ndarray) -> np.ndarray: ...

    def __matmul__(self, other: Union[Self, np.ndarray]) -> Union[Self, np.ndarray]:
        """Overload of the @ operator to support both group composition and matrix application."""
        if isinstance(other, np.ndarray):
            return self.apply(target=other)
        assert isinstance(other, MatrixLieGroup)
        return self.multiply(other=other)

    # Factory methods.

    @classmethod
    @abc.abstractmethod
    def identity(cls) -> Self:
        """Returns the identity element of the group."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_matrix(cls, matrix: np.ndarray) -> Self:
        """Creates a group member from its matrix representation."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def sample_uniform(cls) -> Self:
        """Draws a uniform sample from the group."""
        raise NotImplementedError

    # Accessor methods.

    @abc.abstractmethod
    def as_matrix(self) -> np.ndarray:
        """Returns the transformation as a matrix."""
        raise NotImplementedError

    @abc.abstractmethod
    def parameters(self) -> np.ndarray:
        """Returns the underlying parameter representation of the transformation."""
        raise NotImplementedError

    # Group operations.

    @abc.abstractmethod
    def apply(self, target: np.ndarray) -> np.ndarray:
        """Applies the group action to a point or set of points."""
        raise NotImplementedError

    @abc.abstractmethod
    def multiply(self, other: Self) -> Self:
        """Composes this transformation with another transformation."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def exp(cls, tangent: np.ndarray) -> Self:
        """Computes the exponential map of a tangent vector, resulting in a group element."""
        raise NotImplementedError

    @abc.abstractmethod
    def log(self) -> np.ndarray:
        """Computes the logarithmic map of the transformation, resulting in a tangent vector."""
        raise NotImplementedError

    @abc.abstractmethod
    def adjoint(self) -> np.ndarray:
        """Computes the adjoint representation of the transformation."""
        raise NotImplementedError

    @abc.abstractmethod
    def inverse(self) -> Self:
        """Computes the inverse of the transformation."""
        raise NotImplementedError

    @abc.abstractmethod
    def normalize(self) -> Self:
        """Normalizes the transformation to ensure it lies on the group manifold."""
        raise NotImplementedError

    # Right and left plus and minus operators.

    # Equation 25.
    def rplus(self, other: np.ndarray) -> Self:
        """Applies a tangent vector to the transformation using the right plus operator."""
        return self @ self.exp(other)

    # Equation 26.
    def rminus(self, other: Self) -> np.ndarray:
        """Computes the difference between two transformations using the right minus operator."""
        return (other.inverse() @ self).log()

    # Equation 27.
    def lplus(self, other: np.ndarray) -> Self:
        """Applies a tangent vector to the transformation using the left plus operator."""
        return self.exp(other) @ self

    # Equation 28.
    def lminus(self, other: Self) -> np.ndarray:
        """Computes the difference between two transformations using the left minus operator."""
        return (self @ other.inverse()).log()

    def plus(self, other: np.ndarray) -> Self:
        """Alias for the right plus operator."""
        return self.rplus(other)

    def minus(self, other: Self) -> np.ndarray:
        """Alias for the right minus operator."""
        return self.rminus(other)

    # Jacobian methods.

    @classmethod
    @abc.abstractmethod
    def ljac(cls, other: np.ndarray) -> np.ndarray:
        """Computes the left Jacobian of the exponential map."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def ljacinv(cls, other: np.ndarray) -> np.ndarray:
        """Computes the inverse of the left Jacobian of the exponential map."""
        # NOTE: This can be computed as np.linalg.inv(cls.ljac(other)).
        raise NotImplementedError

    # Equation 67.
    @classmethod
    def rjac(cls, other: np.ndarray) -> np.ndarray:
        """Computes the right Jacobian of the exponential map."""
        return cls.ljac(-other)

    @classmethod
    def rjacinv(cls, other: np.ndarray) -> np.ndarray:
        """Computes the inverse of the right Jacobian of the exponential map."""
        return cls.ljacinv(-other)

    # Equation 79.
    def jlog(self) -> np.ndarray:
        """Computes the Jacobian of the logarithmic map."""
        return self.rjacinv(self.log())