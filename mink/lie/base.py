import abc
from typing import Union, overload

import numpy as np
from typing_extensions import Self


class MatrixLieGroup(abc.ABC):
    """Abstract base class for matrix Lie groups.

    Attributes:
        matrix_dim: Dimension of the square matrix representation.
        parameters_dim: Dimension of the underlying parameters.
        tangent_dim: Dimension of the tangent space.
        space_dim: Dimension of the space on which the group acts.
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
        """Overloads the @ operator for group composition and transformation application."""
        if isinstance(other, np.ndarray):
            return self.apply(target=other)
        assert isinstance(other, MatrixLieGroup)
        return self.multiply(other=other)

    # Factory.

    @classmethod
    @abc.abstractmethod
    def identity(cls) -> Self:
        """Returns the identity element."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_matrix(cls, matrix: np.ndarray) -> Self:
        """Constructs a group element from a matrix."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def sample_uniform(cls) -> Self:
        """Draws a random sample from the group."""
        raise NotImplementedError

    # Accessors.

    @abc.abstractmethod
    def as_matrix(self) -> np.ndarray:
        """Gets the matrix representation."""
        raise NotImplementedError

    @abc.abstractmethod
    def parameters(self) -> np.ndarray:
        """Gets the underlying parameters."""
        raise NotImplementedError

    # Operations.

    @abc.abstractmethod
    def apply(self, target: np.ndarray) -> np.ndarray:
        """Applies the group action to a point."""
        raise NotImplementedError

    @abc.abstractmethod
    def multiply(self, other: Self) -> Self:
        """Composes this transformation with another."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def exp(cls, tangent: np.ndarray) -> Self:
        """Computes the exponential map."""
        raise NotImplementedError

    @abc.abstractmethod
    def log(self) -> np.ndarray:
        """Computes the logarithmic map."""
        raise NotImplementedError

    @abc.abstractmethod
    def adjoint(self) -> np.ndarray:
        """Computes the adjoint."""
        raise NotImplementedError

    @abc.abstractmethod
    def inverse(self) -> Self:
        """Computes the inverse."""
        raise NotImplementedError

    @abc.abstractmethod
    def normalize(self) -> Self:
        """Normalizes the group element."""
        raise NotImplementedError

    # Right and left plus and minus.

    # Eqn. 25.
    def rplus(self, other: np.ndarray) -> Self:
        """Applies a tangent vector using the right plus operator."""
        return self @ self.exp(other)

    # Eqn. 26.
    def rminus(self, other: Self) -> np.ndarray:
        """Computes the difference using the right minus operator."""
        return (other.inverse() @ self).log()

    # Eqn. 27.
    def lplus(self, other: np.ndarray) -> Self:
        """Applies a tangent vector using the left plus operator."""
        return self.exp(other) @ self

    # Eqn. 28.
    def lminus(self, other: Self) -> np.ndarray:
        """Computes the difference using the left minus operator."""
        return (self @ other.inverse()).log()

    def plus(self, other: np.ndarray) -> Self:
        """Alias for rplus."""
        return self.rplus(other)

    def minus(self, other: Self) -> np.ndarray:
        """Alias for rminus."""
        return self.rminus(other)

    # Jacobians.

    @classmethod
    @abc.abstractmethod
    def ljac(cls, other: np.ndarray) -> np.ndarray:
        """Computes the left Jacobian."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def ljacinv(cls, other: np.ndarray) -> np.ndarray:
        """Computes the inverse of the left Jacobian."""
        raise NotImplementedError

    # Eqn. 67.
    @classmethod
    def rjac(cls, other: np.ndarray) -> np.ndarray:
        """Computes the right Jacobian."""
        return cls.ljac(-other)

    # Eqn. 68.
    @classmethod
    def rjacinv(cls, other: np.ndarray) -> np.ndarray:
        """Computes the inverse of the right Jacobian."""
        return cls.ljacinv(-other)

    # Eqn. 79.
    def jlog(self) -> np.ndarray:
        """Computes the Jacobian of the logarithmic map."""
        return self.rjacinv(self.log())