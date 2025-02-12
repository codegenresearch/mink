import abc
from typing import Union, overload

import numpy as np
from typing_extensions import Self


class MatrixLieGroup(abc.ABC):
    """Interface definition for matrix Lie groups.\n\n    Attributes:\n        matrix_dim: Dimension of the square matrix output.\n        parameters_dim: Dimension of the underlying parameters.\n        tangent_dim: Dimension of the tangent space.\n        space_dim: Dimension of the coordinates that can be transformed.\n    """

    matrix_dim: int
    parameters_dim: int
    tangent_dim: int
    space_dim: int

    @overload
    def __matmul__(self, other: Self) -> Self: ...

    @overload
    def __matmul__(self, other: np.ndarray) -> np.ndarray: ...

    def __matmul__(self, other: Union[Self, np.ndarray]) -> Union[Self, np.ndarray]:
        """Overload of the @ operator to compose transformations or apply to points."""
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
        """Constructs a group member from its matrix representation."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def sample_uniform(cls) -> Self:
        """Samples a uniform element from the group."""
        raise NotImplementedError

    # Accessors.

    @abc.abstractmethod
    def as_matrix(self) -> np.ndarray:
        """Returns the transformation as a matrix."""
        raise NotImplementedError

    @abc.abstractmethod
    def parameters(self) -> np.ndarray:
        """Returns the underlying parameter representation."""
        raise NotImplementedError

    # Operations.

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
        """Computes the exponential map from the tangent space to the group, i.e., expm(wedge(tangent))."""
        raise NotImplementedError

    @abc.abstractmethod
    def log(self) -> np.ndarray:
        """Computes the logarithmic map from the group to the tangent space, i.e., vee(logm(transformation matrix))."""
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
        """Normalizes the transformation parameters and returns the normalized transformation."""
        raise NotImplementedError

    # Plus and minus operators.

    # Eqn. 25.
    def rplus(self, other: np.ndarray) -> Self:
        """Right plus operator: adds a tangent vector to the transformation."""
        return self @ self.exp(other)

    # Eqn. 26.
    def rminus(self, other: Self) -> np.ndarray:
        """Right minus operator: computes the tangent vector difference between two transformations."""
        return (other.inverse() @ self).log()

    # Eqn. 27.
    def lplus(self, other: np.ndarray) -> Self:
        """Left plus operator: adds a tangent vector to the transformation."""
        return self.exp(other) @ self

    # Eqn. 28.
    def lminus(self, other: Self) -> np.ndarray:
        """Left minus operator: computes the tangent vector difference between two transformations."""
        return (self @ other.inverse()).log()

    def plus(self, other: np.ndarray) -> Self:
        """Alias for the right plus operator."""
        return self.rplus(other)

    def minus(self, other: Self) -> np.ndarray:
        """Alias for the right minus operator."""
        return self.rminus(other)

    # Jacobians.

    @classmethod
    @abc.abstractmethod
    def ljac(cls, other: np.ndarray) -> np.ndarray:
        """Computes the left Jacobian of the exponential map."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def ljacinv(cls, other: np.ndarray) -> np.ndarray:
        """Computes the inverse of the left Jacobian."""
        # NOTE: Can be computed as np.linalg.inv(cls.ljac(other)).
        raise NotImplementedError

    # Eqn. 67.
    @classmethod
    def rjac(cls, other: np.ndarray) -> np.ndarray:
        """Computes the right Jacobian of the exponential map."""
        return cls.ljac(-other)

    @classmethod
    def rjacinv(cls, other: np.ndarray) -> np.ndarray:
        """Computes the inverse of the right Jacobian."""
        return cls.ljacinv(-other)

    # Eqn. 79.
    def jlog(self) -> np.ndarray:
        """Computes the Jacobian of the logarithmic map."""
        return self.rjacinv(self.log())