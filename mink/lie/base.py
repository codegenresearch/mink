import abc
from typing import Union, overload

import numpy as np
from typing_extensions import Self


class MatrixLieGroup(abc.ABC):
    """Abstract base class defining the interface for matrix Lie groups.

    This class outlines the essential operations and properties that any matrix Lie group should implement.
    It provides a framework for group operations, transformations, and their mathematical representations.

    Attributes:
        matrix_dim (int): The dimension of the square matrix representation of the group element.
        parameters_dim (int): The dimension of the underlying parameterization of the group element.
        tangent_dim (int): The dimension of the tangent space at the identity element.
        space_dim (int): The dimension of the space on which the group acts.
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
        """Overloads the @ operator to support both group composition and transformation application.

        If the right operand is a group element, it performs group composition.
        If the right operand is a numpy array, it applies the group action to the array.

        Args:
            other (Union[Self, np.ndarray]): The right operand for the operation.

        Returns:
            Union[Self, np.ndarray]: The result of the operation, either a new group element or a transformed array.
        """
        if isinstance(other, np.ndarray):
            return self.apply(target=other)
        assert isinstance(other, MatrixLieGroup)
        return self.multiply(other=other)

    # Factory methods.

    @classmethod
    @abc.abstractmethod
    def identity(cls) -> Self:
        """Returns the identity element of the group.

        The identity element is the group element that, when composed with any other element, leaves it unchanged.

        Returns:
            Self: The identity element of the group.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_matrix(cls, matrix: np.ndarray) -> Self:
        """Constructs a group element from its matrix representation.

        Args:
            matrix (np.ndarray): The matrix representation of the group element.

        Returns:
            Self: The group element corresponding to the given matrix.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def sample_uniform(cls) -> Self:
        """Draws a random sample from the group with a uniform distribution.

        Returns:
            Self: A randomly sampled group element.
        """
        raise NotImplementedError

    # Accessor methods.

    @abc.abstractmethod
    def as_matrix(self) -> np.ndarray:
        """Returns the matrix representation of the group element.

        Returns:
            np.ndarray: The matrix representation of the group element.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def parameters(self) -> np.ndarray:
        """Returns the underlying parameterization of the group element.

        Returns:
            np.ndarray: The parameters representing the group element.
        """
        raise NotImplementedError

    # Group operations.

    @abc.abstractmethod
    def apply(self, target: np.ndarray) -> np.ndarray:
        """Applies the group action to a point in the space.

        Args:
            target (np.ndarray): The point to be transformed.

        Returns:
            np.ndarray: The transformed point.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def multiply(self, other: Self) -> Self:
        """Composes this group element with another group element.

        Args:
            other (Self): The group element to be composed with.

        Returns:
            Self: The result of the composition.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def exp(cls, tangent: np.ndarray) -> Self:
        """Computes the exponential map of a tangent vector.

        The exponential map maps a tangent vector at the identity element to a group element.

        Args:
            tangent (np.ndarray): The tangent vector.

        Returns:
            Self: The group element corresponding to the tangent vector.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def log(self) -> np.ndarray:
        """Computes the logarithmic map of the group element.

        The logarithmic map maps a group element to a tangent vector at the identity element.

        Returns:
            np.ndarray: The tangent vector corresponding to the group element.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def adjoint(self) -> np.ndarray:
        """Computes the adjoint representation of the group element.

        The adjoint representation is a linear map that describes how the group element acts on the tangent space.

        Returns:
            np.ndarray: The adjoint representation of the group element.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def inverse(self) -> Self:
        """Computes the inverse of the group element.

        Returns:
            Self: The inverse of the group element.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def normalize(self) -> Self:
        """Normalizes the group element.

        This method ensures that the group element satisfies the group constraints, such as orthogonality for rotation matrices.

        Returns:
            Self: The normalized group element.
        """
        raise NotImplementedError

    # Right and left plus and minus operators.

    # Eqn. 25.
    def rplus(self, other: np.ndarray) -> Self:
        """Applies a tangent vector to the group element using the right plus operator.

        Args:
            other (np.ndarray): The tangent vector.

        Returns:
            Self: The resulting group element after applying the tangent vector.
        """
        return self @ self.exp(other)

    # Eqn. 26.
    def rminus(self, other: Self) -> np.ndarray:
        """Computes the difference between two group elements using the right minus operator.

        Args:
            other (Self): The other group element.

        Returns:
            np.ndarray: The tangent vector representing the difference between the two group elements.
        """
        return (other.inverse() @ self).log()

    # Eqn. 27.
    def lplus(self, other: np.ndarray) -> Self:
        """Applies a tangent vector to the group element using the left plus operator.

        Args:
            other (np.ndarray): The tangent vector.

        Returns:
            Self: The resulting group element after applying the tangent vector.
        """
        return self.exp(other) @ self

    # Eqn. 28.
    def lminus(self, other: Self) -> np.ndarray:
        """Computes the difference between two group elements using the left minus operator.

        Args:
            other (Self): The other group element.

        Returns:
            np.ndarray: The tangent vector representing the difference between the two group elements.
        """
        return (self @ other.inverse()).log()

    def plus(self, other: np.ndarray) -> Self:
        """Alias for the right plus operator.

        Args:
            other (np.ndarray): The tangent vector.

        Returns:
            Self: The resulting group element after applying the tangent vector.
        """
        return self.rplus(other)

    def minus(self, other: Self) -> np.ndarray:
        """Alias for the right minus operator.

        Args:
            other (Self): The other group element.

        Returns:
            np.ndarray: The tangent vector representing the difference between the two group elements.
        """
        return self.rminus(other)

    # Jacobian methods.

    @classmethod
    @abc.abstractmethod
    def ljac(cls, other: np.ndarray) -> np.ndarray:
        """Computes the left Jacobian of the exponential map.

        Args:
            other (np.ndarray): The tangent vector.

        Returns:
            np.ndarray: The left Jacobian matrix.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def ljacinv(cls, other: np.ndarray) -> np.ndarray:
        """Computes the inverse of the left Jacobian of the exponential map.

        Args:
            other (np.ndarray): The tangent vector.

        Returns:
            np.ndarray: The inverse of the left Jacobian matrix.
        """
        # NOTE: Can just be np.linalg.inv(cls.ljac(other)).
        raise NotImplementedError

    # Eqn. 67.
    @classmethod
    def rjac(cls, other: np.ndarray) -> np.ndarray:
        """Computes the right Jacobian of the exponential map.

        Args:
            other (np.ndarray): The tangent vector.

        Returns:
            np.ndarray: The right Jacobian matrix.
        """
        return cls.ljac(-other)

    @classmethod
    def rjacinv(cls, other: np.ndarray) -> np.ndarray:
        """Computes the inverse of the right Jacobian of the exponential map.

        Args:
            other (np.ndarray): The tangent vector.

        Returns:
            np.ndarray: The inverse of the right Jacobian matrix.
        """
        return cls.ljacinv(-other)

    # Eqn. 79.
    def jlog(self) -> np.ndarray:
        """Computes the Jacobian of the logarithmic map.

        Returns:
            np.ndarray: The Jacobian matrix of the logarithmic map.
        """
        return self.rjacinv(self.log())