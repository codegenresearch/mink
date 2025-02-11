"""Kinematic tasks."""

import abc
from typing import NamedTuple

import numpy as np

from ..configuration import Configuration
from .exceptions import InvalidDamping, InvalidGain


class Objective(NamedTuple):
    """Quadratic objective of the form 0.5 * x^T H x + c^T x."""

    H: np.ndarray
    """Hessian matrix, shape (n_v, n_v)"""
    c: np.ndarray
    """Linear vector, shape (n_v,)"""

    def value(self, x: np.ndarray) -> float:
        """Returns the value of the objective at the input vector."""
        return 0.5 * x.T @ self.H @ x + self.c @ x


class Task(abc.ABC):
    """Abstract base class for kinematic tasks."""

    def __init__(
        self,
        cost: np.ndarray,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        """
        Args:
            cost: Cost vector with the same dimension as the error of the task.
            gain: Task gain alpha in [0, 1] for additional low-pass filtering. Defaults
                to 1.0 (no filtering) for dead-beat control.
            lm_damping: Unitless scale of the Levenberg-Marquardt (only when the error
                is large) regularization term, which helps when targets are infeasible.
                Increase this value if the task is too jerky under unfeasible targets, but
                beware that a larger damping slows down the task.
        """
        if not 0.0 <= gain <= 1.0:
            raise InvalidGain("`gain` must be in the range [0, 1]")

        if lm_damping < 0.0:
            raise InvalidDamping("`lm_damping` must be >= 0")

        self.cost = cost
        self.gain = gain
        self.lm_damping = lm_damping

    @abc.abstractmethod
    def compute_error(self, configuration: Configuration) -> np.ndarray:
        """
        Compute the task error function at the current configuration.

        Args:
            configuration: Robot configuration.

        Returns:
            Task error vector.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        """
        Compute the task Jacobian at the current configuration.

        Args:
            configuration: Robot configuration.

        Returns:
            Task Jacobian.
        """
        raise NotImplementedError

    def compute_qp_objective(self, configuration: Configuration) -> Objective:
        """
        Compute the matrix-vector pair (H, c) of the QP objective.

        Args:
            configuration: Robot configuration.

        Returns:
            Pair (H(q), c(q)).
        """
        jacobian = self.compute_jacobian(configuration)  # (k, nv)
        minus_gain_error = -self.gain * self.compute_error(configuration)  # (k,)

        weight = np.diag(self.cost)
        weighted_jacobian = weight @ jacobian
        weighted_error = weight @ minus_gain_error

        mu = self.lm_damping * weighted_error @ weighted_error
        eye_tg = np.eye(configuration.model.nv)

        H = weighted_jacobian.T @ weighted_jacobian + mu * eye_tg  # (nv, nv)
        c = -weighted_error.T @ weighted_jacobian  # (nv,)

        return Objective(H, c)