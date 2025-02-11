"""Kinematic tasks."""

import abc
from typing import NamedTuple

import numpy as np

from ..configuration import Configuration
from .exceptions import InvalidDamping, InvalidGain


class Objective(NamedTuple):
    r"""Quadratic objective of the form :math:`\frac{1}{2} x^T H x + c^T x`."""

    H: np.ndarray
    """Hessian matrix, of shape (n_v, n_v)"""
    c: np.ndarray
    """Linear vector, of shape (n_v,)."""

    def value(self, x: np.ndarray) -> float:
        """Returns the value of the objective at the input vector."""
        return x.T @ self.H @ x + self.c @ x


class Task(abc.ABC):
    """Abstract base class for kinematic tasks.

    This class defines the structure for kinematic tasks, which are used to compute
    the error and Jacobian necessary for inverse kinematics. Subclasses must implement
    the `compute_error` and `compute_jacobian` methods.

    The task dynamics are defined by:

    .. math::

        J(q) \Delta q = -\alpha e(q)

    where :math:`J(q)` is the task Jacobian, :math:`\Delta q` is the configuration
    displacement, :math:`\alpha` is the task gain, and :math:`e(q)` is the task error.
    """

    def __init__(
        self,
        cost: np.ndarray,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        """Constructor.

        Args:
            cost: Cost vector with the same dimension as the error of the task.
            gain: Task gain :math:`\alpha` in [0, 1] for additional low-pass filtering.
                Defaults to 1.0 (no filtering) for dead-beat control.
            lm_damping: Unitless scale of the Levenberg-Marquardt regularization term,
                which helps when targets are infeasible. Increase this value if the task
                is too jerky under unfeasible targets, but beware that a larger damping
                slows down the task.

        Raises:
            InvalidGain: If `gain` is not in the range [0, 1].
            InvalidDamping: If `lm_damping` is negative.
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
        r"""Compute the task error function at the current configuration.

        The error function :math:`e(q) \in \mathbb{R}^{k}` is the quantity that
        the task aims to drive to zero (:math:`k` is the dimension of the task).

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Task error vector :math:`e(q)`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the task Jacobian at the current configuration.

        The task Jacobian :math:`J(q) \in \mathbb{R}^{k \times n_v}` is the first-order
        derivative of the error :math:`e(q) \in \mathbb{R}^{k}` with respect to the
        configuration :math:`q \in \mathbb{R}^{n_q}`, where :math:`k` is the dimension
        of the task and :math:`n_v` is the dimension of the robot's tangent space.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Task Jacobian matrix :math:`J(q)`.
        """
        raise NotImplementedError

    def compute_qp_objective(self, configuration: Configuration) -> Objective:
        r"""Compute the matrix-vector pair :math:`(H, c)` of the QP objective.

        This pair is such that the contribution of the task to the QP objective is:

        .. math::

            \| J \Delta q + \alpha e \|_{W}^2 = \frac{1}{2} \Delta q^T H
            \Delta q + c^T q

        The weight matrix :math:`W \in \mathbb{R}^{k \times k}` weights and
        normalizes task coordinates to the same unit. The unit of the overall
        contribution is [cost]^2. The configuration displacement :math:`\Delta
        q` is the output of inverse kinematics (we divide it by dt to get a
        commanded velocity).

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Pair :math:`(H(q), c(q))`.
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