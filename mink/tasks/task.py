"""Kinematic tasks."""

import abc
from typing import NamedTuple

import numpy as np

from ..configuration import Configuration
from .exceptions import InvalidDamping, InvalidGain


class Objective(NamedTuple):
    r"""Quadratic objective of the form :math:`\frac{1}{2} x^T H x + c^T x`."""

    H: np.ndarray
    r"""Hessian matrix, shape :math:`(n_v, n_v)`"""
    c: np.ndarray
    r"""Linear vector, shape :math:`(n_v,)`"""

    def value(self, x: np.ndarray) -> float:
        r"""Returns the value of the objective at the input vector :math:`x`.

        Args:
            x: Input vector, shape :math:`(n_v,)`.

        Returns:
            Value of the objective, a scalar.
        """
        return 0.5 * x.T @ self.H @ x + self.c @ x


class Task(abc.ABC):
    r"""Abstract base class for kinematic tasks.

    This class defines the interface for kinematic tasks, which are used to specify
    desired behaviors for a robot's configuration. Subclasses must implement the
    `compute_error` and `compute_jacobian` methods to define the specific task dynamics.

    The task dynamics are governed by the equation:

    .. math::

        J(q) \Delta q = -\alpha e(q)

    where :math:`J(q)` is the task Jacobian, :math:`\Delta q` is the configuration
    displacement, :math:`\alpha` is the task gain, and :math:`e(q)` is the task error.

    Attributes:
        cost: Cost vector with the same dimension as the error of the task.
        gain: Task gain :math:`\alpha` in [0, 1] for additional low-pass filtering.
        lm_damping: Unitless scale of the Levenberg-Marquardt regularization term.
    """

    def __init__(
        self,
        cost: np.ndarray,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        r"""Initialize a kinematic task with specified cost, gain, and Levenberg-Marquardt damping.

        Args:
            cost: Cost vector with the same dimension as the error of the task.
            gain: Task gain :math:`\alpha` in [0, 1] for additional low-pass filtering.
                Defaults to 1.0 (no filtering) for dead-beat control.
            lm_damping: Unitless scale of the Levenberg-Marquardt regularization term.
                Helps when targets are infeasible. Increase this value if the task is too
                jerky under unfeasible targets, but beware that a larger damping slows down
                the task.

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
        It appears in the first-order task dynamics:

        .. math::

            J(q) \Delta q = -\alpha e(q)

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Task error vector :math:`e(q)`, shape :math:`(k,)`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the task Jacobian at the current configuration.

        The task Jacobian :math:`J(q) \in \mathbb{R}^{k \times n_v}` is the first order
        derivative of the error :math:`e(q) \in \mathbb{R}^{k}` that defines the task,
        with :math:`k` the dimension of the task and :math:`n_v` the dimension of the
        robot's tangent space.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Task Jacobian :math:`J(q)`, shape :math:`(k, n_v)`.
        """
        raise NotImplementedError

    def compute_qp_objective(self, configuration: Configuration) -> Objective:
        r"""Compute the matrix-vector pair :math:`(H, c)` of the QP objective.

        This pair is such that the contribution of the task to the QP objective is:

        .. math::

            \| J \Delta q + \alpha e \|_{W}^2 = \frac{1}{2} \Delta q^T H
            \Delta q + c^T \Delta q

        The weight matrix :math:`W \in \mathbb{R}^{k \times k}` weights and
        normalizes task coordinates to the same unit. The unit of the overall
        contribution is [cost]^2. The configuration displacement :math:`\Delta
        q` is the output of inverse kinematics (we divide it by dt to get a
        commanded velocity).

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Pair :math:`(H(q), c(q))`, where :math:`H(q)` is the Hessian matrix
            and :math:`c(q)` is the linear vector, both in terms of the configuration
            displacement :math:`\Delta q`. The Hessian matrix :math:`H(q)` has shape
            :math:`(n_v, n_v)` and the linear vector :math:`c(q)` has shape :math:`(n_v,)`.
        """
        jacobian = self.compute_jacobian(configuration)  # (k, nv)
        error = self.compute_error(configuration)  # (k,)
        weighted_error = -self.gain * self.cost * error  # (k,)

        weight_matrix = np.diag(self.cost)
        weighted_jacobian = weight_matrix @ jacobian  # (k, nv)

        lm_term = self.lm_damping * weighted_error @ weighted_error
        identity_matrix = np.eye(configuration.model.nv)

        H = weighted_jacobian.T @ weighted_jacobian + lm_term * identity_matrix  # (nv, nv)
        c = weighted_error.T @ weighted_jacobian  # (nv,)

        return Objective(H, c)