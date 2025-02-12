"""Kinematic tasks."""

import abc
from typing import NamedTuple

import numpy as np

from ..configuration import Configuration
from .exceptions import InvalidDamping, InvalidGain


class QuadraticObjective(NamedTuple):
    r"""Quadratic objective of the form :math:`\frac{1}{2} x^T H x + c^T x`."""

    hessian_matrix: np.ndarray
    """Hessian matrix, of shape (n_v, n_v)"""
    linear_vector: np.ndarray
    """Linear vector, of shape (n_v,)."""

    def evaluate(self, vector: np.ndarray) -> float:
        """Returns the value of the objective at the input vector."""
        return vector.T @ self.hessian_matrix @ vector + self.linear_vector @ vector


class KinematicTask(abc.ABC):
    """Abstract base class for kinematic tasks."""

    def __init__(
        self,
        cost_vector: np.ndarray,
        task_gain: float = 1.0,
        lm_damping_factor: float = 0.0,
    ):
        """Constructor.\n\n        Args:\n            cost_vector: Cost vector with the same dimension as the error of the task.\n            task_gain: Task gain alpha in [0, 1] for additional low-pass filtering. Defaults\n                to 1.0 (no filtering) for dead-beat control.\n            lm_damping_factor: Unitless scale of the Levenberg-Marquardt (only when the error\n                is large) regularization term, which helps when targets are infeasible.\n                Increase this value if the task is too jerky under unfeasible targets, but\n                beware that a larger damping slows down the task.\n        """
        if not 0.0 <= task_gain <= 1.0:
            raise InvalidGain("`task_gain` must be in the range [0, 1]")

        if lm_damping_factor < 0.0:
            raise InvalidDamping("`lm_damping_factor` must be >= 0")

        self.cost = cost_vector
        self.gain = task_gain
        self.lm_damping = lm_damping_factor

    @abc.abstractmethod
    def calculate_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the task error function at the current configuration.\n\n        The error function :math:`e(q) \in \mathbb{R}^{k}` is the quantity that\n        the task aims to drive to zero (:math:`k` is the dimension of the\n        task). It appears in the first-order task dynamics:\n\n        .. math::\n\n            J(q) \Delta q = -\alpha e(q)\n\n        The Jacobian matrix :math:`J(q) \in \mathbb{R}^{k \times n_v}`, with\n        :math:`n_v` the dimension of the robot's tangent space, is the\n        derivative of the task error :math:`e(q)` with respect to the\n        configuration :math:`q \in \mathbb{R}^{n_q}`. This Jacobian is\n        implemented in :func:`KinematicTask.calculate_jacobian`. Finally, the\n        configuration displacement :math:`\Delta q` is the output of inverse\n        kinematics.\n\n        In the first-order task dynamics, the error :math:`e(q)` is multiplied\n        by the task gain :math:`\alpha \in [0, 1]`. This gain can be 1.0 for\n        dead-beat control (*i.e.* converge as fast as possible), but might be\n        unstable as it neglects our first-order approximation. Lower values\n        cause slow down the task, similar to low-pass filtering.\n\n        Args:\n            configuration: Robot configuration :math:`q`.\n\n        Returns:\n            Task error vector :math:`e(q)`.\n        """
        raise NotImplementedError

    @abc.abstractmethod
    def calculate_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the task Jacobian at the current configuration.\n\n        The task Jacobian :math:`J(q) \in \mathbb{R}^{k \times n_v}` is the first order\n        derivative of the error :math:`e(q) \in \mathbb{R}^{k}` that defines the task,\n        with :math:`k` the dimension of the task and :math:`(n_v,)` the dimension of the\n        robot's tangent space.\n\n        Args:\n            configuration: Robot configuration :math:`q`.\n\n        Returns:\n            Task Jacobian :math:`J(q)`.\n        """
        raise NotImplementedError

    def compute_quadratic_program_objective(self, configuration: Configuration) -> QuadraticObjective:
        r"""Compute the matrix-vector pair :math:`(H, c)` of the QP objective.\n\n        This pair is such that the contribution of the task to the QP objective is:\n\n        .. math::\n\n            \| J \Delta q + \alpha e \|_{W}^2 = \frac{1}{2} \Delta q^T H\n            \Delta q + c^T q\n\n        The weight matrix :math:`W \in \mathbb{R}^{k \times k}` weights and\n        normalizes task coordinates to the same unit. The unit of the overall\n        contribution is [cost]^2. The configuration displacement :math:`\Delta\n        q` is the output of inverse kinematics (we divide it by dt to get a\n        commanded velocity).\n\n        Args:\n            configuration: Robot configuration :math:`q`.\n\n        Returns:\n            Pair :math:`(H(q), c(q))`.\n        """
        jacobian = self.calculate_jacobian(configuration)  # (k, nv)
        weighted_error = -self.gain * self.calculate_error(configuration)  # (k,)

        weight_matrix = np.diag(self.cost)
        weighted_jacobian = weight_matrix @ jacobian
        weighted_error = weight_matrix @ weighted_error

        damping_effect = self.lm_damping * weighted_error @ weighted_error
        identity_matrix = np.eye(configuration.model.nv)

        hessian_matrix = weighted_jacobian.T @ weighted_jacobian + damping_effect * identity_matrix  # (nv, nv)
        linear_vector = -weighted_error.T @ weighted_jacobian  # (nv,)

        return QuadraticObjective(hessian_matrix, linear_vector)