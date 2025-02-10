"""Posture task implementation.

This module defines a task for regulating the joint angles of a robot towards a desired posture.
A posture is defined as a vector of actuated joint angles, and this task does not affect floating-base coordinates.

References:
- For more information on the Mujoco library, see: https://mujoco.readthedocs.io/
"""

from typing import Optional

import mujoco
import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from ..utils import get_freejoint_dims
from .exceptions import InvalidTarget, TargetNotSet, TaskDefinitionError
from .task import Task


class PostureTask(Task):
    """Regulate the joint angles of the robot towards a desired posture.

    A posture is a vector of actuated joint angles. Floating-base coordinates are not
    affected by this task.

    Attributes:
        target_q: Target configuration for the robot's joint angles.
        _v_ids: Indices of the free joint dimensions, if any.
        k: Number of degrees of freedom for the task.
        nq: Total number of joint angles in the model.
    """

    target_q: Optional[np.ndarray]
    _v_ids: Optional[np.ndarray]
    k: int
    nq: int

    def __init__(
        self,
        model: mujoco.MjModel,
        cost: float,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        """Initialize the PostureTask.

        Args:
            model: Mujoco model of the robot.
            cost: Cost associated with the task. Must be non-negative.
            gain: Gain for the task. Defaults to 1.0.
            lm_damping: Damping for the Levenberg-Marquardt algorithm. Defaults to 0.0.

        Raises:
            TaskDefinitionError: If the cost is negative.
        """
        if cost < 0.0:
            raise TaskDefinitionError(f"{self.__class__.__name__} cost must be >= 0")

        super().__init__(
            cost=np.asarray([cost] * model.nv),
            gain=gain,
            lm_damping=lm_damping,
        )
        self.target_q = None

        # Determine the indices of the free joint dimensions, if any
        _, v_ids_or_none = get_freejoint_dims(model)
        self._v_ids = np.asarray(v_ids_or_none) if v_ids_or_none is not None else None

        # Set the number of degrees of freedom and total number of joint angles
        self.k = model.nv
        self.nq = model.nq

    def set_target(self, target_q: npt.ArrayLike) -> None:
        """Set the target posture for the robot.

        Args:
            target_q: Desired joint configuration. Must be a 1D array with length equal to the number of joint angles.

        Raises:
            InvalidTarget: If the target posture does not have the correct shape.
        """
        target_q = np.atleast_1d(target_q)
        if target_q.ndim != 1 or target_q.shape[0] != self.nq:
            raise InvalidTarget(
                f"Expected target posture to have shape ({self.nq},) "
                f"but got {target_q.shape}"
            )
        self.target_q = target_q.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Set the target posture from the current configuration of the robot.

        Args:
            configuration: Current robot configuration.
        """
        self.set_target(configuration.q)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the error for the posture task.

        The error is defined as:

        .. math::

            e(q) = q^* \ominus q

        where :math:`q^*` is the target posture and :math:`q` is the current posture.

        Args:
            configuration: Current robot configuration.

        Returns:
            Posture task error vector :math:`e(q)`.

        Raises:
            TargetNotSet: If the target posture has not been set.
        """
        if self.target_q is None:
            raise TargetNotSet(self.__class__.__name__)

        # Calculate the difference between the target and current posture
        qvel = np.empty(configuration.nv)
        mujoco.mj_differentiatePos(
            m=configuration.model,
            qvel=qvel,
            dt=1.0,
            qpos1=configuration.q,
            qpos2=self.target_q,
        )

        # Set the error for free joint dimensions to zero
        if self._v_ids is not None:
            qvel[self._v_ids] = 0.0

        return qvel

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the Jacobian for the posture task.

        The task Jacobian is the negative identity matrix :math:`-I_{n_v}`.

        Args:
            configuration: Current robot configuration.

        Returns:
            Posture task Jacobian :math:`J(q)`.

        Raises:
            TargetNotSet: If the target posture has not been set.
        """
        if self.target_q is None:
            raise TargetNotSet(self.__class__.__name__)

        # Initialize the Jacobian as the negative identity matrix
        jac = -np.eye(configuration.nv)

        # Set the Jacobian for free joint dimensions to zero
        if self._v_ids is not None:
            jac[:, self._v_ids] = 0.0

        return jac