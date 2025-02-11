"""Posture task implementation."""

from __future__ import annotations

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
        target_q: Target configuration for the robot's joints.
        _v_ids: Indices of free joint dimensions.
    """

    target_q: Optional[np.ndarray]
    _v_ids: Optional[np.ndarray]

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

        # Identify the indices of free joint dimensions
        _, v_ids_or_none = get_freejoint_dims(model)
        self._v_ids = np.asarray(v_ids_or_none) if v_ids_or_none else None

        self.k = model.nv
        self.nq = model.nq

    def set_target(self, target_q: npt.ArrayLike) -> None:
        """Set the target posture.

        Args:
            target_q: Desired joint configuration.

        Raises:
            InvalidTarget: If the target posture does not match the expected shape.
        """
        target_q = np.atleast_1d(target_q)
        if target_q.ndim != 1 or target_q.shape[0] != self.nq:
            raise InvalidTarget(
                f"Expected target posture to have shape ({self.nq},) but got {target_q.shape}"
            )
        self.target_q = target_q.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Set the target posture from the current configuration of the robot.

        Args:
            configuration: Current configuration of the robot.
        """
        self.set_target(configuration.q)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the posture task error.

        The error is defined as:

        .. math::

            e(q) = q^* \ominus q

        Args:
            configuration: Current configuration of the robot.

        Returns:
            Posture task error vector :math:`e(q)`.

        Raises:
            TargetNotSet: If the target posture has not been set.
        """
        if self.target_q is None:
            raise TargetNotSet(self.__class__.__name__)

        # Calculate the difference in joint positions
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
        r"""Compute the posture task Jacobian.

        The task Jacobian is the negative identity :math:`I_{n_v}`.

        Args:
            configuration: Current configuration of the robot.

        Returns:
            Posture task jacobian :math:`J(q)`.

        Raises:
            TargetNotSet: If the target posture has not been set.
        """
        if self.target_q is None:
            raise TargetNotSet(self.__class__.__name__)

        # Initialize the Jacobian as the negative identity matrix
        jac = -np.eye(configuration.nv)

        # Set the Jacobian entries for free joint dimensions to zero
        if self._v_ids is not None:
            jac[:, self._v_ids] = 0.0

        return jac