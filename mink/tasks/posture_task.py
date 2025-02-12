"""Posture task implementation."""

from typing import Optional

import mujoco
import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from ..utils import get_freejoint_dims
from .exceptions import InvalidTarget, TargetNotSet, TaskDefinitionError
from .task import Task


class PostureTask(Task):
    """Regulate the robot's joint angles towards a desired posture.\n\n    A posture is a vector of actuated joint angles. Floating-base coordinates are not\n    affected by this task.\n\n    Attributes:\n        target_q: Target joint configuration.\n    """

    target_q: Optional[np.ndarray]

    def __init__(
        self,
        model: mujoco.MjModel,
        cost: float,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        """Initialize the PostureTask.\n\n        Args:\n            model: Mujoco model of the robot.\n            cost: Cost weight for the task.\n            gain: Gain for the task.\n            lm_damping: Damping term for the task.\n\n        Raises:\n            TaskDefinitionError: If the cost is negative.\n        """
        if cost < 0.0:
            raise TaskDefinitionError(f"{self.__class__.__name__} cost must be >= 0")

        super().__init__(
            cost=np.asarray([cost] * model.nv),
            gain=gain,
            lm_damping=lm_damping,
        )
        self.target_q = None

        # Identify the indices of free joints if any.
        _, free_joint_indices = get_freejoint_dims(model)
        self._free_joint_indices = np.asarray(free_joint_indices) if free_joint_indices else None

        self.k = model.nv
        self.nq = model.nq

    def set_target(self, target_q: npt.ArrayLike) -> None:
        """Set the target joint configuration.\n\n        Args:\n            target_q: Desired joint configuration.\n\n        Raises:\n            InvalidTarget: If the target configuration has an incorrect shape.\n        """
        target_q = np.atleast_1d(target_q)
        if target_q.ndim != 1 or target_q.shape[0] != self.nq:
            raise InvalidTarget(
                f"Expected target posture to have shape ({self.nq},) but got "
                f"{target_q.shape}"
            )
        self.target_q = target_q.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Set the target posture from the current configuration.\n\n        Args:\n            configuration: Current robot configuration.\n        """
        self.set_target(configuration.q)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        """Compute the posture task error.\n\n        The error is calculated as the difference between the target and current configuration.\n\n        Args:\n            configuration: Current robot configuration.\n\n        Returns:\n            Posture task error vector.\n\n        Raises:\n            TargetNotSet: If the target configuration is not set.\n        """
        if self.target_q is None:
            raise TargetNotSet(self.__class__.__name__)

        # Calculate the difference between the target and current configuration.
        qvel = np.empty(configuration.nv)
        mujoco.mj_differentiatePos(
            m=configuration.model,
            qvel=qvel,
            dt=1.0,
            qpos1=configuration.q,
            qpos2=self.target_q,
        )

        # Set the error for free joints to zero.
        if self._free_joint_indices is not None:
            qvel[self._free_joint_indices] = 0.0

        return qvel

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        """Compute the posture task Jacobian.\n\n        The Jacobian for the posture task is the negative identity matrix.\n\n        Args:\n            configuration: Current robot configuration.\n\n        Returns:\n            Posture task Jacobian.\n\n        Raises:\n            TargetNotSet: If the target configuration is not set.\n        """
        if self.target_q is None:
            raise TargetNotSet(self.__class__.__name__)

        # Initialize the Jacobian as the negative identity matrix.
        jac = -np.eye(configuration.nv)

        # Set the Jacobian for free joints to zero.
        if self._free_joint_indices is not None:
            jac[:, self._free_joint_indices] = 0.0

        return jac