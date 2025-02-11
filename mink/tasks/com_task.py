"""Center-of-mass task implementation."""

from typing import Optional

import mujoco
import numpy as np

from ..configuration import Configuration
from .exceptions import InvalidTarget, TargetNotSet, TaskDefinitionError
from .task import Task


class ComTask(Task):
    """Regulate the center-of-mass (CoM) of a robot."""

    k: int = 3
    target_com: Optional[np.ndarray]

    def __init__(
        self,
        cost,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        super().__init__(cost=np.zeros((self.k,)), gain=gain, lm_damping=lm_damping)
        self.target_com = None

        self.set_cost(cost)

    def set_cost(self, cost) -> None:
        """Set a new cost for all CoM coordinates.

        The cost must be a vector of shape (1,) (identical cost for all coordinates)
        or (3,). All cost values must be non-negative.
        """
        cost = np.atleast_1d(cost)
        if cost.ndim != 1 or cost.shape[0] not in (1, self.k):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} cost must be a vector of shape (1,) "
                f"(identical cost for all coordinates) or ({self.k},). "
                f"Got {cost.shape}"
            )
        if not np.all(cost >= 0.0):
            raise TaskDefinitionError(f"{self.__class__.__name__} cost must be >= 0")
        self.cost[:] = cost

    def set_target(self, target_com) -> None:
        """Set the target CoM position in the world frame.

        The target CoM must be a vector of shape (3,).
        """
        target_com = np.atleast_1d(target_com)
        if target_com.ndim != 1 or target_com.shape[0] != self.k:
            raise InvalidTarget(
                f"Expected target CoM to have shape ({self.k},) but got "
                f"{target_com.shape}"
            )
        self.target_com = target_com.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Set the target CoM from a given robot configuration."""
        self.set_target(configuration.data.subtree_com[1])

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        """Compute the CoM task error.

        The error is defined as the difference between the target CoM position and
        the current CoM position.

        Args:
            configuration: Robot configuration.

        Returns:
            Center-of-mass task error vector.
        """
        if self.target_com is None:
            raise TargetNotSet(self.__class__.__name__)
        return configuration.data.subtree_com[1] - self.target_com

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        """Compute the CoM task Jacobian.

        The task Jacobian is the derivative of the CoM position with respect to the
        current configuration.

        Args:
            configuration: Robot configuration.

        Returns:
            Center-of-mass task Jacobian.
        """
        if self.target_com is None:
            raise TargetNotSet(self.__class__.__name__)
        jac = np.empty((self.k, configuration.nv))
        mujoco.mj_jacSubtreeCom(configuration.model, configuration.data, jac, 1)
        return jac