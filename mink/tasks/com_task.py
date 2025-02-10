"""Center-of-mass task implementation."""

from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from .exceptions import InvalidTarget, TargetNotSet, TaskDefinitionError
from .task import Task


class ComTask(Task):
    """Regulate the center-of-mass (CoM) position of a robot.

    Attributes:
        target_com: Desired center-of-mass position in the world frame.
    """

    k: int = 3
    target_com: Optional[np.ndarray]

    def __init__(
        self,
        cost: npt.ArrayLike,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        """Initialize the CoM task.

        Args:
            cost: Cost vector for the CoM task. Can be a scalar (identical cost for all
                coordinates) or a vector of shape (3,) for specific costs per coordinate.
            gain: Proportional gain for the task.
            lm_damping: Damping term for the Levenberg-Marquardt algorithm.
        """
        super().__init__(cost=np.zeros((self.k,)), gain=gain, lm_damping=lm_damping)
        self.target_com = None

        self.set_cost(cost)

    def set_cost(self, cost: npt.ArrayLike) -> None:
        """Set a new cost for all CoM coordinates.

        Args:
            cost: Cost vector for the CoM task. Can be a scalar (identical cost for all
                coordinates) or a vector of shape (3,) for specific costs per coordinate.

        Raises:
            TaskDefinitionError: If the cost vector has an invalid shape or contains
                negative values.
        """
        cost = np.atleast_1d(cost)
        if cost.ndim != 1 or cost.shape[0] not in (1, self.k):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} cost must be a vector of shape (1,) "
                f"(aka identical cost for all coordinates) or ({self.k},). Got {cost.shape}"
            )
        if not np.all(cost >= 0.0):
            raise TaskDefinitionError(f"{self.__class__.__name__} cost must be >= 0")
        self.cost[:] = cost

    def set_target(self, target_com: npt.ArrayLike) -> None:
        """Set the target CoM position in the world frame.

        Args:
            target_com: Desired center-of-mass position in the world frame.

        Raises:
            InvalidTarget: If the target CoM position has an invalid shape.
        """
        target_com = np.atleast_1d(target_com)
        if target_com.ndim != 1 or target_com.shape[0] != self.k:
            raise InvalidTarget(
                f"Expected target CoM to have shape ({self.k},) but got {target_com.shape}"
            )
        self.target_com = target_com.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Set the target CoM from a given robot configuration.

        Args:
            configuration: Robot configuration :math:`q`.
        """
        self.set_target(configuration.data.subtree_com[1])

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the CoM task error.

        The error is defined as:

        .. math::

            e(q) = c - c^*

        where :math:`c` is the current CoM position and :math:`c^*` is the target CoM
        position.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Center-of-mass task error vector :math:`e(q)`.

        Raises:
            TargetNotSet: If the target CoM position has not been set.
        """
        if self.target_com is None:
            raise TargetNotSet(self.__class__.__name__)
        return configuration.data.subtree_com[1] - self.target_com

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the CoM task Jacobian.

        The task Jacobian :math:`J(q) \in \mathbb{R}^{3 \times n_v}` is the derivative
        of the CoM position with respect to the current configuration :math:`q`.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Center-of-mass task Jacobian :math:`J(q)`.

        Raises:
            TargetNotSet: If the target CoM position has not been set.
        """
        if self.target_com is None:
            raise TargetNotSet(self.__class__.__name__)
        jac = np.empty((self.k, configuration.nv))
        mujoco.mj_jacSubtreeCom(configuration.model, configuration.data, jac, 1)
        return jac