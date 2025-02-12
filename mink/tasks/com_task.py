"""Center-of-mass (CoM) task implementation."""

from typing import Optional

import mujoco
import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from .exceptions import InvalidTarget, TargetNotSet, TaskDefinitionError
from .task import Task


class CoMTask(Task):
    """Regulate the center-of-mass (CoM) position of a robot.\n\n    Attributes:\n        target_com: Target position of the CoM in the world frame.\n    """

    COM_DIMENSION: int = 3
    target_com: Optional[np.ndarray]

    def __init__(
        self,
        cost: npt.ArrayLike,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        """Initialize the CoM task.\n\n        Args:\n            cost: Cost associated with each CoM coordinate. Can be a scalar or a vector\n                of length 3.\n            gain: Proportional gain for the task.\n            lm_damping: Levenberg-Marquardt damping factor.\n        """
        super().__init__(cost=np.zeros((self.COM_DIMENSION,)), gain=gain, lm_damping=lm_damping)
        self.target_com = None

        self.set_cost(cost)

    def set_cost(self, cost: npt.ArrayLike) -> None:
        """Set the cost for the CoM task.\n\n        Args:\n            cost: Cost associated with each CoM coordinate. Can be a scalar or a vector\n                of length 3.\n\n        Raises:\n            TaskDefinitionError: If the cost is not a scalar or a vector of length 3, or\n                if any cost value is negative.\n        """
        cost = np.atleast_1d(cost)
        if cost.ndim != 1 or cost.shape[0] not in (1, self.COM_DIMENSION):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} cost must be a scalar or a vector of shape "
                f"({self.COM_DIMENSION},). Got {cost.shape}"
            )
        if not np.all(cost >= 0.0):
            raise TaskDefinitionError(f"{self.__class__.__name__} cost must be non-negative")
        self.cost[:] = cost

    def set_target(self, target_com: npt.ArrayLike) -> None:
        """Set the target CoM position in the world frame.\n\n        Args:\n            target_com: Desired center-of-mass position in the world frame.\n\n        Raises:\n            InvalidTarget: If the target CoM position does not have the correct shape.\n        """
        target_com = np.atleast_1d(target_com)
        if target_com.ndim != 1 or target_com.shape[0] != self.COM_DIMENSION:
            raise InvalidTarget(
                f"Expected target CoM to have shape ({self.COM_DIMENSION},) but got "
                f"{target_com.shape}"
            )
        self.target_com = target_com.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Set the target CoM position from a given robot configuration.\n\n        Args:\n            configuration: Robot configuration.\n\n        Raises:\n            ValueError: If the configuration does not contain a valid CoM position.\n        """
        try:
            self.set_target(configuration.data.subtree_com[1])
        except IndexError:
            raise ValueError("Configuration does not contain a valid CoM position.")

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the CoM task error.\n\n        The error is defined as:\n\n        .. math::\n\n            e(q) = c^* - c\n\n        where :math:`c^*` is the target CoM position and :math:`c` is the current CoM\n        position.\n\n        Args:\n            configuration: Robot configuration.\n\n        Returns:\n            Center-of-mass task error vector :math:`e(q)`.\n\n        Raises:\n            TargetNotSet: If the target CoM position has not been set.\n        """
        if self.target_com is None:
            raise TargetNotSet(self.__class__.__name__)
        return configuration.data.subtree_com[1] - self.target_com

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the CoM task Jacobian.\n\n        The task Jacobian :math:`J(q) \in \mathbb{R}^{3 \times n_v}` is the\n        derivative of the CoM position with respect to the current configuration\n        :math:`q`.\n\n        Args:\n            configuration: Robot configuration.\n\n        Returns:\n            Center-of-mass task Jacobian :math:`J(q)`.\n\n        Raises:\n            TargetNotSet: If the target CoM position has not been set.\n        """
        if self.target_com is None:
            raise TargetNotSet(self.__class__.__name__)
        jacobian = np.empty((self.COM_DIMENSION, configuration.nv))
        mujoco.mj_jacSubtreeCom(configuration.model, configuration.data, jacobian, 1)
        return jacobian