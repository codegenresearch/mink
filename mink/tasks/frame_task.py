"""Frame task implementation."""

from typing import Optional

import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from ..lie import SE3
from .exceptions import TargetNotSet, TaskDefinitionError
from .task import Task


class FrameTask(Task):
    """Regulate the pose of a robot frame in the world frame.\n\n    This task ensures that a specified frame of the robot aligns with a desired pose in the world frame.\n    The pose is defined by a transformation matrix (SE3).\n\n    Attributes:\n        frame_name (str): Name of the frame to regulate.\n        frame_type (str): The type of the frame, which can be 'body', 'geom', or 'site'.\n        transform_target_to_world (Optional[SE3]): Target pose of the frame in the world frame.\n        position_cost (np.ndarray): Cost associated with position error.\n        orientation_cost (np.ndarray): Cost associated with orientation error.\n    """

    k: int = 6
    transform_target_to_world: Optional[SE3]
    position_cost: np.ndarray
    orientation_cost: np.ndarray

    def __init__(
        self,
        frame_name: str,
        frame_type: str,
        position_cost: npt.ArrayLike,
        orientation_cost: npt.ArrayLike,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ) -> None:
        """\n        Initialize the FrameTask with the specified parameters.\n\n        Args:\n            frame_name (str): Name of the frame to regulate.\n            frame_type (str): Type of the frame ('body', 'geom', or 'site').\n            position_cost (npt.ArrayLike): Cost for position error, can be a scalar or a vector of size 3.\n            orientation_cost (npt.ArrayLike): Cost for orientation error, can be a scalar or a vector of size 3.\n            gain (float, optional): Gain for the task. Defaults to 1.0.\n            lm_damping (float, optional): Damping factor for Levenberg-Marquardt. Defaults to 0.0.\n        """
        super().__init__(cost=np.zeros((self.k,)), gain=gain, lm_damping=lm_damping)
        self.frame_name = frame_name
        self.frame_type = frame_type
        self.transform_target_to_world = None
        self.set_position_cost(position_cost)
        self.set_orientation_cost(orientation_cost)

    def set_position_cost(self, position_cost: npt.ArrayLike) -> None:
        """\n        Set the cost for position error.\n\n        Args:\n            position_cost (npt.ArrayLike): Cost for position error, can be a scalar or a vector of size 3.\n\n        Raises:\n            TaskDefinitionError: If the cost is not a valid vector or contains negative values.\n        """
        position_cost = np.atleast_1d(position_cost)
        if position_cost.ndim != 1 or position_cost.shape[0] not in (1, 3):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} position cost should be a vector of shape "
                "1 (identical cost for all coordinates) or (3,) but got "
                f"{position_cost.shape}"
            )
        if not np.all(position_cost >= 0.0):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} position cost should be >= 0"
            )
        self.cost[:3] = position_cost

    def set_orientation_cost(self, orientation_cost: npt.ArrayLike) -> None:
        """\n        Set the cost for orientation error.\n\n        Args:\n            orientation_cost (npt.ArrayLike): Cost for orientation error, can be a scalar or a vector of size 3.\n\n        Raises:\n            TaskDefinitionError: If the cost is not a valid vector or contains negative values.\n        """
        orientation_cost = np.atleast_1d(orientation_cost)
        if orientation_cost.ndim != 1 or orientation_cost.shape[0] not in (1, 3):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} orientation cost should be a vector of "
                "shape 1 (identical cost for all coordinates) or (3,) but got "
                f"{orientation_cost.shape}"
            )
        if not np.all(orientation_cost >= 0.0):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} orientation cost should be >= 0"
            )
        self.cost[3:] = orientation_cost

    def set_target(self, transform_target_to_world: SE3) -> None:
        """\n        Set the target pose in the world frame.\n\n        Args:\n            transform_target_to_world (SE3): Transform from the task target frame to the world frame.\n        """
        self.transform_target_to_world = transform_target_to_world.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """\n        Set the target pose from a given robot configuration.\n\n        Args:\n            configuration (Configuration): Robot configuration.\n        """
        self.set_target(
            configuration.get_transform_frame_to_world(self.frame_name, self.frame_type)
        )

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""\n        Compute the frame task error.\n\n        The error is a twist :math:`e(q) \in se(3)` expressed in the local frame, i.e., it is a body twist.\n        It is computed by taking the right-minus difference between the target pose :math:`T_{0t}` and current frame pose :math:`T_{0b}`:\n\n        .. math::\n\n            e(q) := {}_b \xi_{0b} = -(T_{t0} \ominus T_{b0})\n            = -\log(T_{t0} \cdot T_{0b}) = -\log(T_{tb}) = \log(T_{bt})\n\n        where :math:`b` denotes our frame, :math:`t` the target frame, and :math:`0` the inertial frame.\n\n        Args:\n            configuration (Configuration): Robot configuration.\n\n        Returns:\n            np.ndarray: Frame task error vector :math:`e(q)`.\n\n        Raises:\n            TargetNotSet: If the target pose has not been set.\n        """
        if self.transform_target_to_world is None:
            raise TargetNotSet(self.__class__.__name__)

        transform_frame_to_world = configuration.get_transform_frame_to_world(
            self.frame_name, self.frame_type
        )
        return self.transform_target_to_world.minus(transform_frame_to_world)

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""\n        Compute the frame task Jacobian.\n\n        Args:\n            configuration (Configuration): Robot configuration.\n\n        Returns:\n            np.ndarray: Frame task Jacobian :math:`J(q)`.\n\n        Raises:\n            TargetNotSet: If the target pose has not been set.\n        """
        if self.transform_target_to_world is None:
            raise TargetNotSet(self.__class__.__name__)

        jac = configuration.get_frame_jacobian(self.frame_name, self.frame_type)

        transform_frame_to_world = configuration.get_transform_frame_to_world(
            self.frame_name, self.frame_type
        )

        T_tb = self.transform_target_to_world.inverse() @ transform_frame_to_world
        return -T_tb.jlog() @ jac