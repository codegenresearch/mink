"""Configuration space of a robot model.\n\nThe :class:`Configuration` class encapsulates a MuJoCo model and its associated data,\nenabling easy access to kinematic quantities such as frame transforms and Jacobians. It\nautomatically performs forward kinematics at each time step, ensuring that all\nkinematic queries return up-to-date information.\n\nIn this context, a frame refers to a coordinate system that can be attached to\ndifferent elements of the robot model. Currently supported frames include\n`body`, `geom`, and `site`.\n\nKey functionalities include:\n\n- Running forward kinematics to update the state.\n- Checking configuration limits.\n- Computing Jacobians for different frames.\n- Retrieving frame transforms relative to the world frame.\n- Integrating velocities to update configurations.\n"""

from typing import Optional

import mujoco
import numpy as np

from . import constants as consts
from . import exceptions
from .lie import SE3, SO3


class Configuration:
    """Encapsulates a model and data for convenient access to kinematic quantities.\n\n    This class provides convenient methods to access and update the kinematic quantities\n    of a robot model, such as frame transforms and Jacobians. It ensures that forward\n    kinematics is computed at each time step, allowing the user to query up-to-date\n    information about the robot's state.\n    """

    def __init__(
        self,
        model: mujoco.MjModel,
        initial_configuration: Optional[np.ndarray] = None,
    ):
        """Initialize the configuration.\n\n        Args:\n            model: Mujoco model.\n            initial_configuration: Configuration to initialize from. If None, the\n                configuration is initialized to the reference configuration `qpos0`.\n        """
        self.model = model
        self.data = mujoco.MjData(model)
        self.update(initial_configuration)

    def update(self, configuration: Optional[np.ndarray] = None) -> None:
        """Run forward kinematics.\n\n        Args:\n            configuration: Optional configuration vector to override internal data.qpos with.\n        """
        if configuration is not None:
            self.data.qpos = configuration
        # Perform forward kinematics and update Jacobians.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def update_from_keyframe(self, keyframe_name: str) -> None:
        """Update the configuration from a keyframe.\n\n        Args:\n            keyframe_name: The name of the keyframe.\n\n        Raises:\n            exceptions.InvalidKeyframe: If no keyframe named `keyframe_name` is found in the model.\n        """
        keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
        if keyframe_id == -1:
            raise exceptions.InvalidKeyframe(keyframe_name, self.model)
        self.update(self.model.key_qpos[keyframe_id])

    def check_limits(self, tolerance: float = 1e-6, raise_on_violation: bool = True) -> None:
        """Check that the current configuration is within joint limits.\n\n        Args:\n            tolerance: Tolerance in radians.\n            raise_on_violation: If True, raise an exception if the configuration is out of bounds.\n                If False, print a warning and continue execution.\n\n        Raises:\n            exceptions.NotWithinConfigurationLimits: If the configuration is out of bounds and\n                `raise_on_violation` is True.\n        """
        for joint_id in range(self.model.njnt):
            if (
                self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
                or not self.model.jnt_limited[joint_id]
            ):
                continue
            joint_position = self.q[self.model.jnt_qposadr[joint_id]]
            joint_lower_limit = self.model.jnt_range[joint_id, 0]
            joint_upper_limit = self.model.jnt_range[joint_id, 1]
            if joint_position < joint_lower_limit - tolerance or joint_position > joint_upper_limit + tolerance:
                if raise_on_violation:
                    raise exceptions.NotWithinConfigurationLimits(
                        joint_id=joint_id,
                        value=joint_position,
                        lower=joint_lower_limit,
                        upper=joint_upper_limit,
                        model=self.model,
                    )
                else:
                    print(
                        f"Value {joint_position} at joint index {joint_id} is out of limits: "
                        f"[{joint_lower_limit}, {joint_upper_limit}]"
                    )

    def get_frame_jacobian(self, frame_name: str, frame_type: str) -> np.ndarray:
        """Compute the Jacobian matrix of a frame velocity.\n\n        Denoting our frame by :math:`B` and the world frame by :math:`W`, the\n        Jacobian matrix :math:`{}_B J_{WB}` is related to the body velocity\n        :math:`{}_B v_{WB}` by:\n\n        .. math::\n\n            {}_B v_{WB} = {}_B J_{WB} \dot{q}\n\n        Args:\n            frame_name: Name of the frame in the MJCF.\n            frame_type: Type of frame. Can be 'geom', 'body', or 'site'.\n\n        Returns:\n            Jacobian :math:`{}_B J_{WB}` of the frame.\n\n        Raises:\n            exceptions.UnsupportedFrame: If the frame type is not supported.\n            exceptions.InvalidFrame: If the frame name is not found in the model.\n        """
        if frame_type not in consts.SUPPORTED_FRAMES:
            raise exceptions.UnsupportedFrame(frame_type, supported_frames=consts.SUPPORTED_FRAMES)

        frame_id = mujoco.mj_name2id(
            self.model, consts.FRAME_TO_ENUM[frame_type], frame_name
        )
        if frame_id == -1:
            raise exceptions.InvalidFrame(
                frame_name=frame_name,
                frame_type=frame_type,
                model=self.model,
            )

        jacobian = np.empty((6, self.model.nv))
        jacobian_function = consts.FRAME_TO_JAC_FUNC[frame_type]
        jacobian_function(self.model, self.data, jacobian[:3], jacobian[3:], frame_id)

        # Transform the Jacobian to the body frame.
        xmat = getattr(self.data, consts.FRAME_TO_XMAT_ATTR[frame_type])[frame_id]
        rotation_world_to_frame = SO3.from_matrix(xmat.reshape(3, 3))
        adjoint_transform = SE3.from_rotation(rotation_world_to_frame.inverse()).adjoint()
        jacobian = adjoint_transform @ jacobian

        return jacobian

    def get_transform_frame_to_world(self, frame_name: str, frame_type: str) -> SE3:
        """Get the pose of a frame at the current configuration.\n\n        Args:\n            frame_name: Name of the frame in the MJCF.\n            frame_type: Type of frame. Can be 'geom', 'body', or 'site'.\n\n        Returns:\n            The pose of the frame in the world frame.\n\n        Raises:\n            exceptions.UnsupportedFrame: If the frame type is not supported.\n            exceptions.InvalidFrame: If the frame name is not found in the model.\n        """
        if frame_type not in consts.SUPPORTED_FRAMES:
            raise exceptions.UnsupportedFrame(frame_type, supported_frames=consts.SUPPORTED_FRAMES)

        frame_id = mujoco.mj_name2id(
            self.model, consts.FRAME_TO_ENUM[frame_type], frame_name
        )
        if frame_id == -1:
            raise exceptions.InvalidFrame(
                frame_name=frame_name,
                frame_type=frame_type,
                model=self.model,
            )

        position = getattr(self.data, consts.FRAME_TO_POS_ATTR[frame_type])[frame_id]
        rotation_matrix = getattr(self.data, consts.FRAME_TO_XMAT_ATTR[frame_type])[frame_id]
        return SE3.from_rotation_and_translation(
            rotation=SO3.from_matrix(rotation_matrix.reshape(3, 3)),
            translation=position,
        )

    def integrate(self, velocity: np.ndarray, time_step: float) -> np.ndarray:
        """Integrate a velocity starting from the current configuration.\n\n        Args:\n            velocity: The velocity in tangent space.\n            time_step: Integration duration in seconds.\n\n        Returns:\n            The new configuration after integration.\n        """
        new_configuration = self.data.qpos.copy()
        mujoco.mj_integratePos(self.model, new_configuration, velocity, time_step)
        return new_configuration

    def integrate_inplace(self, velocity: np.ndarray, time_step: float) -> None:
        """Integrate a velocity and update the current configuration in place.\n\n        Args:\n            velocity: The velocity in tangent space.\n            time_step: Integration duration in seconds.\n        """
        mujoco.mj_integratePos(self.model, self.data.qpos, velocity, time_step)
        self.update()

    # Aliases.

    @property
    def q(self) -> np.ndarray:
        """The current configuration vector."""
        return self.data.qpos.copy()

    @property
    def nv(self) -> int:
        """The dimension of the tangent space."""
        return self.model.nv

    @property
    def nq(self) -> int:
        """The dimension of the configuration space."""
        return self.model.nq