"""Configuration space of a robot model.

The Configuration class encapsulates a MuJoCo model and its associated data,
enabling easy access to kinematic quantities such as frame transforms and Jacobians.
It automatically performs forward kinematics at each time step, ensuring that all
kinematic queries return up-to-date information.

In this context, a frame refers to a coordinate system that can be attached to
different elements of the robot model. Currently supported frames include
`body`, `geom`, and `site`.

Key functionalities include:

    - Running forward kinematics to update the state.
    - Checking configuration limits.
    - Computing Jacobians for different frames.
    - Retrieving frame transforms relative to the world frame.
    - Integrating velocities to update configurations.
"""

from typing import Optional

import mujoco
import numpy as np

from . import constants as consts
from . import exceptions
from .lie import SE3, SO3


class Configuration:
    """Encapsulates a model and data for convenient access to kinematic quantities.

    This class provides convenient methods to access and update the kinematic quantities
    of a robot model, such as frame transforms and Jacobians. It ensures that forward
    kinematics is computed at each time step, allowing the user to query up-to-date
    information about the robot's state.

    Key functionalities include:

        - Running forward kinematics to update the state.
        - Checking configuration limits.
        - Computing Jacobians for different frames.
        - Retrieving frame transforms relative to the world frame.
        - Integrating velocities to update configurations.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        q: Optional[np.ndarray] = None,
    ):
        """Initialize the Configuration with a model and an optional initial configuration.

        Args:
            model: Mujoco model.
            q: Configuration to initialize from. If None, the configuration
                is initialized to the reference configuration `qpos0`.
        """
        self.model = model
        self.data = mujoco.MjData(model)
        self.update(q=q)

    def update(self, q: Optional[np.ndarray] = None) -> None:
        """Run forward kinematics to update the state.

        Args:
            q: Optional configuration vector to override the internal data.qpos.
        """
        if q is not None:
            self.data.qpos = q
        # Perform forward kinematics to update frame transforms and Jacobians.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def update_from_keyframe(self, key_name: str) -> None:
        """Update the configuration from a specified keyframe.

        Args:
            key_name: The name of the keyframe.

        Raises:
            InvalidKeyframe: If no keyframe with the specified name is found in the model.
        """
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        if key_id == -1:
            raise exceptions.InvalidKeyframe(key_name, self.model)
        self.update(q=self.model.key_qpos[key_id])

    def check_limits(self, tol: float = 1e-6, safety_break: bool = True) -> None:
        """Check if the current configuration is within the specified joint limits.

        Args:
            tol: Tolerance in radians for checking joint limits.
            safety_break: If True, raise an exception if the configuration is out of limits.
                If False, print a warning and continue execution.

        Raises:
            NotWithinConfigurationLimits: If the configuration is out of limits and safety_break is True.
        """
        for jnt in range(self.model.njnt):
            jnt_type = self.model.jnt_type[jnt]
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE or not self.model.jnt_limited[jnt]:
                continue
            padr = self.model.jnt_qposadr[jnt]
            qval = self.q[padr]
            qmin = self.model.jnt_range[jnt, 0]
            qmax = self.model.jnt_range[jnt, 1]
            if qval < qmin - tol or qval > qmax + tol:
                if safety_break:
                    raise exceptions.NotWithinConfigurationLimits(
                        joint_id=jnt,
                        value=qval,
                        lower=qmin,
                        upper=qmax,
                        model=self.model,
                    )
                else:
                    print(
                        f"Joint value {qval} at index {jnt} is out of limits: "
                        f"[{qmin}, {qmax}]"
                    )

    def get_frame_jacobian(self, frame_name: str, frame_type: str) -> np.ndarray:
        """Compute the Jacobian matrix of a frame velocity relative to the world frame.

        Args:
            frame_name: Name of the frame in the MJCF.
            frame_type: Type of frame. Can be 'geom', 'body', or 'site'.

        Returns:
            Jacobian matrix of the frame velocity relative to the world frame.

        Raises:
            UnsupportedFrame: If the frame type is not supported.
            InvalidFrame: If the frame name is not found in the model.
        """
        if frame_type not in consts.SUPPORTED_FRAMES:
            raise exceptions.UnsupportedFrame(frame_type, consts.SUPPORTED_FRAMES)

        frame_id = mujoco.mj_name2id(
            self.model, consts.FRAME_TO_ENUM[frame_type], frame_name
        )
        if frame_id == -1:
            raise exceptions.InvalidFrame(
                frame_name=frame_name,
                frame_type=frame_type,
                model=self.model,
            )

        jac = np.empty((6, self.model.nv))
        jac_func = consts.FRAME_TO_JAC_FUNC[frame_type]
        jac_func(self.model, self.data, jac[:3], jac[3:], frame_id)

        # Convert the Jacobian from world frame to the local frame.
        xmat = getattr(self.data, consts.FRAME_TO_XMAT_ATTR[frame_type])[frame_id]
        R_wf = SO3.from_matrix(xmat.reshape(3, 3))
        A_fw = SE3.from_rotation(R_wf.inverse()).adjoint()
        jac = A_fw @ jac

        return jac

    def get_transform_frame_to_world(self, frame_name: str, frame_type: str) -> SE3:
        """Get the pose of a frame relative to the world frame at the current configuration.

        Args:
            frame_name: Name of the frame in the MJCF.
            frame_type: Type of frame. Can be 'geom', 'body', or 'site'.

        Returns:
            Pose of the frame relative to the world frame.

        Raises:
            UnsupportedFrame: If the frame type is not supported.
            InvalidFrame: If the frame name is not found in the model.
        """
        if frame_type not in consts.SUPPORTED_FRAMES:
            raise exceptions.UnsupportedFrame(frame_type, consts.SUPPORTED_FRAMES)

        frame_id = mujoco.mj_name2id(
            self.model, consts.FRAME_TO_ENUM[frame_type], frame_name
        )
        if frame_id == -1:
            raise exceptions.InvalidFrame(
                frame_name=frame_name,
                frame_type=frame_type,
                model=self.model,
            )

        xpos = getattr(self.data, consts.FRAME_TO_POS_ATTR[frame_type])[frame_id]
        xmat = getattr(self.data, consts.FRAME_TO_XMAT_ATTR[frame_type])[frame_id]
        return SE3.from_rotation_and_translation(
            rotation=SO3.from_matrix(xmat.reshape(3, 3)),
            translation=xpos,
        )

    def integrate(self, velocity: np.ndarray, dt: float) -> np.ndarray:
        """Integrate a velocity starting from the current configuration.

        Args:
            velocity: The velocity in tangent space.
            dt: Integration duration in seconds.

        Returns:
            The new configuration after integration.
        """
        q = self.data.qpos.copy()
        mujoco.mj_integratePos(self.model, q, velocity, dt)
        return q

    def integrate_inplace(self, velocity: np.ndarray, dt: float) -> None:
        """Integrate a velocity and update the current configuration in place.

        Args:
            velocity: The velocity in tangent space.
            dt: Integration duration in seconds.
        """
        mujoco.mj_integratePos(self.model, self.data.qpos, velocity, dt)
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