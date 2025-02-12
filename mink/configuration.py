"""Configuration space of a robot model.\n\nThe :class:`Configuration` class encapsulates a MuJoCo\n`model <https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjmodel>`__\nand `data <https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjdata>`__,\noffering easy access to frame transforms and frame Jacobians. A frame refers to a coordinate\nsystem that can be attached to various parts of the robot, such as a body, geom, or site.\n"""

import logging
from typing import Optional

import mujoco
import numpy as np

from . import constants as consts
from . import exceptions
from .lie import SE3, SO3


class Configuration:
    """Encapsulates a model and data for convenient access to kinematic quantities.\n\n    This class provides methods to access and update the kinematic quantities of a robot\n    model, such as frame transforms and Jacobians. It performs forward kinematics at every\n    time step, ensuring up-to-date information about the robot's state.\n\n    Key functionalities include:\n\n    * Running forward kinematics to update the state.\n    * Checking configuration limits.\n    * Computing Jacobians for different frames.\n    * Retrieving frame transforms relative to the world frame.\n    * Integrating velocities to update configurations.\n    * Handling posture tasks for better control.\n    * Enhancing collision avoidance mechanisms.\n    """

    def __init__(
        self,
        model: mujoco.MjModel,
        q: Optional[np.ndarray] = None,
    ):
        """Constructor.\n\n        Args:\n            model: Mujoco model.\n            q: Configuration to initialize from. If None, the configuration\n                is initialized to the default configuration `qpos0`.\n        """
        self.model = model
        self.data = mujoco.MjData(model)
        self.update(q=q)
        self.posture_task = mink.PostureTask(model=model, cost=5e-2)  # Added posture task for better control

    def update(self, q: Optional[np.ndarray] = None) -> None:
        """Run forward kinematics.\n\n        Args:\n            q: Optional configuration vector to override internal `data.qpos` with.\n        """
        if q is not None:
            self.data.qpos = q
        # The minimal function call required to get updated frame transforms is
        # mj_kinematics. An extra call to mj_comPos is required for updated Jacobians.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def update_from_keyframe(self, key_name: str) -> None:
        """Update the configuration from a keyframe.\n\n        Args:\n            key_name: The name of the keyframe.\n        """
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        if key_id == -1:
            raise exceptions.InvalidKeyframe(key_name, self.model)
        self.update(q=self.model.key_qpos[key_id])
        self.posture_task.set_target_from_configuration(self)  # Update posture task with new configuration

    def check_limits(self, tol: float = 1e-6, safety_break: bool = True) -> None:
        """Check that the current configuration is within bounds.\n\n        Args:\n            tol: Tolerance in [rad].\n            safety_break: If True, stop execution and raise an exception if the current\n                configuration is outside limits. If False, print a warning and continue\n                execution.\n        """
        for jnt in range(self.model.njnt):
            jnt_type = self.model.jnt_type[jnt]
            if (
                jnt_type == mujoco.mjtJoint.mjJNT_FREE
                or not self.model.jnt_limited[jnt]
            ):
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
                    logging.warning(
                        f"Value {qval:.2f} at index {jnt} is outside of its limits: "
                        f"[{qmin:.2f}, {qmax:.2f}]"
                    )

    def get_frame_jacobian(self, frame_name: str, frame_type: str) -> np.ndarray:
        r"""Compute the Jacobian matrix of a frame velocity.\n\n        Denoting our frame by :math:`B` and the world frame by :math:`W`, the\n        Jacobian matrix :math:`{}_B J_{WB}` is related to the body velocity\n        :math:`{}_B v_{WB}` by:\n\n        .. math::\n\n            {}_B v_{WB} = {}_B J_{WB} \dot{q}\n\n        Args:\n            frame_name: Name of the frame in the MJCF.\n            frame_type: Type of frame. Can be a geom, a body or a site.\n\n        Returns:\n            Jacobian :math:`{}_B J_{WB}` of the frame.\n        """
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

        # MuJoCo jacobians have a frame of reference centered at the local frame but
        # aligned with the world frame. To obtain a jacobian expressed in the local
        # frame, aka body jacobian, we need to left-multiply by A[T_fw].
        xmat = getattr(self.data, consts.FRAME_TO_XMAT_ATTR[frame_type])[frame_id]
        R_wf = SO3.from_matrix(xmat.reshape(3, 3))
        A_fw = SE3.from_rotation(R_wf.inverse()).adjoint()
        jac = A_fw @ jac

        return jac

    def get_transform_frame_to_world(self, frame_name: str, frame_type: str) -> SE3:
        """Get the pose of a frame at the current configuration.\n\n        Args:\n            frame_name: Name of the frame in the MJCF.\n            frame_type: Type of frame. Can be a geom, a body or a site.\n\n        Returns:\n            The pose of the frame in the world frame.\n        """
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

    def get_transform(
        self,
        source_name: str,
        source_type: str,
        dest_name: str,
        dest_type: str,
    ) -> SE3:
        """Get the pose of a frame with respect to another frame at the current\n        configuration.\n\n        Args:\n            source_name: Name of the frame in the MJCF.\n            source_type: Source type of frame. Can be a geom, a body or a site.\n            dest_name: Name of the frame to get the pose in.\n            dest_type: Dest type of frame. Can be a geom, a body or a site.\n\n        Returns:\n            The pose of `source_name` in `dest_name`.\n        """
        transform_source_to_world = self.get_transform_frame_to_world(
            source_name, source_type
        )
        transform_dest_to_world = self.get_transform_frame_to_world(
            dest_name, dest_type
        )
        return transform_dest_to_world.inverse() @ transform_source_to_world

    def integrate(self, velocity: np.ndarray, dt: float) -> np.ndarray:
        """Integrate a velocity starting from the current configuration.\n\n        Args:\n            velocity: The velocity in tangent space.\n            dt: Integration duration in [s].\n\n        Returns:\n            The new configuration after integration.\n        """
        q = self.data.qpos.copy()
        mujoco.mj_integratePos(self.model, q, velocity, dt)
        return q

    def integrate_inplace(self, velocity: np.ndarray, dt: float) -> None:
        """Integrate a velocity and update the current configuration inplace.\n\n        Args:\n            velocity: The velocity in tangent space.\n            dt: Integration duration in [s].\n        """
        mujoco.mj_integratePos(self.model, self.data.qpos, velocity, dt)
        self.update()

    def check_collisions(self, tolerance: float = 1e-4, safety_break: bool = True) -> None:
        """Check for potential collisions between bodies in the model.\n\n        Args:\n            tolerance: Tolerance for collision detection in meters.\n            safety_break: If True, stop execution and raise an exception if a collision is detected.\n                If False, print a warning and continue execution.\n        """
        mujoco.mj_forward(self.model, self.data)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.dist < tolerance:
                if safety_break:
                    raise exceptions.CollisionDetected(
                        contact.geom1,
                        contact.geom2,
                        distance=contact.dist,
                        model=self.model,
                    )
                else:
                    logging.warning(
                        f"Collision detected between geom {contact.geom1} and geom {contact.geom2} with distance {contact.dist:.4f}"
                    )

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