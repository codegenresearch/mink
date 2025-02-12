"""Configuration space of a robot model.\n\nThe :class:`Configuration` class bundles a MuJoCo `model <https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjmodel>`__\nand `data <https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjdata>`__,\nand enables easy access to kinematic quantities such as frame transforms and frame\nJacobians.\n\nFrames are coordinate systems that can be attached to different elements of\nthe robot model. mink supports frames of type `body`, `geom` and `site`.\n\nVersion: dynamically loaded from configuration.\n"""

from typing import Optional

import mujoco
import numpy as np

from . import constants as consts
from . import exceptions
from .lie import SE3, SO3


class Configuration:
    """Encapsulates a model and data for convenient access to kinematic quantities.\n\n    This class provides convenient methods to access and update the kinematic quantities\n    of a robot model, such as frame transforms and Jacobians. It ensures that forward\n    kinematics is computed at each time step, allowing the user to query up-to-date\n    information about the robot's state.\n\n    In this context, a frame refers to a coordinate system that can be attached to\n    different elements of the robot model. Currently supported frames include\n    `body`, `geom` and `site`.\n\n    Key functionalities include:\n\n        - Running forward kinematics to update the state.\n        - Checking configuration limits.\n        - Computing Jacobians for different frames.\n        - Retrieving frame transforms relative to the world frame.\n        - Integrating velocities to update configurations.\n    """

    def __init__(
        self,
        model: mujoco.MjModel,
        q: Optional[np.ndarray] = None,
    ):
        """Constructor.\n\n        Args:\n            model: Mujoco model.\n            q: Configuration to initialize from. If None, the configuration\n                is initialized to the reference configuration `qpos0`.\n        """
        self.model = model
        self.data = mujoco.MjData(model)
        self.update(q=q)

    def update(self, q: Optional[np.ndarray] = None) -> None:
        """Run forward kinematics.\n\n        Args:\n            q: Optional configuration vector to override internal data.qpos with.\n        """
        if q is not None:
            self.data.qpos = q
        # The minimal function call required to get updated frame transforms is
        # mj_kinematics. An extra call to mj_comPos is required for updated Jacobians.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def update_from_keyframe(self, key_name: str) -> None:
        """Update the configuration from a keyframe.\n\n        Args:\n            key_name: The name of the keyframe.\n\n        Raises:\n            ValueError: if no key named `key` was found in the model.\n        """
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        if key_id == -1:
            raise exceptions.InvalidKeyframe(key_name, self.model)
        self.update(q=self.model.key_qpos[key_id])

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
                    print(
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

    def integrate(self, velocity: np.ndarray, dt: float) -> np.ndarray:
        """Integrate a velocity starting from the current configuration.\n\n        Args:\n            velocity: The velocity in tangent space.\n            dt: Integration duration in [s].\n\n        Returns:\n            The new configuration after integration.\n        """
        q = self.data.qpos.copy()
        mujoco.mj_integratePos(self.model, q, velocity, dt)
        return q

    def integrate_inplace(self, velocity: np.ndarray, dt: float) -> None:
        """Integrate a velocity and update the current configuration inplace.\n\n        Args:\n            velocity: The velocity in tangent space.\n            dt: Integration duration in [s].\n        """
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