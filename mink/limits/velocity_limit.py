"""Joint velocity limit with posture task and enhanced collision avoidance."""

from typing import Mapping

import mujoco
import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from ..constants import dof_width
from .exceptions import LimitDefinitionError
from .limit import Constraint, Limit


class VelocityLimit(Limit):
    """Inequality constraint on joint velocities in a robot model.\n\n    Floating base joints are ignored. This class also considers a posture task and\n    enhances collision avoidance handling.\n\n    Attributes:\n        indices: Tangent indices corresponding to velocity-limited joints.\n        limit: Maximum allowed velocity magnitude for velocity-limited joints, in\n            [m]/[s] for slide joints and [rad]/[s] for hinge joints.\n        projection_matrix: Projection from tangent space to subspace with\n            velocity-limited joints.\n    """

    indices: np.ndarray
    limit: np.ndarray
    projection_matrix: np.ndarray

    def __init__(
        self,
        model: mujoco.MjModel,
        velocities: Mapping[str, npt.ArrayLike] = {},
        posture_gain: float = 0.1,
        collision_avoidance_gain: float = 0.5,
    ):
        """Initialize velocity limits.\n\n        Args:\n            model: MuJoCo model.\n            velocities: Dictionary mapping joint name to maximum allowed magnitude in\n                [m]/[s] for slide joints and [rad]/[s] for hinge joints.\n            posture_gain: Gain factor for posture task.\n            collision_avoidance_gain: Gain factor for collision avoidance handling.\n        """
        limit_list: list[float] = []
        index_list: list[int] = []
        for joint_name, max_vel in velocities.items():
            jid = model.joint(joint_name).id
            jnt_type = model.jnt_type[jid]
            jnt_dim = dof_width(jnt_type)
            jnt_id = model.jnt_dofadr[jid]
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                raise LimitDefinitionError(f"Free joint {joint_name} is not supported")
            max_vel = np.atleast_1d(max_vel)
            if max_vel.shape != (jnt_dim,):
                raise LimitDefinitionError(
                    f"Joint {joint_name} must have a limit of shape ({jnt_dim},). "
                    f"Got: {max_vel.shape}"
                )
            index_list.extend(range(jnt_id, jnt_id + jnt_dim))
            limit_list.extend(max_vel.tolist())

        self.indices = np.array(index_list)
        self.indices.setflags(write=False)
        self.limit = np.array(limit_list)
        self.limit.setflags(write=False)
        self.posture_gain = posture_gain
        self.collision_avoidance_gain = collision_avoidance_gain

        dim = len(self.indices)
        self.projection_matrix = np.eye(model.nv)[self.indices] if dim > 0 else None

    def compute_qp_inequalities(
        self, configuration: Configuration, dt: float
    ) -> Constraint:
        r"""Compute the configuration-dependent joint velocity limits.\n\n        The limits are defined as:\n\n        .. math::\n\n            -v_{\text{max}} \cdot dt \leq \Delta q \leq v_{\text{max}} \cdot dt\n\n        where :math:`v_{max} \in {\cal T}` is the robot's velocity limit\n        vector and :math:`\Delta q \in T_q({\cal C})` is the displacement in the\n        tangent space at :math:`q`. See the :ref:`derivations` section for\n        more information.\n\n        Args:\n            configuration: Robot configuration :math:`q`.\n            dt: Integration timestep in [s].\n\n        Returns:\n            Pair :math:`(G, h)` representing the inequality constraint as\n            :math:`G \Delta q \leq h`, or ``None`` if there is no limit.\n        """
        if self.projection_matrix is None:
            return Constraint()

        # Velocity limits
        G_velocity = np.vstack([self.projection_matrix, -self.projection_matrix])
        h_velocity = np.hstack([dt * self.limit, dt * self.limit])

        # Posture task
        q_desired = configuration.q_desired  # Assuming Configuration has q_desired attribute
        posture_error = q_desired - configuration.q
        G_posture = self.projection_matrix
        h_posture = self.posture_gain * posture_error

        # Collision avoidance
        collision_vector = configuration.collision_vector  # Assuming Configuration has collision_vector attribute
        G_collision = -self.projection_matrix
        h_collision = -self.collision_avoidance_gain * collision_vector

        # Combine all constraints
        G = np.vstack([G_velocity, G_posture, G_collision])
        h = np.hstack([h_velocity, h_posture, h_collision])

        return Constraint(G=G, h=h)