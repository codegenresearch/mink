"""This module defines an inequality constraint on joint positions in a robot model, ignoring floating base joints."""

import mujoco
import numpy as np

from ..configuration import Configuration
from ..constants import qpos_width
from .exceptions import LimitDefinitionError
from .limit import Constraint, Limit


class ConfigurationLimit(Limit):
    """Inequality constraint on joint positions in a robot model, ignoring floating base joints.

    Attributes:
        indices: Indices of the degrees of freedom that are limited.
        lower: Lower position limits for the limited joints.
        upper: Upper position limits for the limited joints.
        projection_matrix: Projection matrix from the full tangent space to the subspace of limited joints.
        model: MuJoCo model.
        gain: Gain factor for the position limits.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        gain: float = 0.95,
        min_distance_from_limits: float = 0.0,
    ):
        """Initialize the configuration limits.

        Args:
            model: MuJoCo model.
            gain: Gain factor in (0, 1] that determines how fast each joint is allowed to move towards the joint limits at each timestep. Values lower than 1 are safer but may make the joints move slowly.
            min_distance_from_limits: Offset in meters (slide joints) or radians (hinge joints) to be added to the limits. Positive values decrease the range of motion, negative values increase it (i.e., negative values allow penetration).

        Raises:
            LimitDefinitionError: If the gain is not in the range (0, 1].
        """
        if not 0.0 < gain <= 1.0:
            raise LimitDefinitionError(
                f"{self.__class__.__name__} gain must be in the range (0, 1]"
            )

        jnt_indices: list[int] = []  # List to store DoF indices that are limited.
        lower = np.full(model.nq, -mujoco.mjMAXVAL)
        upper = np.full(model.nq, mujoco.mjMAXVAL)
        for jnt in range(model.njnt):
            jnt_type = model.jnt_type[jnt]
            jnt_dim = qpos_width(jnt_type)
            jnt_range = model.jnt_range[jnt]
            padr = model.jnt_qposadr[jnt]
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE or not model.jnt_limited[jnt]:
                continue  # Skip free joints and joints without limits.
            lower[padr : padr + jnt_dim] = jnt_range[0] + min_distance_from_limits
            upper[padr : padr + jnt_dim] = jnt_range[1] - min_distance_from_limits
            jnt_indices.extend(range(model.jnt_dofadr[jnt], model.jnt_dofadr[jnt] + jnt_dim))

        self.indices = np.array(jnt_indices)
        self.indices.setflags(write=False)

        dim = len(self.indices)
        self.projection_matrix = np.eye(model.nv)[self.indices] if dim > 0 else None

        self.lower = lower
        self.upper = upper
        self.model = model
        self.gain = gain

    def compute_qp_inequalities(
        self,
        configuration: Configuration,
        dt: float,
    ) -> Constraint:
        r"""Compute the configuration-dependent joint position limits.

        The limits are defined as:

        .. math::

            {q \ominus q_{min}} \leq \Delta q \leq {q_{max} \ominus q}

        where :math:`q \in {\cal C}` is the robot's configuration and
        :math:`\Delta q \in T_q({\cal C})` is the displacement in the tangent
        space at :math:`q`. See the :ref:`derivations` section for more information.

        Args:
            configuration: Robot configuration :math:`q`.
            dt: Integration timestep in [s].

        Returns:
            Pair :math:`(G, h)` representing the inequality constraint as
            :math:`G \Delta q \leq h`, or ``None`` if there is no limit.
        """
        del dt  # Unused.

        if self.projection_matrix is None:
            return Constraint()

        if configuration.q.size != self.model.nq:
            raise ValueError("Configuration size does not match the model's number of joint positions.")

        # Calculate the maximum allowable change in position towards the upper limit.
        delta_q_max = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(
            m=self.model,
            qvel=delta_q_max,
            dt=1.0,
            qpos1=configuration.q,
            qpos2=self.upper,
        )

        # Calculate the maximum allowable change in position towards the lower limit.
        delta_q_min = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(
            m=self.model,
            qvel=delta_q_min,
            dt=1.0,
            qpos1=self.lower,
            qpos2=configuration.q,
        )

        p_min = self.gain * delta_q_min[self.indices]
        p_max = self.gain * delta_q_max[self.indices]
        G = np.vstack([self.projection_matrix, -self.projection_matrix])
        h = np.hstack([p_max, p_min])
        return Constraint(G=G, h=h)