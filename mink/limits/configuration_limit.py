"""Module for defining joint position limits in a robot model."""

import mujoco
import numpy as np

from ..configuration import Configuration
from ..constants import qpos_width
from .exceptions import LimitDefinitionError
from .limit import Constraint, Limit


class PositionLimit(Limit):
    """Inequality constraint on joint positions in a robot model.\n\n    This class handles the position limits for joints in a MuJoCo model, ensuring\n    that the robot's joints do not exceed their specified limits. Floating base\n    joints are ignored.\n\n    Attributes:\n        indices: Degrees of freedom (DoF) indices that are limited.\n        lower: Lower position limits for each DoF.\n        upper: Upper position limits for each DoF.\n        projection_matrix: Projection matrix from the full tangent space to the\n            subspace of limited DoFs.\n        model: MuJoCo model.\n        gain: Gain factor that controls the speed of approach towards joint limits.\n    """

    def __init__(
        self,
        model: mujoco.MjModel,
        gain: float = 0.95,
        min_distance_from_limits: float = 0.0,
    ):
        """Initialize the position limits for the robot model.\n\n        Args:\n            model: MuJoCo model instance.\n            gain: Gain factor in the range (0, 1] that determines how quickly each\n                joint can move towards its limits. Values closer to 0 are safer but\n                slower.\n            min_distance_from_limits: Offset to be added to the joint limits. Positive\n                values reduce the range of motion, negative values increase it.\n\n        Raises:\n            LimitDefinitionError: If the gain is not within the valid range.\n        """
        if not 0.0 < gain <= 1.0:
            raise LimitDefinitionError(
                f"{self.__class__.__name__} gain must be in the range (0, 1]"
            )

        index_list = []  # List to store indices of limited DoFs.
        lower = np.full(model.nq, -np.inf)
        upper = np.full(model.nq, np.inf)

        for jnt in range(model.njnt):
            jnt_type = model.jnt_type[jnt]
            qpos_dim = qpos_width(jnt_type)
            jnt_range = model.jnt_range[jnt]
            padr = model.jnt_qposadr[jnt]

            if jnt_type == mujoco.mjtJoint.mjJNT_FREE or not model.jnt_limited[jnt]:
                continue

            lower[padr : padr + qpos_dim] = jnt_range[0] + min_distance_from_limits
            upper[padr : padr + qpos_dim] = jnt_range[1] - min_distance_from_limits
            index_list.append(model.jnt_dofadr[jnt])

        self.indices = np.array(index_list)
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
        r"""Compute the configuration-dependent joint position limits.\n\n        The position limits are defined as:\n\n        .. math::\n\n            {q \ominus q_{min}} \leq \Delta q \leq {q_{max} \ominus q}\n\n        where :math:`q \in {\cal C}` is the robot's configuration and\n        :math:`\Delta q \in T_q({\cal C})` is the displacement in the tangent\n        space at :math:`q`.\n\n        Args:\n            configuration: Current robot configuration.\n            dt: Integration timestep (unused in this method).\n\n        Returns:\n            Constraint: Pair :math:`(G, h)` representing the inequality constraint\n                as :math:`G \Delta q \leq h`, or an inactive constraint if there are\n                no limits.\n        """
        del dt  # Unused.

        delta_q_max = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(
            m=self.model,
            qvel=delta_q_max,
            dt=1.0,
            qpos1=configuration.q,
            qpos2=self.upper,
        )

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