"""Joint velocity limit."""

from typing import Mapping, List

import mujoco
import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from ..constants import dof_width
from .exceptions import LimitDefinitionError
from .limit import Constraint, Limit


class VelocityLimit(Limit):
    """Inequality constraint on joint velocities in a robot model.

    Floating base joints are ignored.

    Attributes:
        indices: Indices of velocity-limited joints in the tangent space.
        limit: Maximum allowed velocity for each limited joint.
        projection_matrix: Projection matrix to the subspace of velocity-limited joints.
    """

    indices: np.ndarray
    limit: np.ndarray
    projection_matrix: np.ndarray

    def __init__(
        self,
        model: mujoco.MjModel,
        velocities: Mapping[str, npt.ArrayLike] = {},
    ):
        """Initialize velocity limits.

        Args:
            model: MuJoCo model.
            velocities: Dictionary mapping joint name to maximum allowed velocity.
        """
        limit_list: List[float] = []
        index_list: List[int] = []
        for joint_name, max_vel in velocities.items():
            jid = model.joint(joint_name).id
            jnt_type = model.jnt_type[jid]
            vdim = dof_width(jnt_type)
            vadr = model.jnt_dofadr[jid]
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                raise LimitDefinitionError(f"Joint {joint_name} is a free joint and is not supported")
            max_vel = np.atleast_1d(max_vel)
            if vdim == 3 and max_vel.shape != (3,):
                raise LimitDefinitionError(f"Joint {joint_name} must have a limit of shape (3,). Got: {max_vel.shape}")
            elif vdim != 3 and max_vel.shape != (vdim,):
                raise LimitDefinitionError(f"Joint {joint_name} must have a limit of shape ({vdim},). Got: {max_vel.shape}")
            index_list.extend(range(vadr, vadr + vdim))
            limit_list.extend(max_vel.tolist())

        self.indices = np.array(index_list)
        self.indices.setflags(write=False)
        self.limit = np.array(limit_list)
        self.limit.setflags(write=False)

        dim = len(self.indices)
        self.projection_matrix = np.eye(model.nv)[self.indices] if dim > 0 else None

    def compute_qp_inequalities(
        self, configuration: Configuration, dt: float
    ) -> Constraint:
        r"""Compute the configuration-dependent joint velocity limits.

        The limits are defined as:

        .. math::

            -v_{\text{max}} \cdot dt \leq \Delta q \leq v_{\text{max}} \cdot dt

        where :math:`v_{\text{max}}` is the maximum velocity vector and :math:`\Delta q`
        is the velocity displacement in the tangent space.

        Args:
            configuration: Robot configuration.
            dt: Integration timestep in seconds.

        Returns:
            Pair :math:`(G, h)` representing the inequality constraint as
            :math:`G \Delta q \leq h`, where :math:`G` is a matrix of shape
            :math:`(2 \times \text{dim}, \text{nv})` and :math:`h` is a vector of
            shape :math:`(2 \times \text{dim},)`, or `None` if no limits are set.
        """
        del configuration  # Unused.
        if self.projection_matrix is None:
            return Constraint()
        G = np.vstack([self.projection_matrix, -self.projection_matrix])
        h = np.hstack([dt * self.limit, dt * self.limit])
        return Constraint(G=G, h=h)