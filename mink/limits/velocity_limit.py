"""Joint velocity limit."""

from typing import Mapping

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
        idx: Indices of velocity-limited joints in the tangent space.
        max_vel: Maximum allowed velocity for each limited joint.
        proj_matrix: Projection matrix to the subspace of velocity-limited joints.
    """

    idx: np.ndarray
    max_vel: np.ndarray
    proj_matrix: np.ndarray

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
        max_vel_list = []
        idx_list = []
        for joint_name, max_vel in velocities.items():
            jid = model.joint(joint_name).id
            jnt_type = model.jnt_type[jid]
            jnt_dim = dof_width(jnt_type)
            jnt_id = model.jnt_dofadr[jid]
            assert jnt_type != mujoco.mjtJoint.mjJNT_FREE, f"Free joint {joint_name} is not supported"
            max_vel = np.atleast_1d(max_vel)
            assert max_vel.shape == (jnt_dim,), f"Joint {joint_name} must have a limit of shape ({jnt_dim},). Got: {max_vel.shape}"
            idx_list.extend(range(jnt_id, jnt_id + jnt_dim))
            max_vel_list.extend(max_vel.tolist())

        self.idx = np.array(idx_list)
        self.idx.setflags(write=False)
        self.max_vel = np.array(max_vel_list)
        self.max_vel.setflags(write=False)

        dim = len(self.idx)
        self.proj_matrix = np.eye(model.nv)[self.idx] if dim > 0 else None

    def compute_qp_inequalities(
        self, config: Configuration, dt: float
    ) -> Constraint:
        r"""Compute the configuration-dependent joint velocity limits.

        The limits are defined as:

        .. math::

            -v_{\text{max}} \cdot dt \leq \Delta q \leq v_{\text{max}} \cdot dt

        where :math:`v_{max}` is the maximum velocity vector and :math:`\Delta q`
        is the velocity displacement in the tangent space.

        Args:
            config: Robot configuration.
            dt: Integration timestep in seconds.

        Returns:
            Pair :math:`(G, h)` representing the inequality constraint as
            :math:`G \Delta q \leq h`, or an empty constraint if no limits are set.
        """
        if self.proj_matrix is None:
            return Constraint()
        G = np.vstack([self.proj_matrix, -self.proj_matrix])
        h = np.hstack([dt * self.max_vel, dt * self.max_vel])
        return Constraint(G=G, h=h)