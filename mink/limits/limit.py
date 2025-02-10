"""All kinematic limits derive from the :class:`Limit` base class."""

import abc
from typing import NamedTuple, Optional

import numpy as np

from ..configuration import Configuration


class LinearConstraint(NamedTuple):
    r"""Linear inequality constraint of the form :math:`G(q) \Delta q \leq h(q)`.

    Inactive if G and h are None.
    """

    inequality_matrix: Optional[np.ndarray] = None
    """Shape (nv, nv)."""
    inequality_vector: Optional[np.ndarray] = None
    """Shape (nv,)."""

    @property
    def is_inactive(self) -> bool:
        """Returns True if the constraint is inactive."""
        return self.inequality_matrix is None and self.inequality_vector is None


class KinematicLimit(abc.ABC):
    """Abstract base class for kinematic limits.

    Subclasses must implement the :py:meth:`~KinematicLimit.compute_qp_inequalities` method
    which takes in the current robot configuration and integration time step and
    returns an instance of :class:`LinearConstraint`.
    """

    @abc.abstractmethod
    def compute_qp_inequalities(
        self,
        robot_configuration: Configuration,
        time_step: float,
    ) -> LinearConstraint:
        r"""Compute limit as linearized QP inequalities of the form:

        .. math::

            G(q) \Delta q \leq h(q)

        where :math:`q \in {\cal C}` is the robot's configuration and
        :math:`\Delta q \in T_q({\cal C})` is the displacement in the tangent
        space at :math:`q`.

        Args:
            robot_configuration: Robot configuration :math:`q`.
            time_step: Integration time step in [s].

        Returns:
            Pair :math:`(G, h)`.
        """
        raise NotImplementedError