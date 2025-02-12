"""All kinematic limits derive from the :class:`Limit` base class."""

import abc
from typing import NamedTuple, Optional

import numpy as np

from ..configuration import Configuration


class Constraint(NamedTuple):
    r"""Linear inequality constraint of the form :math:`G(q) \Delta q \leq h(q)`.\n\n    Inactive if G and h are None.\n    """

    G: Optional[np.ndarray] = None
    """Shape (nv, nv)."""
    h: Optional[np.ndarray] = None
    """Shape (nv,)."""

    @property
    def inactive(self) -> bool:
        """Returns True if the constraint is inactive."""
        return self.G is None and self.h is None


class Limit(abc.ABC):
    """Abstract base class for kinematic limits.\n\n    Subclasses must implement the :py:meth:`~Limit.compute_qp_inequalities` method\n    which takes in the current robot configuration and integration time step and\n    returns an instance of :class:`Constraint`.\n    """

    @abc.abstractmethod
    def compute_qp_inequalities(
        self,
        configuration: Configuration,
        dt: float,
    ) -> Constraint:
        r"""Compute limit as linearized QP inequalities of the form:\n\n        .. math::\n\n            G(q) \Delta q \leq h(q)\n\n        where :math:`q \in {\cal C}` is the robot's configuration and\n        :math:`\Delta q \in T_q({\cal C})` is the displacement in the tangent\n        space at :math:`q`.\n\n        Args:\n            configuration: Robot configuration :math:`q`.\n            dt: Integration time step in [s].\n\n        Returns:\n            Pair :math:`(G, h)`.\n        """
        raise NotImplementedError