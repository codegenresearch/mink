"""Collision avoidance limit with clear documentation and improved mathematical clarity."""

from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import List, Sequence, Union

import mujoco
import numpy as np

from ..configuration import Configuration
from .limit import Constraint, Limit

# Type aliases for clarity.
Geom = Union[int, str]
GeomSequence = Sequence[Geom]
CollisionPair = tuple[GeomSequence, GeomSequence]
CollisionPairs = Sequence[CollisionPair]


@dataclass(frozen=True)
class _Contact:
    """Data class to store contact information between two geoms."""
    dist: float  # Distance between the two geoms.
    fromto: np.ndarray  # Coordinates of the two closest points on the geoms.
    geom1: int  # ID of the first geom.
    geom2: int  # ID of the second geom.
    distmax: float  # Maximum distance before collision detection is active.

    @property
    def normal(self) -> np.ndarray:
        """Compute the unit normal vector from geom1 to geom2."""
        normal = self.fromto[3:] - self.fromto[:3]
        return normal / (np.linalg.norm(normal) + 1e-9)

    @property
    def inactive(self) -> bool:
        """Check if the contact is inactive (i.e., at maximum distance or no contact)."""
        return self.dist == self.distmax and not self.fromto.any()


def _is_welded_together(model: mujoco.MjModel, geom_id1: int, geom_id2: int) -> bool:
    """Determine if two geoms are part of the same body or are welded together."""
    body1 = model.geom_bodyid[geom_id1]
    body2 = model.geom_bodyid[geom_id2]
    weld1 = model.body_weldid[body1]
    weld2 = model.body_weldid[body2]
    return weld1 == weld2


def _are_geom_bodies_parent_child(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    """Check if the bodies of two geoms have a parent-child relationship."""
    body_id1 = model.geom_bodyid[geom_id1]
    body_id2 = model.geom_bodyid[geom_id2]

    # Get the weld IDs for both bodies.
    body_weldid1 = model.body_weldid[body_id1]
    body_weldid2 = model.body_weldid[body_id2]

    # Get the parent IDs of the welds.
    weld_parent_id1 = model.body_parentid[body_weldid1]
    weld_parent_id2 = model.body_parentid[body_weldid2]

    # Check if one body is the parent of the other.
    cond1 = body_weldid1 == weld_parent_weldid2
    cond2 = body_weldid2 == weld_parent_weldid1
    return cond1 or cond2


def _is_pass_contype_conaffinity_check(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    """Verify if the geoms pass the contype/conaffinity check for collision detection."""
    cond1 = bool(model.geom_contype[geom_id1] & model.geom_conaffinity[geom_id2])
    cond2 = bool(model.geom_contype[geom_id2] & model.geom_conaffinity[geom_id1])
    return cond1 or cond2


class CollisionAvoidanceLimit(Limit):
    """Class to enforce collision avoidance between specified geom pairs in a MuJoCo model.

    Attributes:
        model: MuJoCo model instance.
        geom_pairs: List of collision pairs, where each pair consists of two geom groups.
            Each geom group is a sequence of geom names or IDs. The class computes joint
            velocities to avoid collisions between every geom in the first group with every
            geom in the second group.
        gain: Gain factor in (0, 1] that controls the speed of approach to collision limits.
            Lower values are safer but may slow down the approach.
        minimum_distance_from_collisions: Minimum distance to maintain between geoms.
            Negative values allow penetration by the specified amount.
        collision_detection_distance: Distance at which collision detection becomes active.
            Larger values detect collisions earlier but may increase computational cost.
        bound_relaxation: Offset applied to the upper bound of collision avoidance constraints.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        geom_pairs: CollisionPairs,
        gain: float = 0.85,
        minimum_distance_from_collisions: float = 0.005,
        collision_detection_distance: float = 0.01,
        bound_relaxation: float = 0.0,
    ):
        """Initialize the collision avoidance limit with specified parameters.

        Args:
            model: MuJoCo model instance.
            geom_pairs: List of collision pairs, where each pair consists of two geom groups.
                Each geom group is a sequence of geom names or IDs. The class computes joint
                velocities to avoid collisions between every geom in the first group with every
                geom in the second group.
            gain: Gain factor in (0, 1] that controls the speed of approach to collision limits.
                Lower values are safer but may slow down the approach.
            minimum_distance_from_collisions: Minimum distance to maintain between geoms.
                Negative values allow penetration by the specified amount.
            collision_detection_distance: Distance at which collision detection becomes active.
                Larger values detect collisions earlier but may increase computational cost.
            bound_relaxation: Offset applied to the upper bound of collision avoidance constraints.
        """
        self.model = model
        self.gain = gain
        self.minimum_distance_from_collisions = minimum_distance_from_collisions
        self.collision_detection_distance = collision_detection_distance
        self.bound_relaxation = bound_relaxation
        self.geom_id_pairs = self._construct_geom_id_pairs(geom_pairs)
        self.max_num_contacts = len(self.geom_id_pairs)

    def compute_qp_inequalities(
        self,
        configuration: Configuration,
        dt: float,
    ) -> Constraint:
        """Compute the quadratic programming inequalities for collision avoidance.

        Args:
            configuration: Current configuration of the robot.
            dt: Time step for the simulation.

        Returns:
            Constraint: A constraint object containing the coefficient matrix (G) and
                upper bound vector (h) for the quadratic programming problem.
        """
        upper_bound = np.full((self.max_num_contacts,), np.inf)
        coefficient_matrix = np.zeros((self.max_num_contacts, self.model.nv))
        for idx, (geom1_id, geom2_id) in enumerate(self.geom_id_pairs):
            contact = self._compute_contact_with_minimum_distance(
                configuration.data, geom1_id, geom2_id
            )
            if contact.inactive:
                continue
            hi_bound_dist = contact.dist
            if hi_bound_dist > self.minimum_distance_from_collisions:
                dist = hi_bound_dist - self.minimum_distance_from_collisions
                upper_bound[idx] = (self.gain * dist / dt) + self.bound_relaxation
            else:
                upper_bound[idx] = self.bound_relaxation
            jac = self._compute_contact_normal_jacobian(configuration.data, contact)
            coefficient_matrix[idx] = -jac
        return Constraint(G=coefficient_matrix, h=upper_bound)

    # Private methods.

    def _compute_contact_with_minimum_distance(
        self, data: mujoco.MjData, geom1_id: int, geom2_id: int
    ) -> _Contact:
        """Compute the contact information between two geoms with a minimum distance threshold.

        Args:
            data: MuJoCo data instance.
            geom1_id: ID of the first geom.
            geom2_id: ID of the second geom.

        Returns:
            _Contact: Contact information between the two geoms.
        """
        fromto = np.empty(6)
        dist = mujoco.mj_geomDistance(
            self.model,
            data,
            geom1_id,
            geom2_id,
            self.collision_detection_distance,
            fromto,
        )
        return _Contact(
            dist, fromto, geom1_id, geom2_id, self.collision_detection_distance
        )

    def _compute_contact_normal_jacobian(
        self, data: mujoco.MjData, contact: _Contact
    ) -> np.ndarray:
        """Compute the Jacobian matrix for the normal component of the relative velocity between two geoms.

        The Jacobian relates joint velocities to the normal component of the relative
        Cartesian linear velocity between the two geoms. The relationship is given by:

            J dq = n^T (v_2 - v_1)

        where:
        * J is the Jacobian matrix.
        * dq is the joint velocity vector.
        * n^T is the transpose of the unit normal vector from geom1 to geom2.
        * v_1, v_2 are the linear components of the Cartesian velocity at the closest
          points on geom1 and geom2, respectively.

        Args:
            data: MuJoCo data instance.
            contact: Contact information between two geoms.

        Returns:
            np.ndarray: Jacobian matrix for the normal component of the relative velocity.
        """
        geom1_body = self.model.geom_bodyid[contact.geom1]
        geom2_body = self.model.geom_bodyid[contact.geom2]
        geom1_contact_pos = contact.fromto[:3]
        geom2_contact_pos = contact.fromto[3:]
        jac2 = np.empty((3, self.model.nv))
        mujoco.mj_jac(self.model, data, jac2, None, geom2_contact_pos, geom2_body)
        jac1 = np.empty((3, self.model.nv))
        mujoco.mj_jac(self.model, data, jac1, None, geom1_contact_pos, geom1_body)
        return contact.normal @ (jac2 - jac1)

    def _homogenize_geom_id_list(self, geom_list: GeomSequence) -> List[int]:
        """Convert a list of geoms specified by names or IDs into a list of IDs.

        Args:
            geom_list: Sequence of geom names or IDs.

        Returns:
            List[int]: List of geom IDs.
        """
        list_of_int: list[int] = []
        for g in geom_list:
            if isinstance(g, int):
                list_of_int.append(g)
            else:
                assert isinstance(g, str)
                list_of_int.append(self.model.geom(g).id)
        return list_of_int

    def _collision_pairs_to_geom_id_pairs(self, collision_pairs: CollisionPairs):
        """Convert collision pairs of geom names into pairs of geom IDs.

        Args:
            collision_pairs: List of collision pairs, where each pair consists of two geom groups.

        Returns:
            List of geom ID pairs.
        """
        geom_id_pairs = []
        for collision_pair in collision_pairs:
            id_pair_A = self._homogenize_geom_id_list(collision_pair[0])
            id_pair_B = self._homogenize_geom_id_list(collision_pair[1])
            id_pair_A = list(set(id_pair_A))
            id_pair_B = list(set(id_pair_B))
            geom_id_pairs.append((id_pair_A, id_pair_B))
        return geom_id_pairs

    def _construct_geom_id_pairs(self, geom_pairs):
        """Generate all possible geom-geom collision pairs based on specified criteria.

        The pairs are generated based on the following heuristics:
            1) Exclude geoms that are part of the same body or are welded together.
            2) Exclude geoms where one body is a parent of the other.
            3) Exclude geoms that fail the contype-conaffinity check.

        Args:
            geom_pairs: List of collision pairs, where each pair consists of two geom groups.

        Returns:
            List of geom ID pairs for collision avoidance.
        """
        geom_id_pairs = []
        for id_pair in self._collision_pairs_to_geom_id_pairs(geom_pairs):
            for geom_a, geom_b in itertools.product(*id_pair):
                weld_body_cond = not _is_welded_together(self.model, geom_a, geom_b)
                parent_child_cond = not _are_geom_bodies_parent_child(
                    self.model, geom_a, geom_b
                )
                contype_conaffinity_cond = _is_pass_contype_conaffinity_check(
                    self.model, geom_a, geom_b
                )
                if weld_body_cond and parent_child_cond and contype_conaffinity_cond:
                    geom_id_pairs.append((min(geom_a, geom_b), max(geom_a, geom_b)))
        return geom_id_pairs