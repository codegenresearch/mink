"""Collision avoidance limit with enhanced testing and SE3 validation."""

import itertools
from dataclasses import dataclass
from typing import List, Sequence, Union

import mujoco
import numpy as np

from ..configuration import Configuration
from .limit import Constraint, Limit

# Type aliases.
Geom = Union[int, str]
GeomSequence = Sequence[Geom]
CollisionPair = tuple[GeomSequence, GeomSequence]
CollisionPairs = Sequence[CollisionPair]


@dataclass(frozen=True)
class Contact:
    """Represents a contact between two geoms."""
    dist: float
    fromto: np.ndarray
    geom1: int
    geom2: int
    distmax: float

    @property
    def normal(self) -> np.ndarray:
        """Normal vector pointing from geom1 to geom2."""
        normal = self.fromto[3:] - self.fromto[:3]
        return normal / (np.linalg.norm(normal) + 1e-9)

    @property
    def inactive(self) -> bool:
        """Indicates if the contact is inactive."""
        return self.dist == self.distmax and not self.fromto.any()


def _is_welded_together(model: mujoco.MjModel, geom_id1: int, geom_id2: int) -> bool:
    """Check if two geoms are part of the same body or are welded together."""
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

    weld_parent_id1 = model.body_parentid[model.body_weldid[body_id1]]
    weld_parent_id2 = model.body_parentid[model.body_weldid[body_id2]]

    return weld_parent_id1 == model.body_weldid[body_id2] or weld_parent_id2 == model.body_weldid[body_id1]


def _is_pass_contype_conaffinity_check(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    """Check if two geoms pass the contype/conaffinity check."""
    return (model.geom_contype[geom_id1] & model.geom_conaffinity[geom_id2]) or \
           (model.geom_contype[geom_id2] & model.geom_conaffinity[geom_id1])


class CollisionAvoidanceLimit(Limit):
    """Normal velocity limit between geom pairs.

    This class computes constraints to avoid collisions between specified geom pairs
    in a MuJoCo model. It ensures that geoms maintain a minimum distance from each other
    and provides a mechanism to adjust the gain and detection distance for collision
    avoidance.

    Attributes:
        model: MuJoCo model.
        geom_pairs: Set of collision pairs.
        gain: Gain factor in (0, 1].
        minimum_distance_from_collisions: Minimum distance to leave between geoms.
        collision_detection_distance: Distance at which collision avoidance is active.
        bound_relaxation: Offset on the upper bound of each constraint.
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
        """Initialize collision avoidance limit."""
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
            configuration: Robot configuration.
            dt: Integration timestep in [s].

        Returns:
            Constraint object representing the inequality constraint.
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
            jac = self.compute_contact_normal_jacobian(configuration.data, contact)
            coefficient_matrix[idx] = -jac
        return Constraint(G=coefficient_matrix, h=upper_bound)

    def compute_contact_normal_jacobian(
        self, data: mujoco.MjData, contact: Contact
    ) -> np.ndarray:
        """Compute the Jacobian mapping joint velocities to the normal component of
        the relative Cartesian linear velocity between the geom pair.

        Args:
            data: MuJoCo data.
            contact: Contact object representing the contact between two geoms.

        Returns:
            Jacobian matrix.
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

    def _compute_contact_with_minimum_distance(
        self, data: mujoco.MjData, geom1_id: int, geom2_id: int
    ) -> Contact:
        """Compute the smallest signed distance between a geom pair.

        Args:
            data: MuJoCo data.
            geom1_id: ID of the first geom.
            geom2_id: ID of the second geom.

        Returns:
            Contact object representing the smallest signed distance between the geoms.
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
        return Contact(dist, fromto, geom1_id, geom2_id, self.collision_detection_distance)

    def _homogenize_geom_id_list(self, geom_list: GeomSequence) -> List[int]:
        """Convert a list of geoms (specified by ID or name) to a list of IDs.

        Args:
            geom_list: List of geoms specified by ID or name.

        Returns:
            List of geom IDs.
        """
        return [g if isinstance(g, int) else self.model.geom(g).id for g in geom_list]

    def _construct_geom_id_pairs(self, geom_pairs: CollisionPairs) -> List[tuple[int, int]]:
        """Construct a set of geom ID pairs for all possible geom-geom collisions.

        Args:
            geom_pairs: Set of collision pairs.

        Returns:
            List of geom ID pairs.
        """
        geom_id_pairs = []
        for collision_pair in geom_pairs:
            id_pair_A = self._homogenize_geom_id_list(collision_pair[0])
            id_pair_B = self._homogenize_geom_id_list(collision_pair[1])
            for geom_a, geom_b in itertools.product(id_pair_A, id_pair_B):
                is_same_geom = geom_a == geom_b
                is_welded = _is_welded_together(self.model, geom_a, geom_b)
                is_parent_child = _are_geom_bodies_parent_child(self.model, geom_a, geom_b)
                is_contype_conaffinity_pass = _is_pass_contype_conaffinity_check(self.model, geom_a, geom_b)
                if not is_same_geom and not is_welded and not is_parent_child and is_contype_conaffinity_pass:
                    geom_id_pairs.append((min(geom_a, geom_b), max(geom_a, geom_b)))
        return geom_id_pairs