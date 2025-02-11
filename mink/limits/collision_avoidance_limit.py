"""Collision avoidance limit."""

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
    dist: float
    fromto: np.ndarray
    geom1: int
    geom2: int
    distmax: float

    @property
    def normal(self) -> np.ndarray:
        normal = self.fromto[3:] - self.fromto[:3]
        return normal / (np.linalg.norm(normal) + 1e-9)

    @property
    def inactive(self) -> bool:
        return self.dist == self.distmax and not self.fromto.any()


def compute_contact_normal_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contact: Contact,
) -> np.ndarray:
    geom1_body = model.geom_bodyid[contact.geom1]
    geom2_body = model.geom_bodyid[contact.geom2]
    geom1_contact_pos = contact.fromto[:3]
    geom2_contact_pos = contact.fromto[3:]
    jac2 = np.empty((3, model.nv))
    mujoco.mj_jac(model, data, jac2, None, geom2_contact_pos, geom2_body)
    jac1 = np.empty((3, model.nv))
    mujoco.mj_jac(model, data, jac1, None, geom1_contact_pos, geom1_body)
    return contact.normal @ (jac2 - jac1)


def _is_welded_together(model: mujoco.MjModel, geom_id1: int, geom_id2: int) -> bool:
    """Returns true if the geoms are part of the same body, or if their bodies are
    welded together."""
    body1 = model.geom_bodyid[geom_id1]
    body2 = model.geom_bodyid[geom_id2]
    weld1 = model.body_weldid[body1]
    weld2 = model.body_weldid[body2]
    return weld1 == weld2


def _are_geom_bodies_parent_child(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    """Returns true if the geom bodies have a parent-child relationship."""
    body_id1 = model.geom_bodyid[geom_id1]
    body_id2 = model.geom_bodyid[geom_id2]

    weld_id1 = model.body_weldid[body_id1]
    weld_id2 = model.body_weldid[body_id2]

    parent_id1 = model.body_parentid[weld_id1]
    parent_id2 = model.body_parentid[weld_id2]

    return weld_id1 == parent_id2 or weld_id2 == parent_id1


def _is_pass_contype_conaffinity_check(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    """Returns true if the geoms pass the contype/conaffinity check."""
    return (bool(model.geom_contype[geom_id1] & model.geom_conaffinity[geom_id2]) or
            bool(model.geom_contype[geom_id2] & model.geom_conaffinity[geom_id1]))


class CollisionAvoidanceLimit(Limit):
    """Normal velocity limit between geom pairs.

    Attributes:
        model: MuJoCo model.
        geom_pairs: Set of collision pairs in which to perform active collision
            avoidance. A collision pair is defined as a pair of geom groups. A geom
            group is a set of geom names. For each collision pair, the solver will
            attempt to compute joint velocities that avoid collisions between every
            geom in the first geom group with every geom in the second geom group.
            Self collision is achieved by adding a collision pair with the same
            geom group in both pair fields.
        gain: Gain factor in (0, 1] that determines how fast the geoms are
            allowed to move towards each other at each iteration. Smaller values
            are safer but may make the geoms move slower towards each other.
        min_distance: The minimum distance to leave between any two geoms. A negative
            distance allows the geoms to penetrate by the specified amount.
        detection_distance: The distance between two geoms at which the active collision
            avoidance limit will be active. A large value will cause collisions to be
            detected early, but may incur high computational cost. A negative value will
            cause the geoms to be detected only after they penetrate by the specified
            amount.
        bound_relaxation: An offset on the upper bound of each collision avoidance
            constraint.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        geom_pairs: CollisionPairs,
        gain: float = 0.85,
        min_distance: float = 0.005,
        detection_distance: float = 0.01,
        bound_relaxation: float = 0.0,
    ):
        """Initialize collision avoidance limit.

        Args:
            model: MuJoCo model.
            geom_pairs: Set of collision pairs in which to perform active collision
                avoidance. A collision pair is defined as a pair of geom groups. A geom
                group is a set of geom names. For each collision pair, the mapper will
                attempt to compute joint velocities that avoid collisions between every
                geom in the first geom group with every geom in the second geom group.
                Self collision is achieved by adding a collision pair with the same
                geom group in both pair fields.
            gain: Gain factor in (0, 1] that determines how fast the geoms are
                allowed to move towards each other at each iteration. Smaller values
                are safer but may make the geoms move slower towards each other.
            min_distance: The minimum distance to leave between any two geoms. A negative
                distance allows the geoms to penetrate by the specified amount.
            detection_distance: The distance between two geoms at which the active collision
                avoidance limit will be active. A large value will cause collisions to be
                detected early, but may incur high computational cost. A negative value will
                cause the geoms to be detected only after they penetrate by the specified
                amount.
            bound_relaxation: An offset on the upper bound of each collision avoidance
                constraint.
        """
        self.model = model
        self.gain = gain
        self.min_distance = min_distance
        self.detection_distance = detection_distance
        self.bound_relaxation = bound_relaxation
        self.geom_id_pairs = self._construct_geom_id_pairs(geom_pairs)
        self.max_num_contacts = len(self.geom_id_pairs)

    def compute_qp_inequalities(
        self,
        configuration: Configuration,
        dt: float,
    ) -> Constraint:
        upper_bounds = np.full((self.max_num_contacts,), np.inf)
        coefficient_matrix = np.zeros((self.max_num_contacts, self.model.nv))
        for index, (geom1, geom2) in enumerate(self.geom_id_pairs):
            contact = self._compute_contact_with_minimum_distance(
                configuration.data, geom1, geom2
            )
            if contact.inactive:
                continue
            distance = contact.dist
            if distance > self.min_distance:
                adjusted_distance = distance - self.min_distance
                upper_bounds[index] = (self.gain * adjusted_distance / dt) + self.bound_relaxation
            else:
                upper_bounds[index] = self.bound_relaxation
            jacobian = compute_contact_normal_jacobian(
                self.model, configuration.data, contact
            )
            coefficient_matrix[index] = -jacobian
        return Constraint(G=coefficient_matrix, h=upper_bounds)

    # Private methods.

    def _compute_contact_with_minimum_distance(
        self, data: mujoco.MjData, geom1: int, geom2: int
    ) -> Contact:
        """Returns the smallest signed distance between a geom pair."""
        fromto = np.empty(6)
        dist = mujoco.mj_geomDistance(
            self.model,
            data,
            geom1,
            geom2,
            self.detection_distance,
            fromto,
        )
        return Contact(
            dist, fromto, geom1, geom2, self.detection_distance
        )

    def _convert_geom_identifiers_to_ids(self, geom_list: GeomSequence) -> List[int]:
        """Converts a list of geoms (specified by ID or name) to a list of IDs."""
        geom_ids = []
        for geom in geom_list:
            if isinstance(geom, int):
                geom_ids.append(geom)
            else:
                assert isinstance(geom, str)
                geom_ids.append(self.model.geom(geom).id)
        return geom_ids

    def _convert_collision_pairs_to_geom_id_pairs(self, collision_pairs: CollisionPairs):
        geom_id_pairs = []
        for collision_pair in collision_pairs:
            geom_ids_1 = self._convert_geom_identifiers_to_ids(collision_pair[0])
            geom_ids_2 = self._convert_geom_identifiers_to_ids(collision_pair[1])
            geom_ids_1 = list(set(geom_ids_1))
            geom_ids_2 = list(set(geom_ids_2))
            geom_id_pairs.append((geom_ids_1, geom_ids_2))
        return geom_id_pairs

    def _construct_geom_id_pairs(self, geom_pairs):
        """Returns a set of geom ID pairs for all possible geom-geom collisions.

        The contacts are added based on the following heuristics:
            1) Geoms that are part of the same body or weld are not included.
            2) Geoms where the body of one geom is a parent of the body of the other
                geom are not included.
            3) Geoms that fail the contype-conaffinity check are ignored.

        Note:
            1) If two bodies are kinematically welded together (no joints between them)
                they are considered to be the same body within this function.
        """
        geom_id_pairs = []
        for id_pair in self._convert_collision_pairs_to_geom_id_pairs(geom_pairs):
            for geom_a, geom_b in itertools.product(*id_pair):
                if (not _is_welded_together(self.model, geom_a, geom_b) and
                    not _are_geom_bodies_parent_child(self.model, geom_a, geom_b) and
                    _is_pass_contype_conaffinity_check(self.model, geom_a, geom_b)):
                    geom_id_pairs.append((min(geom_a, geom_b), max(geom_a, geom_b)))
        return geom_id_pairs