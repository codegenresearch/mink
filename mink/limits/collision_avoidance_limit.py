"""Collision avoidance limit."""

import itertools
from dataclasses import dataclass
from typing import List, Sequence, Union

import mujoco
import numpy as np

from ..configuration import Configuration
from .limit import Constraint, Limit

# Type aliases.
GeomIdentifier = Union[int, str]
GeomSequence = Sequence[GeomIdentifier]
CollisionPair = tuple[GeomSequence, GeomSequence]
CollisionPairs = Sequence[CollisionPair]


@dataclass(frozen=True)
class Contact:
    distance: float
    contact_points: np.ndarray
    geom_id_1: int
    geom_id_2: int
    max_distance: float

    @property
    def normal_vector(self) -> np.ndarray:
        normal = self.contact_points[3:] - self.contact_points[:3]
        return normal / (np.linalg.norm(normal) + 1e-9)

    @property
    def is_inactive(self) -> bool:
        return self.distance == self.max_distance and not self.contact_points.any()


def compute_contact_normal_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contact: Contact,
) -> np.ndarray:
    body_id_1 = model.geom_bodyid[contact.geom_id_1]
    body_id_2 = model.geom_bodyid[contact.geom_id_2]
    contact_point_1 = contact.contact_points[:3]
    contact_point_2 = contact.contact_points[3:]
    jacobian_2 = np.empty((3, model.nv))
    mujoco.mj_jac(model, data, jacobian_2, None, contact_point_2, body_id_2)
    jacobian_1 = np.empty((3, model.nv))
    mujoco.mj_jac(model, data, jacobian_1, None, contact_point_1, body_id_1)
    return contact.normal_vector @ (jacobian_2 - jacobian_1)


def _are_geoms_connected_by_weld(
    model: mujoco.MjModel, geom_id_1: int, geom_id_2: int
) -> bool:
    """Returns True if the geoms are part of the same body or are welded together."""
    body_id_1 = model.geom_bodyid[geom_id_1]
    body_id_2 = model.geom_bodyid[geom_id_2]
    weld_id_1 = model.body_weldid[body_id_1]
    weld_id_2 = model.body_weldid[body_id_2]
    return weld_id_1 == weld_id_2


def _are_geoms_parent_child(
    model: mujoco.MjModel, geom_id_1: int, geom_id_2: int
) -> bool:
    """Returns True if the geoms have a parent-child relationship."""
    body_id_1 = model.geom_bodyid[geom_id_1]
    body_id_2 = model.geom_bodyid[geom_id_2]

    weld_id_1 = model.body_weldid[body_id_1]
    weld_id_2 = model.body_weldid[body_id_2]

    weld_parent_id_1 = model.body_parentid[weld_id_1]
    weld_parent_id_2 = model.body_parentid[weld_id_2]

    weld_parent_weld_id_1 = model.body_weldid[weld_parent_id_1]
    weld_parent_weld_id_2 = model.body_weldid[weld_parent_id_2]

    return weld_id_1 == weld_parent_weld_id_2 or weld_id_2 == weld_parent_weld_id_1


def _pass_contype_conaffinity_check(
    model: mujoco.MjModel, geom_id_1: int, geom_id_2: int
) -> bool:
    """Returns True if the geoms pass the contype/conaffinity check."""
    return (
        bool(model.geom_contype[geom_id_1] & model.geom_conaffinity[geom_id_2])
        or bool(model.geom_contype[geom_id_2] & model.geom_conaffinity[geom_id_1])
    )


class CollisionAvoidanceLimit(Limit):
    """Normal velocity limit between geom pairs.\n\n    Attributes:\n        model: MuJoCo model.\n        geom_pairs: Set of collision pairs in which to perform active collision\n            avoidance. A collision pair is defined as a pair of geom groups. A geom\n            group is a set of geom identifiers. For each collision pair, the solver will\n            attempt to compute joint velocities that avoid collisions between every\n            geom in the first geom group with every geom in the second geom group.\n            Self collision is achieved by adding a collision pair with the same\n            geom group in both pair fields.\n        gain: Gain factor in (0, 1] that determines how fast the geoms are\n            allowed to move towards each other at each iteration. Smaller values\n            are safer but may make the geoms move slower towards each other.\n        min_collision_distance: The minimum distance to leave between any two geoms.\n            A negative distance allows the geoms to penetrate by the specified amount.\n        detection_distance: The distance at which collision avoidance becomes active.\n            A larger value will detect collisions earlier but may increase computational\n            cost. A negative value will only detect collisions after penetration.\n        bound_offset: An offset applied to the upper bound of each constraint.\n    """

    def __init__(
        self,
        model: mujoco.MjModel,
        geom_pairs: CollisionPairs,
        gain: float = 0.85,
        min_collision_distance: float = 0.005,
        detection_distance: float = 0.01,
        bound_offset: float = 0.0,
    ):
        """Initialize collision avoidance limit.\n\n        Args:\n            model: MuJoCo model.\n            geom_pairs: Set of collision pairs in which to perform active collision\n                avoidance. A collision pair is defined as a pair of geom groups. A geom\n                group is a set of geom identifiers. For each collision pair, the mapper\n                will attempt to compute joint velocities that avoid collisions between\n                every geom in the first geom group with every geom in the second geom\n                group. Self collision is achieved by adding a collision pair with the\n                same geom group in both pair fields.\n            gain: Gain factor in (0, 1] that determines how fast the geoms are\n                allowed to move towards each other at each iteration. Smaller values\n                are safer but may make the geoms move slower towards each other.\n            min_collision_distance: The minimum distance to leave between any two geoms.\n                A negative distance allows the geoms to penetrate by the specified amount.\n            detection_distance: The distance at which collision avoidance becomes active.\n                A larger value will detect collisions earlier but may increase computational\n                cost. A negative value will only detect collisions after penetration.\n            bound_offset: An offset applied to the upper bound of each constraint.\n        """
        self.model = model
        self.gain = gain
        self.min_collision_distance = min_collision_distance
        self.detection_distance = detection_distance
        self.bound_offset = bound_offset
        self.geom_id_pairs = self._construct_geom_id_pairs(geom_pairs)
        self.max_contacts = len(self.geom_id_pairs)

    def compute_qp_inequalities(
        self,
        configuration: Configuration,
        dt: float,
    ) -> Constraint:
        upper_bounds = np.full((self.max_contacts,), np.inf)
        coefficient_matrix = np.zeros((self.max_contacts, self.model.nv))
        for index, (geom_id_1, geom_id_2) in enumerate(self.geom_id_pairs):
            contact = self._compute_contact(configuration.data, geom_id_1, geom_id_2)
            if contact.is_inactive:
                continue
            upper_bounds[index] = self._calculate_upper_bound(contact, dt)
            jacobian = compute_contact_normal_jacobian(
                self.model, configuration.data, contact
            )
            coefficient_matrix[index] = -jacobian
        return Constraint(G=coefficient_matrix, h=upper_bounds)

    # Private methods.

    def _compute_contact(self, data: mujoco.MjData, geom_id_1: int, geom_id_2: int) -> Contact:
        """Returns the smallest signed distance between a geom pair."""
        contact_points = np.empty(6)
        distance = mujoco.mj_geomDistance(
            self.model,
            data,
            geom_id_1,
            geom_id_2,
            self.detection_distance,
            contact_points,
        )
        return Contact(
            distance, contact_points, geom_id_1, geom_id_2, self.detection_distance
        )

    def _convert_geom_identifiers_to_ids(self, geom_list: GeomSequence) -> List[int]:
        """Converts a list of geom identifiers to a list of geom IDs."""
        geom_ids = []
        for geom in geom_list:
            if isinstance(geom, int):
                geom_ids.append(geom)
            else:
                assert isinstance(geom, str)
                geom_ids.append(self.model.geom(geom).id)
        return geom_ids

    def _convert_collision_pairs_to_geom_id_pairs(self, collision_pairs: CollisionPairs) -> List[tuple[List[int], List[int]]]:
        geom_id_pairs = []
        for geom_pair in collision_pairs:
            geom_ids_pair_1 = self._convert_geom_identifiers_to_ids(geom_pair[0])
            geom_ids_pair_2 = self._convert_geom_identifiers_to_ids(geom_pair[1])
            geom_ids_pair_1 = list(set(geom_ids_pair_1))
            geom_ids_pair_2 = list(set(geom_ids_pair_2))
            geom_id_pairs.append((geom_ids_pair_1, geom_ids_pair_2))
        return geom_id_pairs

    def _construct_geom_id_pairs(self, collision_pairs: CollisionPairs) -> List[tuple[int, int]]:
        """Constructs a list of geom ID pairs for all possible collisions.\n\n        Geom pairs that are part of the same body or are welded together, have a\n        parent-child relationship, or fail the contype-conaffinity check are excluded.\n        """
        geom_id_pairs = []
        for id_pair in self._convert_collision_pairs_to_geom_id_pairs(collision_pairs):
            for geom_a, geom_b in itertools.product(*id_pair):
                if (
                    not _are_geoms_connected_by_weld(self.model, geom_a, geom_b)
                    and not _are_geoms_parent_child(self.model, geom_a, geom_b)
                    and _pass_contype_conaffinity_check(self.model, geom_a, geom_b)
                ):
                    geom_id_pairs.append((min(geom_a, geom_b), max(geom_a, geom_b)))
        return geom_id_pairs

    def _calculate_upper_bound(self, contact: Contact, dt: float) -> float:
        """Calculates the upper bound for a given contact."""
        if contact.distance > self.min_collision_distance:
            distance_difference = contact.distance - self.min_collision_distance
            return (self.gain * distance_difference / dt) + self.bound_offset
        return self.bound_offset