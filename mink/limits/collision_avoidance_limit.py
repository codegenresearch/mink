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
    distance: float
    contact_points: np.ndarray
    geom1_id: int
    geom2_id: int
    max_distance: float

    @property
    def normal(self) -> np.ndarray:
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
    geom1_body = model.geom_bodyid[contact.geom1_id]
    geom2_body = model.geom_bodyid[contact.geom2_id]
    geom1_contact_pos = contact.contact_points[:3]
    geom2_contact_pos = contact.contact_points[3:]
    jac2 = np.empty((3, model.nv))
    mujoco.mj_jac(model, data, jac2, None, geom2_contact_pos, geom2_body)
    jac1 = np.empty((3, model.nv))
    mujoco.mj_jac(model, data, jac1, None, geom1_contact_pos, geom1_body)
    return contact.normal @ (jac2 - jac1)


def _are_geoms_welded(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    """Returns true if the geoms are part of the same body or are welded together."""
    body1 = model.geom_bodyid[geom_id1]
    body2 = model.geom_bodyid[geom_id2]
    weld1 = model.body_weldid[body1]
    weld2 = model.body_weldid[body2]
    return weld1 == weld2


def _are_geoms_parent_child(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    """Returns true if the geom bodies have a parent-child relationship."""
    body_id1 = model.geom_bodyid[geom_id1]
    body_id2 = model.geom_bodyid[geom_id2]

    weld_id1 = model.body_weldid[body_id1]
    weld_id2 = model.body_weldid[body_id2]

    parent_id1 = model.body_parentid[weld_id1]
    parent_id2 = model.body_parentid[weld_id2]

    weld_parent_id1 = model.body_weldid[parent_id1]
    weld_parent_id2 = model.body_weldid[parent_id2]

    return weld_id1 == weld_parent_id2 or weld_id2 == weld_parent_id1


def _pass_contype_conaffinity_check(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    """Returns true if the geoms pass the contype/conaffinity check."""
    return (model.geom_contype[geom_id1] & model.geom_conaffinity[geom_id2]) or (
        model.geom_contype[geom_id2] & model.geom_conaffinity[geom_id1]
    )


class CollisionAvoidanceLimit(Limit):
    """Normal velocity limit between geom pairs.\n\n    Attributes:\n        model: MuJoCo model.\n        geom_pairs: Set of collision pairs in which to perform active collision\n            avoidance. A collision pair is defined as a pair of geom groups. A geom\n            group is a set of geom names. For each collision pair, the solver will\n            attempt to compute joint velocities that avoid collisions between every\n            geom in the first geom group with every geom in the second geom group.\n            Self collision is achieved by adding a collision pair with the same\n            geom group in both pair fields.\n        gain: Gain factor in (0, 1] that determines how fast the geoms are\n            allowed to move towards each other at each iteration. Smaller values\n            are safer but may make the geoms move slower towards each other.\n        min_distance: The minimum distance to leave between any two geoms. A negative\n            distance allows the geoms to penetrate by the specified amount.\n        detection_distance: The distance between two geoms at which the active collision\n            avoidance limit will be active. A large value will cause collisions to be\n            detected early, but may incur high computational cost. A negative value will\n            cause the geoms to be detected only after they penetrate by the specified\n            amount.\n        bound_relaxation: An offset on the upper bound of each collision avoidance\n            constraint.\n    """

    def __init__(
        self,
        model: mujoco.MjModel,
        geom_pairs: CollisionPairs,
        gain: float = 0.85,
        min_distance: float = 0.005,
        detection_distance: float = 0.01,
        bound_relaxation: float = 0.0,
    ):
        """Initialize collision avoidance limit.\n\n        Args:\n            model: MuJoCo model.\n            geom_pairs: Set of collision pairs in which to perform active collision\n                avoidance. A collision pair is defined as a pair of geom groups. A geom\n                group is a set of geom names. For each collision pair, the mapper will\n                attempt to compute joint velocities that avoid collisions between every\n                geom in the first geom group with every geom in the second geom group.\n                Self collision is achieved by adding a collision pair with the same\n                geom group in both pair fields.\n            gain: Gain factor in (0, 1] that determines how fast the geoms are\n                allowed to move towards each other at each iteration. Smaller values\n                are safer but may make the geoms move slower towards each other.\n            min_distance: The minimum distance to leave between any two geoms. A negative\n                distance allows the geoms to penetrate by the specified amount.\n            detection_distance: The distance between two geoms at which the active collision\n                avoidance limit will be active. A large value will cause collisions to be\n                detected early, but may incur high computational cost. A negative value will\n                cause the geoms to be detected only after they penetrate by the specified\n                amount.\n            bound_relaxation: An offset on the upper bound of each collision avoidance\n                constraint.\n        """
        self.model = model
        self.gain = gain
        self.min_distance = min_distance
        self.detection_distance = detection_distance
        self.bound_relaxation = bound_relaxation
        self.geom_id_pairs = self._construct_geom_id_pairs(geom_pairs)
        self.max_contacts = len(self.geom_id_pairs)

    def compute_qp_inequalities(
        self,
        config: Configuration,
        dt: float,
    ) -> Constraint:
        upper_bounds = np.full((self.max_contacts,), np.inf)
        coefficients = np.zeros((self.max_contacts, self.model.nv))
        for idx, (geom1_id, geom2_id) in enumerate(self.geom_id_pairs):
            contact = self._compute_min_distance_contact(
                config.data, geom1_id, geom2_id
            )
            if contact.is_inactive:
                continue
            hi_bound_dist = contact.distance
            if hi_bound_dist > self.min_distance:
                dist = hi_bound_dist - self.min_distance
                upper_bounds[idx] = (self.gain * dist / dt) + self.bound_relaxation
            else:
                upper_bounds[idx] = self.bound_relaxation
            jac = compute_contact_normal_jacobian(
                self.model, config.data, contact
            )
            coefficients[idx] = -jac
        return Constraint(G=coefficients, h=upper_bounds)

    # Private methods.

    def _compute_min_distance_contact(
        self, data: mujoco.MjData, geom1_id: int, geom2_id: int
    ) -> Contact:
        """Returns the smallest signed distance between a geom pair."""
        contact_points = np.empty(6)
        distance = mujoco.mj_geomDistance(
            self.model,
            data,
            geom1_id,
            geom2_id,
            self.detection_distance,
            contact_points,
        )
        return Contact(
            distance, contact_points, geom1_id, geom2_id, self.detection_distance
        )

    def _convert_geom_ids(self, geom_list: GeomSequence) -> List[int]:
        """Converts a list of geoms (by ID or name) to a list of IDs."""
        ids = []
        for g in geom_list:
            if isinstance(g, int):
                ids.append(g)
            else:
                assert isinstance(g, str)
                ids.append(self.model.geom(g).id)
        return ids

    def _convert_collision_pairs(self, collision_pairs: CollisionPairs):
        geom_id_pairs = []
        for pair in collision_pairs:
            id_pair_A = self._convert_geom_ids(pair[0])
            id_pair_B = self._convert_geom_ids(pair[1])
            id_pair_A = list(set(id_pair_A))
            id_pair_B = list(set(id_pair_B))
            geom_id_pairs.append((id_pair_A, id_pair_B))
        return geom_id_pairs

    def _construct_geom_id_pairs(self, geom_pairs):
        """Returns a set of geom ID pairs for all possible geom-geom collisions.\n\n        The contacts are added based on the following heuristics:\n            1) Geoms that are part of the same body or weld are not included.\n            2) Geoms where the body of one geom is a parent of the body of the other\n                geom are not included.\n            3) Geoms that fail the contype-conaffinity check are ignored.\n\n        Note:\n            1) If two bodies are kinematically welded together (no joints between them)\n                they are considered to be the same body within this function.\n        """
        geom_id_pairs = []
        for id_pair in self._convert_collision_pairs(geom_pairs):
            for geom_a, geom_b in itertools.product(*id_pair):
                if (
                    not _are_geoms_welded(self.model, geom_a, geom_b)
                    and not _are_geoms_parent_child(self.model, geom_a, geom_b)
                    and _pass_contype_conaffinity_check(self.model, geom_a, geom_b)
                ):
                    geom_id_pairs.append((min(geom_a, geom_b), max(geom_a, geom_b)))
        return geom_id_pairs