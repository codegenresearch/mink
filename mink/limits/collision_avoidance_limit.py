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
class _Contact:
    """Contact information between two geoms."""
    dist: float
    fromto: np.ndarray
    geom1: int
    geom2: int
    distmax: float

    @property
    def normal(self) -> np.ndarray:
        """Normal vector at the contact point."""
        normal = self.fromto[3:] - self.fromto[:3]
        return normal / (np.linalg.norm(normal) + 1e-9)

    @property
    def inactive(self) -> bool:
        """Check if the contact is inactive."""
        return self.dist == self.distmax and not self.fromto.any()


def _is_welded_together(model: mujoco.MjModel, geom_id1: int, geom_id2: int) -> bool:
    """Check if two geoms are part of the same body or if their bodies are welded.\n\n    Args:\n        model: MuJoCo model.\n        geom_id1: ID of the first geom.\n        geom_id2: ID of the second geom.\n\n    Returns:\n        True if the geoms are part of the same body or if their bodies are welded, False otherwise.\n    """
    body1 = model.geom_bodyid[geom_id1]
    body2 = model.geom_bodyid[geom_id2]
    weld1 = model.body_weldid[body1]
    weld2 = model.body_weldid[body2]
    return weld1 == weld2


def _are_geom_bodies_parent_child(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    """Check if the bodies of two geoms have a parent-child relationship.\n\n    Args:\n        model: MuJoCo model.\n        geom_id1: ID of the first geom.\n        geom_id2: ID of the second geom.\n\n    Returns:\n        True if the bodies of the geoms have a parent-child relationship, False otherwise.\n    """
    body_id1 = model.geom_bodyid[geom_id1]
    body_id2 = model.geom_bodyid[geom_id2]

    weld_parent_id1 = model.body_parentid[model.body_weldid[body_id1]]
    weld_parent_id2 = model.body_parentid[model.body_weldid[body_id2]]

    return model.body_weldid[body_id1] == weld_parent_id2 or model.body_weldid[body_id2] == weld_parent_id1


def _is_pass_contype_conaffinity_check(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    """Check if two geoms pass the contype/conaffinity check.\n\n    Args:\n        model: MuJoCo model.\n        geom_id1: ID of the first geom.\n        geom_id2: ID of the second geom.\n\n    Returns:\n        True if the geoms pass the contype/conaffinity check, False otherwise.\n    """
    return bool(model.geom_contype[geom_id1] & model.geom_conaffinity[geom_id2]) or bool(
        model.geom_contype[geom_id2] & model.geom_conaffinity[geom_id1]
    )


class CollisionAvoidanceLimit(Limit):
    """Limit on normal velocity between geom pairs.\n\n    Attributes:\n        model: MuJoCo model.\n        geom_pairs: Set of collision pairs to perform active collision avoidance.\n        gain: Gain factor in (0, 1] determining the speed of geom movement towards each other.\n        minimum_distance_from_collisions: Minimum distance to maintain between geoms.\n        collision_detection_distance: Distance at which collision avoidance becomes active.\n        bound_relaxation: Offset on the upper bound of collision avoidance constraints.\n    """

    def __init__(
        self,
        model: mujoco.MjModel,
        geom_pairs: CollisionPairs,
        gain: float = 0.85,
        minimum_distance_from_collisions: float = 0.005,
        collision_detection_distance: float = 0.01,
        bound_relaxation: float = 0.0,
    ):
        """Initialize collision avoidance limit.\n\n        Args:\n            model: MuJoCo model.\n            geom_pairs: Set of collision pairs to perform active collision avoidance.\n            gain: Gain factor in (0, 1] determining the speed of geom movement towards each other.\n            minimum_distance_from_collisions: Minimum distance to maintain between geoms.\n            collision_detection_distance: Distance at which collision avoidance becomes active.\n            bound_relaxation: Offset on the upper bound of collision avoidance constraints.\n        """
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
        """Compute the quadratic programming inequalities for collision avoidance.\n\n        Args:\n            configuration: Robot configuration.\n            dt: Integration timestep in seconds.\n\n        Returns:\n            Constraint object representing the inequality constraints.\n        """
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

    def _compute_contact_with_minimum_distance(
        self, data: mujoco.MjData, geom1_id: int, geom2_id: int
    ) -> _Contact:
        """Compute the smallest signed distance between a geom pair.\n\n        Args:\n            data: MuJoCo data.\n            geom1_id: ID of the first geom.\n            geom2_id: ID of the second geom.\n\n        Returns:\n            Contact object containing the smallest signed distance and other contact details.\n        """
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
        """Compute the Jacobian mapping joint velocities to the normal component of the relative Cartesian linear velocity.\n\n        Args:\n            data: MuJoCo data.\n            contact: Contact object containing contact details.\n\n        Returns:\n            Jacobian matrix.\n        """
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
        """Convert a list of geoms (specified by ID or name) to a list of IDs.\n\n        Args:\n            geom_list: List of geoms specified by ID or name.\n\n        Returns:\n            List of geom IDs.\n        """
        return [
            g if isinstance(g, int) else self.model.geom(g).id for g in geom_list
        ]

    def _collision_pairs_to_geom_id_pairs(self, collision_pairs: CollisionPairs) -> List[tuple]:
        """Convert collision pairs of geom names to collision pairs of geom IDs.\n\n        Args:\n            collision_pairs: List of collision pairs specified by geom names.\n\n        Returns:\n            List of collision pairs specified by geom IDs.\n        """
        return [
            (list(set(self._homogenize_geom_id_list(pair[0]))), list(set(self._homogenize_geom_id_list(pair[1]))))
            for pair in collision_pairs
        ]

    def _construct_geom_id_pairs(self, geom_pairs: CollisionPairs) -> List[tuple]:
        """Construct a list of geom ID pairs for all possible geom-geom collisions.\n\n        Args:\n            geom_pairs: List of collision pairs specified by geom names.\n\n        Returns:\n            List of geom ID pairs for collision avoidance.\n        """
        geom_id_pairs = []
        for id_pair in self._collision_pairs_to_geom_id_pairs(geom_pairs):
            for geom_a, geom_b in itertools.product(*id_pair):
                if (
                    not _is_welded_together(self.model, geom_a, geom_b)
                    and not _are_geom_bodies_parent_child(self.model, geom_a, geom_b)
                    and _is_pass_contype_conaffinity_check(self.model, geom_a, geom_b)
                ):
                    geom_id_pairs.append((min(geom_a, geom_b), max(geom_a, geom_b)))
        return geom_id_pairs