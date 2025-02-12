"""Collision avoidance limit with improved documentation, type hints, and error handling."""

import itertools
from dataclasses import dataclass
from typing import List, Sequence, Union

import mujoco
import numpy as np

from ..configuration import Configuration
from .limit import Constraint, Limit

# Type aliases for better readability and type hinting.
Geom = Union[int, str]
GeomSequence = Sequence[Geom]
CollisionPair = tuple[GeomSequence, GeomSequence]
CollisionPairs = Sequence[CollisionPair]


@dataclass(frozen=True)
class _Contact:
    """Data class to store contact information between two geoms."""
    dist: float
    fromto: np.ndarray
    geom1: int
    geom2: int
    distmax: float

    @property
    def normal(self) -> np.ndarray:
        """Compute the normalized normal vector from the contact point."""
        normal = self.fromto[3:] - self.fromto[:3]
        return normal / (np.linalg.norm(normal) + 1e-9)

    @property
    def inactive(self) -> bool:
        """Check if the contact is inactive, i.e., the geoms are not in contact."""
        return self.dist == self.distmax and not self.fromto.any()


def _is_welded_together(model: mujoco.MjModel, geom_id1: int, geom_id2: int) -> bool:
    """Determine if two geoms are part of the same body or are welded together."""
    body1 = model.geom_bodyid[geom_id1]
    body2 = model.geom_bodyid[geom_id2]
    weld1 = model.body_weldid[body1]
    weld2 = model.body_weldid[body2]
    return weld1 == weld2


def _are_geom_bodies_parent_child(model: mujoco.MjModel, geom_id1: int, geom_id2: int) -> bool:
    """Check if the bodies of two geoms have a parent-child relationship."""
    body_id1 = model.geom_bodyid[geom_id1]
    body_id2 = model.geom_bodyid[geom_id2]

    # Get the weld IDs of the bodies.
    body_weldid1 = model.body_weldid[body_id1]
    body_weldid2 = model.body_weldid[body_id2]

    # Get the parent IDs of the welds.
    weld_parent_id1 = model.body_parentid[body_weldid1]
    weld_parent_id2 = model.body_parentid[body_weldid2]

    # Check if one body is the parent of the other.
    cond1 = body_weldid1 == weld_parent_weldid2
    cond2 = body_weldid2 == weld_parent_weldid1
    return cond1 or cond2


def _is_pass_contype_conaffinity_check(model: mujoco.MjModel, geom_id1: int, geom_id2: int) -> bool:
    """Verify if two geoms pass the contype/conaffinity check for collision detection."""
    cond1 = bool(model.geom_contype[geom_id1] & model.geom_conaffinity[geom_id2])
    cond2 = bool(model.geom_contype[geom_id2] & model.geom_conaffinity[geom_id1])
    return cond1 or cond2


class CollisionAvoidanceLimit(Limit):
    """Implements a normal velocity limit between specified geom pairs to avoid collisions.\n\n    Attributes:\n        model: MuJoCo model.\n        geom_pairs: A sequence of collision pairs where each pair consists of two sequences of geoms.\n        gain: Gain factor in (0, 1] that controls the speed of approach towards collision limits.\n        minimum_distance_from_collisions: Minimum distance to maintain between geoms.\n        collision_detection_distance: Distance at which collision avoidance is active.\n        bound_relaxation: Offset applied to the upper bound of collision constraints.\n    """

    def __init__(
        self,
        model: mujoco.MjModel,
        geom_pairs: CollisionPairs,
        gain: float = 0.85,
        minimum_distance_from_collisions: float = 0.005,
        collision_detection_distance: float = 0.01,
        bound_relaxation: float = 0.0,
    ):
        """Initialize the collision avoidance limit with the given parameters.\n\n        Args:\n            model: MuJoCo model.\n            geom_pairs: Sequence of collision pairs specifying which geoms should avoid collisions.\n            gain: Gain factor affecting the speed of approach towards collision limits.\n            minimum_distance_from_collisions: Minimum distance to maintain between geoms.\n            collision_detection_distance: Distance at which collision avoidance is active.\n            bound_relaxation: Offset applied to the upper bound of collision constraints.\n        """
        if not 0.0 < gain <= 1.0:
            raise ValueError("Gain must be in the range (0, 1].")
        if minimum_distance_from_collisions < 0:
            raise ValueError("Minimum distance from collisions must be non-negative.")
        if collision_detection_distance < 0:
            raise ValueError("Collision detection distance must be non-negative.")

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
        """Compute quadratic programming inequalities to avoid collisions.\n\n        Args:\n            configuration: Current robot configuration.\n            dt: Integration timestep.\n\n        Returns:\n            Constraint: Constraint object containing the coefficient matrix and upper bound.\n        """
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
        """Compute the smallest signed distance between two geoms.\n\n        Args:\n            data: MuJoCo data structure.\n            geom1_id: ID of the first geom.\n            geom2_id: ID of the second geom.\n\n        Returns:\n            _Contact: Contact information between the two geoms.\n        """
        fromto = np.empty(6)
        dist = mujoco.mj_geomDistance(
            self.model,
            data,
            geom1_id,
            geom2_id,
            self.collision_detection_distance,
            fromto,
        )
        return _Contact(dist, fromto, geom1_id, geom2_id, self.collision_detection_distance)

    def _compute_contact_normal_jacobian(
        self, data: mujoco.MjData, contact: _Contact
    ) -> np.ndarray:
        """Compute the Jacobian for the normal component of relative linear velocity between two geoms.\n\n        Args:\n            data: MuJoCo data structure.\n            contact: Contact information.\n\n        Returns:\n            np.ndarray: Jacobian matrix.\n        """
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
        """Convert a list of geoms specified by names or IDs to a list of IDs.\n\n        Args:\n            geom_list: Sequence of geom names or IDs.\n\n        Returns:\n            List[int]: List of geom IDs.\n        """
        list_of_int: list[int] = []
        for g in geom_list:
            if isinstance(g, int):
                list_of_int.append(g)
            elif isinstance(g, str):
                geom = self.model.geom(g)
                if geom is None:
                    raise ValueError(f"Geom with name {g} not found in the model.")
                list_of_int.append(geom.id)
            else:
                raise TypeError(f"Geom must be an int or str, got {type(g).__name__}.")
        return list_of_int

    def _collision_pairs_to_geom_id_pairs(self, collision_pairs: CollisionPairs) -> List[tuple[List[int], List[int]]]:
        """Convert collision pairs of names or IDs to collision pairs of IDs.\n\n        Args:\n            collision_pairs: Sequence of collision pairs.\n\n        Returns:\n            List[tuple[List[int], List[int]]]: List of collision pairs with geom IDs.\n        """
        geom_id_pairs = []
        for collision_pair in collision_pairs:
            id_pair_A = self._homogenize_geom_id_list(collision_pair[0])
            id_pair_B = self._homogenize_geom_id_list(collision_pair[1])
            id_pair_A = list(set(id_pair_A))
            id_pair_B = list(set(id_pair_B))
            geom_id_pairs.append((id_pair_A, id_pair_B))
        return geom_id_pairs

    def _construct_geom_id_pairs(self, geom_pairs: CollisionPairs) -> List[tuple[int, int]]:
        """Construct a list of geom ID pairs for collision avoidance.\n\n        Args:\n            geom_pairs: Sequence of collision pairs.\n\n        Returns:\n            List[tuple[int, int]]: List of geom ID pairs.\n        """
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