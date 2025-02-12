from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Union

import mujoco
import numpy as np

from ..configuration import Configuration
from .limit import Constraint, Limit

# Type aliases for better readability.
Geom = Union[int, str]
GeomSequence = Sequence[Geom]
CollisionPair = tuple[GeomSequence, GeomSequence]
CollisionPairs = Sequence[CollisionPair]


@dataclass(frozen=True)
class _Contact:
    """Dataclass to store contact information between two geoms.\n\n    Attributes:\n        dist: The signed distance between the two geoms.\n        fromto: A 6-element array representing the two closest points between the geoms.\n        geom1: The ID of the first geom.\n        geom2: The ID of the second geom.\n        distmax: The maximum allowed distance between the geoms.\n    """

    dist: float
    fromto: np.ndarray
    geom1: int
    geom2: int
    distmax: float

    @property
    def normal(self) -> np.ndarray:
        """Compute the unit normal vector pointing from geom1 to geom2.\n\n        Returns:\n            A unit normal vector.\n        """
        normal = self.fromto[3:] - self.fromto[:3]
        return normal / (np.linalg.norm(normal) + 1e-9)

    @property
    def inactive(self) -> bool:
        """Check if the contact is inactive.\n\n        A contact is inactive if the distance between the geoms is equal to the maximum\n        allowed distance and the fromto array is zero.\n\n        Returns:\n            True if the contact is inactive, False otherwise.\n        """
        return self.dist == self.distmax and not self.fromto.any()


def _is_welded_together(model: mujoco.MjModel, geom_id1: int, geom_id2: int) -> bool:
    """Determine if two geoms are part of the same body or are welded together.\n\n    Args:\n        model: The MuJoCo model.\n        geom_id1: The ID of the first geom.\n        geom_id2: The ID of the second geom.\n\n    Returns:\n        True if the geoms are part of the same body or are welded together, False otherwise.\n    """
    body1 = model.geom_bodyid[geom_id1]
    body2 = model.geom_bodyid[geom_id2]
    weld1 = model.body_weldid[body1]
    weld2 = model.body_weldid[body2]
    return weld1 == weld2


def _are_geom_bodies_parent_child(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    """Check if the bodies of two geoms have a parent-child relationship.\n\n    Args:\n        model: The MuJoCo model.\n        geom_id1: The ID of the first geom.\n        geom_id2: The ID of the second geom.\n\n    Returns:\n        True if the bodies have a parent-child relationship, False otherwise.\n    """
    body_id1 = model.geom_bodyid[geom_id1]
    body_id2 = model.geom_bodyid[geom_id2]

    # Get the weld IDs for both bodies.
    body_weldid1 = model.body_weldid[body_id1]
    body_weldid2 = model.body_weldid[body_id2]

    # Get the parent IDs of the welds.
    weld_parent_id1 = model.body_parentid[body_weldid1]
    weld_parent_id2 = model.body_parentid[body_weldid2]

    # Check if one weld is the parent of the other.
    return body_weldid1 == weld_parent_id2 or body_weldid2 == weld_parent_id1


def _is_pass_contype_conaffinity_check(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    """Check if two geoms pass the contype/conaffinity collision filter check.\n\n    Args:\n        model: The MuJoCo model.\n        geom_id1: The ID of the first geom.\n        geom_id2: The ID of the second geom.\n\n    Returns:\n        True if the geoms pass the collision filter check, False otherwise.\n    """
    return bool(model.geom_contype[geom_id1] & model.geom_conaffinity[geom_id2]) or bool(
        model.geom_contype[geom_id2] & model.geom_conaffinity[geom_id1]
    )


class CollisionAvoidanceLimit(Limit):
    """Implements a normal velocity limit between specified geom pairs to avoid collisions.\n\n    Attributes:\n        model: The MuJoCo model.\n        geom_pairs: A sequence of collision pairs where each pair consists of two sequences\n            of geom identifiers (either names or IDs). The mapper computes joint velocities\n            to avoid collisions between each geom in the first sequence and each geom in the\n            second sequence. Self-collision can be enabled by providing the same sequence\n            for both fields of a pair.\n        gain: A gain factor in the range (0, 1] that controls how quickly the geoms can\n            approach each other. Lower values are safer but may reduce the speed of movement.\n        minimum_distance_from_collisions: The minimum distance to maintain between any two\n            geoms. Negative values allow penetration by the specified amount.\n        collision_detection_distance: The distance at which collision avoidance becomes active.\n            Larger values enable earlier detection but may increase computational cost.\n        bound_relaxation: An offset applied to the upper bound of each collision avoidance\n            constraint.\n    """

    def __init__(
        self,
        model: mujoco.MjModel,
        geom_pairs: CollisionPairs,
        gain: float = 0.85,
        minimum_distance_from_collisions: float = 0.005,
        collision_detection_distance: float = 0.01,
        bound_relaxation: float = 0.0,
    ):
        """Initialize the collision avoidance limit.\n\n        Args:\n            model: The MuJoCo model.\n            geom_pairs: A sequence of collision pairs where each pair consists of two sequences\n                of geom identifiers (either names or IDs).\n            gain: A gain factor in the range (0, 1] that controls how quickly the geoms can\n                approach each other.\n            minimum_distance_from_collisions: The minimum distance to maintain between any two\n                geoms.\n            collision_detection_distance: The distance at which collision avoidance becomes active.\n            bound_relaxation: An offset applied to the upper bound of each collision avoidance\n                constraint.\n        """
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
        """Compute the quadratic programming inequality constraints for collision avoidance.\n\n        Args:\n            configuration: The current robot configuration.\n            dt: The time step for the simulation.\n\n        Returns:\n            A Constraint object containing the coefficient matrix G and the upper bound h.\n        """
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
        """Compute the smallest signed distance between two geoms.\n\n        Args:\n            data: The MuJoCo data structure.\n            geom1_id: The ID of the first geom.\n            geom2_id: The ID of the second geom.\n\n        Returns:\n            A _Contact object containing the contact information.\n        """
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
        """Compute the Jacobian mapping joint velocities to the normal component of the relative\n        linear velocity between two geoms.\n\n        The Jacobian-velocity relationship is given by:\n\n            J dq = n^T (v_2 - v_1)\n\n        where:\n        - J is the Jacobian matrix.\n        - dq is the joint velocity vector.\n        - n^T is the normal vector (from geom1 to geom2) transposed.\n        - v_1, v_2 are the linear velocities of the closest points on geom1 and geom2, respectively.\n\n        Args:\n            data: The MuJoCo data structure.\n            contact: A _Contact object containing the contact information.\n\n        Returns:\n            The Jacobian matrix.\n        """
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
        """Convert a list of geoms, specified by either ID or name, into a list of IDs.\n\n        Args:\n            geom_list: A sequence of geom identifiers (either names or IDs).\n\n        Returns:\n            A list of geom IDs.\n        """
        list_of_int: list[int] = []
        for g in geom_list:
            if isinstance(g, int):
                list_of_int.append(g)
            else:
                assert isinstance(g, str)
                list_of_int.append(self.model.geom(g).id)
        return list_of_int

    def _collision_pairs_to_geom_id_pairs(self, collision_pairs: CollisionPairs) -> List[tuple[List[int], List[int]]]:
        """Convert a list of collision pairs, with geoms specified by either ID or name,\n        into a list of pairs of geom IDs.\n\n        Args:\n            collision_pairs: A sequence of collision pairs.\n\n        Returns:\n            A list of pairs of geom IDs.\n        """
        geom_id_pairs = []
        for collision_pair in collision_pairs:
            id_pair_A = self._homogenize_geom_id_list(collision_pair[0])
            id_pair_B = self._homogenize_geom_id_list(collision_pair[1])
            id_pair_A = list(set(id_pair_A))
            id_pair_B = list(set(id_pair_B))
            geom_id_pairs.append((id_pair_A, id_pair_B))
        return geom_id_pairs

    def _construct_geom_id_pairs(self, geom_pairs: CollisionPairs) -> List[tuple[int, int]]:
        """Generate all possible geom-geom collision pairs based on the provided geom pairs,\n        excluding pairs that are part of the same body, have a parent-child relationship, or\n        fail the contype-conaffinity check.\n\n        Args:\n            geom_pairs: A sequence of collision pairs.\n\n        Returns:\n            A list of geom-geom collision pairs as tuples of IDs.\n        """
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