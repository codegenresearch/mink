"""Collision avoidance limit with posture tasks and simplified joint velocity handling."""

import itertools
from dataclasses import dataclass
from typing import List, Sequence, Union

import mujoco
import numpy as np

from ..configuration import Configuration
from .limit import Constraint, Limit
from .exceptions import LimitDefinitionError

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
    """Returns true if the geoms are part of the same body, or if their bodies are\n    welded together."""
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

    # body_weldid is the ID of the body's weld.\n    body_weldid1 = model.body_weldid[body_id1]\n    body_weldid2 = model.body_weldid[body_id2]\n\n    # weld_parent_id is the ID of the parent of the body's weld.
    weld_parent_id1 = model.body_parentid[body_weldid1]
    weld_parent_id2 = model.body_parentid[body_weldid2]

    # weld_parent_weldid is the weld ID of the parent of the body's weld.\n    weld_parent_weldid1 = model.body_weldid[weld_parent_id1]\n    weld_parent_weldid2 = model.body_weldid[weld_parent_id2]\n\n    cond1 = body_weldid1 == weld_parent_weldid2\n    cond2 = body_weldid2 == weld_parent_weldid1\n    return cond1 or cond2\n\n\ndef _is_pass_contype_conaffinity_check(\n    model: mujoco.MjModel, geom_id1: int, geom_id2: int\n) -> bool:\n    """Returns true if the geoms pass the contype/conaffinity check."""\n    cond1 = bool(model.geom_contype[geom_id1] & model.geom_conaffinity[geom_id2])\n    cond2 = bool(model.geom_contype[geom_id2] & model.geom_conaffinity[geom_id1])\n    return cond1 or cond2\n\n\nclass CollisionAvoidanceLimit(Limit):\n    """Normal velocity limit between geom pairs with posture tasks and simplified joint velocity handling.\n\n    Attributes:\n        model: MuJoCo model.\n        geom_pairs: Set of collision pairs in which to perform active collision\n            avoidance. A collision pair is defined as a pair of geom groups. A geom\n            group is a set of geom names. For each collision pair, the solver will\n            attempt to compute joint velocities that avoid collisions between every\n            geom in the first geom group with every geom in the second geom group.\n            Self collision is achieved by adding a collision pair with the same\n            geom group in both pair fields.\n        gain: Gain factor in (0, 1] that determines how fast the geoms are\n            allowed to move towards each other at each iteration. Smaller values\n            are safer but may make the geoms move slower towards each other.\n        minimum_distance_from_collisions: The minimum distance to leave between\n            any two geoms. A negative distance allows the geoms to penetrate by\n            the specified amount.\n        collision_detection_distance: The distance between two geoms at which the\n            active collision avoidance limit will be active. A large value will\n            cause collisions to be detected early, but may incur high computational\n            cost. A negative value will cause the geoms to be detected only after\n            they penetrate by the specified amount.\n        bound_relaxation: An offset on the upper bound of each collision avoidance\n            constraint.\n    """\n\n    def __init__(\n        self,\n        model: mujoco.MjModel,\n        geom_pairs: CollisionPairs,\n        gain: float = 0.85,\n        minimum_distance_from_collisions: float = 0.005,\n        collision_detection_distance: float = 0.01,\n        bound_relaxation: float = 0.0,\n    ):\n        """Initialize collision avoidance limit.\n\n        Args:\n            model: MuJoCo model.\n            geom_pairs: Set of collision pairs in which to perform active collision\n                avoidance. A collision pair is defined as a pair of geom groups. A geom\n                group is a set of geom names. For each collision pair, the mapper will\n                attempt to compute joint velocities that avoid collisions between every\n                geom in the first geom group with every geom in the second geom group.\n                Self collision is achieved by adding a collision pair with the same\n                geom group in both pair fields.\n            gain: Gain factor in (0, 1] that determines how fast the geoms are\n                allowed to move towards each other at each iteration. Smaller values\n                are safer but may make the geoms move slower towards each other.\n            minimum_distance_from_collisions: The minimum distance to leave between\n                any two geoms. A negative distance allows the geoms to penetrate by\n                the specified amount.\n            collision_detection_distance: The distance between two geoms at which the\n                active collision avoidance limit will be active. A large value will\n                cause collisions to be detected early, but may incur high computational\n                cost. A negative value will cause the geoms to be detected only after\n                they penetrate by the specified amount.\n            bound_relaxation: An offset on the upper bound of each collision avoidance\n                constraint.\n        """\n        if not 0.0 < gain <= 1.0:\n            raise LimitDefinitionError(\n                f"{self.__class__.__name__} gain must be in the range (0, 1]"\n            )\n\n        self.model = model\n        self.gain = gain\n        self.minimum_distance_from_collisions = minimum_distance_from_collisions\n        self.collision_detection_distance = collision_detection_distance\n        self.bound_relaxation = bound_relaxation\n        self.geom_id_pairs = self._construct_geom_id_pairs(geom_pairs)\n        self.max_num_contacts = len(self.geom_id_pairs)\n\n    def compute_qp_inequalities(\n        self,\n        configuration: Configuration,\n        dt: float,\n    ) -> Constraint:\n        upper_bound = np.full((self.max_num_contacts,), np.inf)\n        coefficient_matrix = np.zeros((self.max_num_contacts, self.model.nv))\n        for idx, (geom1_id, geom2_id) in enumerate(self.geom_id_pairs):\n            contact = self._compute_contact_with_minimum_distance(\n                configuration.data, geom1_id, geom2_id\n            )\n            if contact.inactive:\n                continue\n            hi_bound_dist = contact.dist\n            if hi_bound_dist > self.minimum_distance_from_collisions:\n                dist = hi_bound_dist - self.minimum_distance_from_collisions\n                upper_bound[idx] = (self.gain * dist) + self.bound_relaxation\n            else:\n                upper_bound[idx] = self.bound_relaxation\n            jac = compute_contact_normal_jacobian(\n                self.model, configuration.data, contact\n            )\n            coefficient_matrix[idx] = -jac\n        return Constraint(G=coefficient_matrix, h=upper_bound)\n\n    # Private methods.\n\n    def _compute_contact_with_minimum_distance(\n        self, data: mujoco.MjData, geom1_id: int, geom2_id: int\n    ) -> Contact:\n        """Returns the smallest signed distance between a geom pair."""\n        fromto = np.empty(6)\n        dist = mujoco.mj_geomDistance(\n            self.model,\n            data,\n            geom1_id,\n            geom2_id,\n            self.collision_detection_distance,\n            fromto,\n        )\n        return Contact(\n            dist, fromto, geom1_id, geom2_id, self.collision_detection_distance\n        )\n\n    def _homogenize_geom_id_list(self, geom_list: GeomSequence) -> List[int]:\n        """Take a heterogeneous list of geoms (specified via ID or name) and return\n        a homogenous list of IDs (int)."""\n        list_of_int: list[int] = []\n        for g in geom_list:\n            if isinstance(g, int):\n                list_of_int.append(g)\n            else:\n                assert isinstance(g, str)\n                list_of_int.append(self.model.geom(g).id)\n        return list_of_int\n\n    def _collision_pairs_to_geom_id_pairs(self, collision_pairs: CollisionPairs):\n        geom_id_pairs = []\n        for collision_pair in collision_pairs:\n            id_pair_A = self._homogenize_geom_id_list(collision_pair[0])\n            id_pair_B = self._homogenize_geom_id_list(collision_pair[1])\n            id_pair_A = list(set(id_pair_A))\n            id_pair_B = list(set(id_pair_B))\n            geom_id_pairs.append((id_pair_A, id_pair_B))\n        return geom_id_pairs\n\n    def _construct_geom_id_pairs(self, geom_pairs):\n        """Returns a set of geom ID pairs for all possible geom-geom collisions.\n\n        The contacts are added based on the following heuristics:\n            1) Geoms that are part of the same body or weld are not included.\n            2) Geoms where the body of one geom is a parent of the body of the other\n                geom are not included.\n            3) Geoms that fail the contype-conaffinity check are ignored.\n\n        Note:\n            1) If two bodies are kinematically welded together (no joints between them)\n                they are considered to be the same body within this function.\n        """\n        geom_id_pairs = []\n        for id_pair in self._collision_pairs_to_geom_id_pairs(geom_pairs):\n            for geom_a, geom_b in itertools.product(*id_pair):\n                weld_body_cond = not _is_welded_together(self.model, geom_a, geom_b)\n                parent_child_cond = not _are_geom_bodies_parent_child(\n                    self.model, geom_a, geom_b\n                )\n                contype_conaffinity_cond = _is_pass_contype_conaffinity_check(\n                    self.model, geom_a, geom_b\n                )\n                if weld_body_cond and parent_child_cond and contype_conaffinity_cond:\n                    geom_id_pairs.append((min(geom_a, geom_b), max(geom_a, geom_b)))\n        return geom_id_pairs