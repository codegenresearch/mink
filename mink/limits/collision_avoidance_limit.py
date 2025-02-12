"""Collision avoidance limit with posture task."""

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
        norm = self.fromto[3:] - self.fromto[:3]
        return norm / (np.linalg.norm(norm) + 1e-9)

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
    body1 = model.geom_bodyid[geom_id1]
    body2 = model.geom_bodyid[geom_id2]
    weld1 = model.body_weldid[body1]
    weld2 = model.body_weldid[body2]
    return weld1 == weld2


def _are_geom_bodies_parent_child(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    body_id1 = model.geom_bodyid[geom_id1]
    body_id2 = model.geom_bodyid[geom_id2]

    weld1 = model.body_weldid[body_id1]
    weld2 = model.body_weldid[body_id2]

    weld_parent_id1 = model.body_parentid[weld1]
    weld_parent_id2 = model.body_parentid[weld2]

    weld_parent_weldid1 = model.body_weldid[weld_parent_id1]
    weld_parent_weldid2 = model.body_weldid[weld_parent_id2]

    cond1 = weld1 == weld_parent_weldid2
    cond2 = weld2 == weld_parent_weldid1
    return cond1 or cond2


def _is_pass_contype_conaffinity_check(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    cond1 = bool(model.geom_contype[geom_id1] & model.geom_conaffinity[geom_id2])
    cond2 = bool(model.geom_contype[geom_id2] & model.geom_conaffinity[geom_id1])
    return cond1 or cond2


class CollisionAvoidanceLimit(Limit):
    """Normal velocity limit between geom pairs with posture task."""

    def __init__(
        self,
        model: mujoco.MjModel,
        geom_pairs: CollisionPairs,
        gain: float = 0.85,
        min_dist: float = 0.005,
        detect_dist: float = 0.01,
        bound_relax: float = 0.0,
        posture_task: bool = False,
    ):
        self.model = model
        self.gain = gain
        self.min_dist = min_dist
        self.detect_dist = detect_dist
        self.bound_relax = bound_relax
        self.geom_id_pairs = self._construct_geom_id_pairs(geom_pairs)
        self.max_contacts = len(self.geom_id_pairs)
        self.posture_task = posture_task

    def compute_qp_inequalities(
        self,
        config: Configuration,
        dt: float,
    ) -> Constraint:
        upper_bounds = np.full((self.max_contacts,), np.inf)
        coeff_matrix = np.zeros((self.max_contacts, self.model.nv))
        for idx, (geom1_id, geom2_id) in enumerate(self.geom_id_pairs):
            contact = self._compute_min_contact(config.data, geom1_id, geom2_id)
            if contact.inactive:
                continue
            hi_dist = contact.dist
            if hi_dist > self.min_dist:
                dist = hi_dist - self.min_dist
                upper_bounds[idx] = (self.gain * dist / dt) + self.bound_relax
            else:
                upper_bounds[idx] = self.bound_relax
            jac = compute_contact_normal_jacobian(self.model, config.data, contact)
            coeff_matrix[idx] = -jac

        if self.posture_task:
            posture_bounds, posture_coeffs = self._add_posture_task(config)
            upper_bounds = np.concatenate([upper_bounds, posture_bounds])
            coeff_matrix = np.vstack([coeff_matrix, posture_coeffs])

        return Constraint(G=coeff_matrix, h=upper_bounds)

    def _compute_min_contact(
        self, data: mujoco.MjData, geom1_id: int, geom2_id: int
    ) -> Contact:
        fromto = np.empty(6)
        dist = mujoco.mj_geomDistance(
            self.model,
            data,
            geom1_id,
            geom2_id,
            self.detect_dist,
            fromto,
        )
        return Contact(dist, fromto, geom1_id, geom2_id, self.detect_dist)

    def _homogenize_geom_ids(self, geom_list: GeomSequence) -> List[int]:
        geom_ids = []
        for g in geom_list:
            if isinstance(g, int):
                geom_ids.append(g)
            elif isinstance(g, str):
                geom_ids.append(self.model.geom(g).id)
        return geom_ids

    def _collision_pairs_to_geom_id_pairs(self, collision_pairs: CollisionPairs):
        geom_id_pairs = []
        for pair in collision_pairs:
            id_pair_A = self._homogenize_geom_ids(pair[0])
            id_pair_B = self._homogenize_geom_ids(pair[1])
            id_pair_A = list(set(id_pair_A))
            id_pair_B = list(set(id_pair_B))
            geom_id_pairs.append((id_pair_A, id_pair_B))
        return geom_id_pairs

    def _construct_geom_id_pairs(self, geom_pairs):
        geom_id_pairs = []
        for id_pair in self._collision_pairs_to_geom_id_pairs(geom_pairs):
            for geom_a, geom_b in itertools.product(*id_pair):
                if (not _is_welded_together(self.model, geom_a, geom_b) and
                    not _are_geom_bodies_parent_child(self.model, geom_a, geom_b) and
                    _is_pass_contype_conaffinity_check(self.model, geom_a, geom_b)):
                    geom_id_pairs.append((min(geom_a, geom_b), max(geom_a, geom_b)))
        return geom_id_pairs

    def _add_posture_task(self, config: Configuration):
        # Example posture task: maintain a specific joint configuration
        target_qpos = np.array([0, -1.57, 1.57, 0, -1.57, 0])  # Example target
        Kp = 100.0  # Proportional gain
        error = config.q - target_qpos
        upper_bounds = Kp * error
        coeff_matrix = np.eye(self.model.nv)
        return upper_bounds, coeff_matrix