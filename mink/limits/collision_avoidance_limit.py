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


def compute_contact_normal_jacobian(model, data, contact):
    """Compute Jacobian for contact normal."""
    geom1_body = model.geom_bodyid[contact.geom1]
    geom2_body = model.geom_bodyid[contact.geom2]
    geom1_contact_pos = contact.fromto[:3]
    geom2_contact_pos = contact.fromto[3:]
    jac2 = np.empty((3, model.nv))
    mujoco.mj_jac(model, data, jac2, None, geom2_contact_pos, geom2_body)
    jac1 = np.empty((3, model.nv))
    mujoco.mj_jac(model, data, jac1, None, geom1_contact_pos, geom1_body)
    return contact.normal @ (jac2 - jac1)


def is_welded_together(model, geom_id1, geom_id2):
    """Check if geoms are welded together."""
    body1 = model.geom_bodyid[geom_id1]
    body2 = model.geom_bodyid[geom_id2]
    weld1 = model.body_weldid[body1]
    weld2 = model.body_weldid[body2]
    return weld1 == weld2


def are_geom_bodies_parent_child(model, geom_id1, geom_id2):
    """Check if geom bodies have a parent-child relationship."""
    body_id1 = model.geom_bodyid[geom_id1]
    body_id2 = model.geom_bodyid[geom_id2]
    weld_parent_id1 = model.body_parentid[model.body_weldid[body_id1]]
    weld_parent_id2 = model.body_parentid[model.body_weldid[body_id2]]
    return weld_parent_id1 == model.body_weldid[body_id2] or weld_parent_id2 == model.body_weldid[body_id1]


def is_pass_contype_conaffinity_check(model, geom_id1, geom_id2):
    """Check if geoms pass contype/conaffinity check."""
    return bool(model.geom_contype[geom_id1] & model.geom_conaffinity[geom_id2]) or bool(
        model.geom_contype[geom_id2] & model.geom_conaffinity[geom_id1]
    )


def validate_se3_transformations(mocap_data):
    """Validate SE3 transformations from mocap data."""
    for pose in mocap_data:
        rotation_matrix = pose[:3, :3]
        if not np.allclose(rotation_matrix.T @ rotation_matrix, np.eye(3)):
            return False
        if not np.isclose(np.linalg.det(rotation_matrix), 1.0):
            return False
    return True


class CollisionAvoidanceLimit(Limit):
    """Normal velocity limit between geom pairs with enhanced testing and SE3 validation."""

    def __init__(
        self,
        model,
        geom_pairs,
        gain=0.85,
        min_dist=0.005,
        detect_dist=0.01,
        bound_relax=0.0,
    ):
        """Initialize collision avoidance limit."""
        self.model = model
        self.gain = gain
        self.min_dist = min_dist
        self.detect_dist = detect_dist
        self.bound_relax = bound_relax
        self.geom_id_pairs = self._construct_geom_id_pairs(geom_pairs)
        self.max_contacts = len(self.geom_id_pairs)

    def compute_qp_inequalities(self, config, dt):
        """Compute quadratic programming inequalities."""
        upper_bound = np.full((self.max_contacts,), np.inf)
        coeff_matrix = np.zeros((self.max_contacts, self.model.nv))
        for idx, (geom1_id, geom2_id) in enumerate(self.geom_id_pairs):
            contact = self._compute_contact(config.data, geom1_id, geom2_id)
            if contact.inactive:
                continue
            hi_bound_dist = contact.dist
            if hi_bound_dist > self.min_dist:
                dist = hi_bound_dist - self.min_dist
                upper_bound[idx] = (self.gain * dist / dt) + self.bound_relax
            else:
                upper_bound[idx] = self.bound_relax
            jac = compute_contact_normal_jacobian(self.model, config.data, contact)
            coeff_matrix[idx] = -jac
        return Constraint(G=coeff_matrix, h=upper_bound)

    def _compute_contact(self, data, geom1_id, geom2_id):
        """Compute contact with minimum distance."""
        fromto = np.empty(6)
        dist = mujoco.mj_geomDistance(
            self.model, data, geom1_id, geom2_id, self.detect_dist, fromto
        )
        return Contact(dist, fromto, geom1_id, geom2_id, self.detect_dist)

    def _homogenize_geom_ids(self, geom_list):
        """Convert geom list to IDs."""
        return [g if isinstance(g, int) else self.model.geom(g).id for g in geom_list]

    def _collision_pairs_to_geom_id_pairs(self, collision_pairs):
        """Convert collision pairs to geom ID pairs."""
        geom_id_pairs = []
        for pair in collision_pairs:
            id_pair_A = self._homogenize_geom_ids(pair[0])
            id_pair_B = self._homogenize_geom_ids(pair[1])
            geom_id_pairs.append((list(set(id_pair_A)), list(set(id_pair_B))))
        return geom_id_pairs

    def _construct_geom_id_pairs(self, geom_pairs):
        """Construct geom ID pairs for collision avoidance."""
        geom_id_pairs = []
        for id_pair in self._collision_pairs_to_geom_id_pairs(geom_pairs):
            for geom_a, geom_b in itertools.product(*id_pair):
                if (
                    not is_welded_together(self.model, geom_a, geom_b)
                    and not are_geom_bodies_parent_child(self.model, geom_a, geom_b)
                    and is_pass_contype_conaffinity_check(self.model, geom_a, geom_b)
                ):
                    geom_id_pairs.append((min(geom_a, geom_b), max(geom_a, geom_b)))
        return geom_id_pairs