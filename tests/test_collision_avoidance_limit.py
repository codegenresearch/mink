"""Tests for collision_avoidance_limit.py."""

import itertools

import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import Configuration
from mink.limits import CollisionAvoidanceLimit
from mink.utils import get_body_geom_ids
from mujoco import Contact, compute_contact_normal_jacobian


class TestCollisionAvoidanceLimit(absltest.TestCase):
    """Test collision avoidance limit."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("ur5e_mj_description")

    def setUp(self):
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("home")

    def test_dimensions(self):
        g1 = get_body_geom_ids(self.model, self.model.body("wrist_2_link").id)
        g2 = get_body_geom_ids(self.model, self.model.body("upper_arm_link").id)

        bound_relaxation = -1e-3
        limit = CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=[(g1, g2)],
            bound_relaxation=bound_relaxation,
        )

        # Filter out non-colliding geoms and calculate expected max contacts.
        g1_coll = [
            g
            for g in g1
            if self.model.geom_conaffinity[g] != 0 and self.model.geom_contype[g] != 0
        ]
        g2_coll = [
            g
            for g in g2
            if self.model.geom_conaffinity[g] != 0 and self.model.geom_contype[g] != 0
        ]
        expected_max_contacts = len(list(itertools.product(g1_coll, g2_coll)))

        # Validate the number of max expected contacts.
        self.assertEqual(limit.max_num_contacts, expected_max_contacts)

        # Compute the quadratic programming inequalities.
        G, h = limit.compute_qp_inequalities(self.configuration, 1e-3)

        # Ensure the upper bound is greater than or equal to the relaxation bound.
        self.assertTrue(np.all(h >= bound_relaxation))

        # Validate the dimensions of the inequality constraints.
        self.assertEqual(G.shape, (expected_max_contacts, self.model.nv))
        self.assertEqual(h.shape, (expected_max_contacts,))

    def test_contact_normal_jac_matches_mujoco(self):
        g1 = get_body_geom_ids(self.model, self.model.body("wrist_2_link").id)
        g2 = get_body_geom_ids(self.model, self.model.body("upper_arm_link").id)

        bound_relaxation = -1e-3
        limit = CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=[(g1, g2)],
            bound_relaxation=bound_relaxation,
        )

        # Compute the contact normal Jacobian using the limit method.
        jac_limit = limit.compute_contact_normal_jacobian(self.configuration)

        # Compute the contact normal Jacobian using Mujoco's method.
        mujoco_contacts = [Contact(self.model, i) for i in range(self.model.ncon)]
        jac_mujoco = compute_contact_normal_jacobian(self.model, mujoco_contacts)

        # Validate that the Jacobians match.
        self.assertTrue(np.allclose(jac_limit, jac_mujoco))


if __name__ == "__main__":
    absltest.main()