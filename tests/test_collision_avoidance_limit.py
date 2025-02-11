"""Tests for collision_avoidance_limit.py."""

import itertools

import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import Configuration
from mink.limits import CollisionAvoidanceLimit
from mink.utils import get_body_geom_ids


class TestCollisionAvoidanceLimit(absltest.TestCase):
    """Test collision avoidance limit."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("ur5e_mj_description")

    def setUp(self):
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("home")

    def test_dimensions(self):
        """Test the dimensions of the collision avoidance limit."""
        g1 = get_body_geom_ids(self.model, self.model.body("wrist_2_link").id)
        g2 = get_body_geom_ids(self.model, self.model.body("upper_arm_link").id)

        bound_relaxation = -1e-3
        limit = CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=[(g1, g2)],
            bound_relaxation=bound_relaxation,
        )

        # Filter out non-colliding geoms
        g1_coll = self._filter_colliding_geoms(g1)
        g2_coll = self._filter_colliding_geoms(g2)
        expected_max_contacts = len(list(itertools.product(g1_coll, g2_coll)))

        # Check the number of max expected contacts
        self.assertEqual(limit.max_num_contacts, expected_max_contacts)

        G, h = limit.compute_qp_inequalities(self.configuration, 1e-3)

        # The upper bound should always be >= relaxation bound
        self.assertTrue(np.all(h >= bound_relaxation))

        # Check the inequality constraint dimensions
        self.assertEqual(G.shape, (expected_max_contacts, self.model.nv))
        self.assertEqual(h.shape, (expected_max_contacts,))

    def test_contact_normal_jacobian(self):
        """Test the contact normal Jacobian."""
        g1 = get_body_geom_ids(self.model, self.model.body("wrist_2_link").id)
        g2 = get_body_geom_ids(self.model, self.model.body("upper_arm_link").id)

        bound_relaxation = -1e-3
        limit = CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=[(g1, g2)],
            bound_relaxation=bound_relaxation,
        )

        Jn = limit.contact_normal_jacobian(self.configuration)
        nv = self.model.nv
        expected_shape = (limit.max_num_contacts, nv)

        # Check the shape of the contact normal Jacobian
        self.assertEqual(Jn.shape, expected_shape)

    def _filter_colliding_geoms(self, geom_ids):
        """Filter out non-colliding geoms based on conaffinity and contype."""
        return [
            g
            for g in geom_ids
            if self.model.geom_conaffinity[g] != 0 and self.model.geom_contype[g] != 0
        ]


if __name__ == "__main__":
    absltest.main()