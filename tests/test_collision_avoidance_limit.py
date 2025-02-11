"""Tests for configuration_limit.py."""

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
        wrist_2_geom_ids = get_body_geom_ids(self.model, self.model.body("wrist_2_link").id)
        upper_arm_geom_ids = get_body_geom_ids(self.model, self.model.body("upper_arm_link").id)

        bound_relaxation = -1e-3
        limit = CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=[(wrist_2_geom_ids, upper_arm_geom_ids)],
            bound_relaxation=bound_relaxation,
        )

        # Filter out non-colliding geoms and calculate expected max contacts.
        wrist_2_colliding_geoms = [
            g
            for g in wrist_2_geom_ids
            if self.model.geom_conaffinity[g] != 0 and self.model.geom_contype[g] != 0
        ]
        upper_arm_colliding_geoms = [
            g
            for g in upper_arm_geom_ids
            if self.model.geom_conaffinity[g] != 0 and self.model.geom_contype[g] != 0
        ]
        expected_max_contacts = len(list(itertools.product(wrist_2_colliding_geoms, upper_arm_colliding_geoms)))
        self.assertEqual(limit.max_num_contacts, expected_max_contacts)

        G, h = limit.compute_qp_inequalities(self.configuration, 1e-3)

        # Verify that the upper bound is always greater than or equal to the relaxation bound.
        self.assertTrue(np.all(h >= bound_relaxation))

        # Check the dimensions of the inequality constraints.
        self.assertEqual(G.shape, (expected_max_contacts, self.model.nv))
        self.assertEqual(h.shape, (expected_max_contacts,))

    def test_invalid_geom_pair(self):
        # Test with an invalid geom pair that does not exist in the model.
        invalid_geom_pair = ([9999], [9999])
        with self.assertRaises(ValueError) as cm:
            CollisionAvoidanceLimit(
                model=self.model,
                geom_pairs=[invalid_geom_pair],
                bound_relaxation=-1e-3,
            )
        self.assertIn("Invalid geom ID", str(cm.exception))

    def test_no_colliding_geoms(self):
        # Test with geom pairs that have no colliding geoms.
        g1 = get_body_geom_ids(self.model, self.model.body("wrist_2_link").id)
        g2 = get_body_geom_ids(self.model, self.model.body("upper_arm_link").id)

        # Temporarily set all geoms to non-colliding.
        original_conaffinity = self.model.geom_conaffinity.copy()
        original_contype = self.model.geom_contype.copy()
        self.model.geom_conaffinity[:] = 0
        self.model.geom_contype[:] = 0

        limit = CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=[(g1, g2)],
            bound_relaxation=-1e-3,
        )

        self.assertEqual(limit.max_num_contacts, 0)

        # Restore original values.
        self.model.geom_conaffinity = original_conaffinity
        self.model.geom_contype = original_contype


if __name__ == "__main__":
    absltest.main()