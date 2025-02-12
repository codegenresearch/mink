"""Tests for configuration_limit.py."""

import itertools

import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import Configuration
from mink.limits import CollisionAvoidanceLimit
from mink.utils import get_collidable_geom_ids


class TestCollisionAvoidanceConstraints(absltest.TestCase):
    """Tests for collision avoidance constraints."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("ur5e_mj_description")

    def setUp(self):
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("home")

    def test_collision_constraint_dimensions(self):
        wrist_geom_ids = get_collidable_geom_ids(self.model, "wrist_2_link")
        upper_arm_geom_ids = get_collidable_geom_ids(self.model, "upper_arm_link")

        bound_relaxation = -1e-3
        limit = CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=[(wrist_geom_ids, upper_arm_geom_ids)],
            bound_relaxation=bound_relaxation,
        )

        # Calculate the expected maximum number of contacts.
        expected_max_contacts = len(list(itertools.product(wrist_geom_ids, upper_arm_geom_ids)))
        self.assertEqual(limit.max_num_contacts, expected_max_contacts)

        # Compute the quadratic programming inequalities.
        G, h = limit.compute_qp_inequalities(self.configuration, 1e-3)

        # Validate the upper bound is greater than or equal to the relaxation bound.
        self.assertTrue(np.all(h >= bound_relaxation))

        # Validate the dimensions of the inequality constraints.
        self.assertEqual(G.shape, (expected_max_contacts, self.model.nv))
        self.assertEqual(h.shape, (expected_max_contacts,))

    def test_invalid_geom_pair_raises_error(self):
        invalid_geom_pair = ([0, 1], [2, 3, 4])
        with self.assertRaises(ValueError) as cm:
            CollisionAvoidanceLimit(
                model=self.model,
                geom_pairs=[invalid_geom_pair],
                bound_relaxation=-1e-3,
            )
        expected_error_message = "Each geom pair must have the same number of elements."
        self.assertEqual(str(cm.exception), expected_error_message)


if __name__ == "__main__":
    absltest.main()