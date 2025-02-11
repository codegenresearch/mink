"""Tests for collision_avoidance_limit.py."""

import itertools

import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import Configuration
from mink.limits import CollisionAvoidanceLimit
from mink.utils import get_body_geom_ids


class TestCollisionAvoidanceConstraint(absltest.TestCase):
    """Test collision avoidance constraint."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("ur5e_mj_description")

    def setUp(self):
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("home")

    def test_constraint_dimensions(self):
        wrist_geom_ids = get_body_geom_ids(self.model, self.model.body("wrist_2_link").id)
        upper_arm_geom_ids = get_body_geom_ids(self.model, self.model.body("upper_arm_link").id)

        bound_relaxation = -1e-3
        constraint = CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=[(wrist_geom_ids, upper_arm_geom_ids)],
            bound_relaxation=bound_relaxation,
        )

        # Filter out non-colliding geoms and calculate expected max contacts.
        colliding_wrist_geom_ids = [
            g
            for g in wrist_geom_ids
            if self.model.geom_conaffinity[g] != 0 and self.model.geom_contype[g] != 0
        ]
        colliding_upper_arm_geom_ids = [
            g
            for g in upper_arm_geom_ids
            if self.model.geom_conaffinity[g] != 0 and self.model.geom_contype[g] != 0
        ]
        expected_max_contacts = len(list(itertools.product(colliding_wrist_geom_ids, colliding_upper_arm_geom_ids)))

        self.assertEqual(constraint.max_num_contacts, expected_max_contacts)

        G, h = constraint.compute_qp_inequalities(self.configuration, 1e-3)

        # Validate that the upper bound is always greater than or equal to the relaxation bound.
        self.assertTrue(np.all(h >= bound_relaxation))

        # Validate the dimensions of the inequality constraints.
        self.assertEqual(G.shape, (expected_max_contacts, self.model.nv))
        self.assertEqual(h.shape, (expected_max_contacts,))


if __name__ == "__main__":
    absltest.main()