"""Tests for collision_avoidance_limit.py."""

import itertools

import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import Configuration
from mink.limits import CollisionAvoidanceLimit
from mink.utils import get_body_geom_ids


class CollisionAvoidanceLimitTests(absltest.TestCase):
    """Tests for the CollisionAvoidanceLimit class."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("ur5e_mj_description")

    def setUp(self):
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("home")

    def test_collision_avoidance_limit_dimensions(self):
        geom_ids_wrist_2 = get_body_geom_ids(self.model, self.model.body("wrist_2_link").id)
        geom_ids_upper_arm = get_body_geom_ids(self.model, self.model.body("upper_arm_link").id)

        bound_relaxation = -1e-3
        limit = CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=[(geom_ids_wrist_2, geom_ids_upper_arm)],
            bound_relaxation=bound_relaxation,
        )

        expected_max_contacts = self._calculate_expected_max_contacts(geom_ids_wrist_2, geom_ids_upper_arm)
        self.assertEqual(limit.max_num_contacts, expected_max_contacts)

        G, h = limit.compute_qp_inequalities(self.configuration, 1e-3)

        self._assert_upper_bound_greater_equal_relaxation(h, bound_relaxation)
        self._assert_inequality_constraint_dimensions(G, h, expected_max_contacts)

    def _calculate_expected_max_contacts(self, geom_ids_1, geom_ids_2):
        colliding_geoms_1 = self._filter_colliding_geoms(geom_ids_1)
        colliding_geoms_2 = self._filter_colliding_geoms(geom_ids_2)
        return len(list(itertools.product(colliding_geoms_1, colliding_geoms_2)))

    def _filter_colliding_geoms(self, geom_ids):
        return [
            g
            for g in geom_ids
            if self.model.geom_conaffinity[g] != 0 and self.model.geom_contype[g] != 0
        ]

    def _assert_upper_bound_greater_equal_relaxation(self, h, bound_relaxation):
        self.assertTrue(np.all(h >= bound_relaxation))

    def _assert_inequality_constraint_dimensions(self, G, h, expected_max_contacts):
        self.assertEqual(G.shape, (expected_max_contacts, self.model.nv))
        self.assertEqual(h.shape, (expected_max_contacts,))


if __name__ == "__main__":
    absltest.main()