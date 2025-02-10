"""Tests for collision_avoidance_limit.py."""

import itertools

import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description
import mujoco

from mink import Configuration
from mink.limits import CollisionAvoidanceLimit
from mink.utils import get_body_geom_ids


class TestCollisionAvoidanceLimit(absltest.TestCase):
    """Tests for the CollisionAvoidanceLimit class."""

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
        g1_coll = [g for g in g1 if self.model.geom_conaffinity[g] != 0 and self.model.geom_contype[g] != 0]
        g2_coll = [g for g in g2 if self.model.geom_conaffinity[g] != 0 and self.model.geom_contype[g] != 0]

        # Calculate expected maximum number of contacts
        expected_max_num_contacts = len(list(itertools.product(g1_coll, g2_coll)))
        self.assertEqual(limit.max_num_contacts, expected_max_num_contacts)

        G, h = limit.compute_qp_inequalities(self.configuration, 1e-3)

        # Check that the upper bound is always greater than or equal to the relaxation bound
        self.assertTrue(np.all(h >= bound_relaxation), "h values should be greater than or equal to the relaxation bound")

        # Check that the inequality constraint dimensions are valid
        self.assertEqual(G.shape, (expected_max_num_contacts, self.model.nv), "G shape mismatch")
        self.assertEqual(h.shape, (expected_max_num_contacts,), "h shape mismatch")

    def test_contact_normal_jac_matches_mujoco(self):
        """Test that the contact normal Jacobian matches MuJoCo's implementation."""
        g1 = get_body_geom_ids(self.model, self.model.body("wrist_2_link").id)
        g2 = get_body_geom_ids(self.model, self.model.body("upper_arm_link").id)

        bound_relaxation = -1e-3
        limit = CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=[(g1, g2)],
            bound_relaxation=bound_relaxation,
        )

        # Configure model options for contact dimensionality and disable unnecessary constraints
        self.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT

        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)

        G, h = limit.compute_qp_inequalities(self.configuration, 1e-3)

        # Compute contact normal Jacobian using MuJoCo
        mujoco_contacts = data.contact

        if len(mujoco_contacts) == 0:
            # No contacts detected, set mujoco_G and mujoco_h to empty arrays with appropriate shapes
            mujoco_G = np.zeros((0, self.model.nv))
            mujoco_h = np.zeros(0)
        else:
            mujoco_G = np.zeros((len(mujoco_contacts), self.model.nv))
            mujoco_h = np.zeros(len(mujoco_contacts))

            for i, contact in enumerate(mujoco_contacts):
                mujoco.mju_contactJacobian(self.model, data, contact.geom1, contact.geom2, mujoco_G[i])
                mujoco_h[i] = contact.dist - bound_relaxation

        # Ensure both G and mujoco_G are empty if no contacts are detected
        if len(mujoco_contacts) == 0:
            G = np.zeros((0, self.model.nv))

        # Check that the computed G and h match MuJoCo's implementation
        np.testing.assert_allclose(G, mujoco_G, err_msg="G does not match MuJoCo's G")
        np.testing.assert_allclose(h, mujoco_h, err_msg="h does not match MuJoCo's h")


if __name__ == "__main__":
    absltest.main()