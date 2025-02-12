"""Tests for configuration_limit.py."""

import itertools

import mujoco
import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import Configuration
from mink.limits import CollisionAvoidanceLimit
from mink.limits.collision_avoidance_limit import (
    Contact,
    compute_contact_normal_jacobian,
)
from mink.utils import get_body_geom_ids


class TestCollisionAvoidanceLimit(absltest.TestCase):
    """Test collision avoidance limit functionality."""

    @classmethod
    def setUpClass(cls):
        """Load the robot model for all tests."""
        cls.model = load_robot_description("ur5e_mj_description")

    def setUp(self):
        """Initialize the configuration for each test."""
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

        # Filter out non-colliding geoms and calculate the expected maximum number of contacts.
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
        expected_max_num_contacts = len(list(itertools.product(g1_coll, g2_coll)))
        self.assertEqual(limit.max_num_contacts, expected_max_num_contacts)

        G, h = limit.compute_qp_inequalities(self.configuration, 1e-3)

        # Verify that the upper bound is always greater than or equal to the relaxation bound.
        self.assertTrue(np.all(h >= bound_relaxation))

        # Check that the inequality constraint dimensions are correct.
        self.assertEqual(G.shape, (expected_max_num_contacts, self.model.nv))
        self.assertEqual(h.shape, (expected_max_num_contacts,))

    def test_contact_normal_jac_matches_mujoco(self):
        """Test that the computed contact normal Jacobian matches MuJoCo's."""
        model = load_robot_description("ur5e_mj_description")
        nv = model.nv

        # Set options for obtaining separation normal and dense matrices.
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE

        # Disable unnecessary constraints.
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EQUALITY
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_LIMIT

        # Set contact dimensionality to 1 (normals only).
        model.geom_condim[:] = 1

        data = mujoco.MjData(model)

        # Use a handcrafted qpos that results in multiple contacts.
        qpos_coll = np.asarray([-1.5708, -1.5708, 3.01632, -1.5708, -1.5708, 0])
        data.qpos = qpos_coll
        mujoco.mj_forward(model, data)
        self.assertGreater(data.ncon, 1)

        for i in range(data.ncon):
            # Retrieve MuJoCo's contact normal Jacobian.\n            contact = data.contact[i]\n            start_idx = contact.efc_address * nv\n            end_idx = start_idx + nv\n            efc_J = data.efc_J[start_idx:end_idx]\n\n            # Manually compute the contact Jacobian.\n            normal = contact.frame[:3]\n            dist = contact.dist\n            fromto = np.empty((6,), dtype=np.float64)\n            fromto[3:] = contact.pos - 0.5 * dist * normal\n            fromto[:3] = contact.pos + 0.5 * dist * normal\n            contact = Contact(\n                dist=contact.dist,\n                fromto=fromto,\n                geom1=contact.geom1,\n                geom2=contact.geom2,\n                distmax=np.inf,\n            )\n            jac = compute_contact_normal_jacobian(model, data, contact)\n\n            # Assert that the manually computed Jacobian matches MuJoCo's.
            np.testing.assert_allclose(jac, efc_J, atol=1e-7)


if __name__ == "__main__":
    absltest.main()