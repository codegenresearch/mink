"""Tests for collision_avoidance_limit.py."""

import itertools

import numpy as np
import numpy.testing as npt
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import Configuration
from mink.limits import CollisionAvoidanceLimit
from mink.limits.collision_avoidance_limit import Contact, compute_contact_normal_jacobian
from mink.utils import get_body_geom_ids
import mujoco


class TestCollisionAvoidanceLimit(absltest.TestCase):
    """Test collision avoidance limit."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("ur5e_mj_description")
        # Configure model options for consistent testing
        cls.model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        cls.model.opt.jacobian = mujoco.mjtJac.mjJAC_DENSE
        cls.model.opt.disableflags = (
            mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL
            | mujoco.mjtDisableBit.mjDSBL_PASSIVE
            | mujoco.mjtDisableBit.mjDSBL_GRAVITY
            | mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
            | mujoco.mjtDisableBit.mjDSBL_LIMIT
        )

    def setUp(self):
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("home")
        self.data = mujoco.MjData(self.model)

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
        expected_max_num_contacts = len(list(itertools.product(g1_coll, g2_coll)))

        # Validate the number of max expected contacts.
        self.assertEqual(limit.max_num_contacts, expected_max_num_contacts)

        # Compute the quadratic programming inequalities.
        G, h = limit.compute_qp_inequalities(self.configuration, 1e-3)

        # Ensure the upper bound is greater than or equal to the relaxation bound.
        self.assertTrue(np.all(h >= bound_relaxation))

        # Validate the dimensions of the inequality constraints.
        self.assertEqual(G.shape, (expected_max_num_contacts, self.model.nv))
        self.assertEqual(h.shape, (expected_max_num_contacts,))

    def test_contact_normal_jac_matches_mujoco(self):
        g1 = get_body_geom_ids(self.model, self.model.body("wrist_2_link").id)
        g2 = get_body_geom_ids(self.model, self.model.body("upper_arm_link").id)

        bound_relaxation = -1e-3
        limit = CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=[(g1, g2)],
            bound_relaxation=bound_relaxation,
        )

        # Step the simulation to ensure contacts are updated
        mujoco.mj_step(self.model, self.data)

        # Extract contact information
        mujoco_contacts = [Contact(self.model, i) for i in range(self.data.ncon)]

        # Compute the contact normal Jacobian using the limit method.
        jac_limit = limit.compute_contact_normal_jacobian(self.configuration)

        # Compute the contact normal Jacobian using Mujoco's method.
        jac_mujoco = compute_contact_normal_jacobian(self.model, mujoco_contacts)

        # Validate that the Jacobians match.
        npt.assert_allclose(jac_limit, jac_mujoco, atol=1e-6)


if __name__ == "__main__":
    absltest.main()