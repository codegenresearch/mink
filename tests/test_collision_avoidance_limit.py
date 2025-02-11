"""Tests for collision avoidance limit functionality in configuration_limit.py."""

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
    """Test suite for the CollisionAvoidanceLimit class."""

    @classmethod
    def setUpClass(cls):
        """Load the UR5E robot model for testing."""
        cls.model = load_robot_description("ur5e_mj_description")

    def setUp(self):
        """Initialize the configuration to a known keyframe."""
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("home")

    def test_dimensions(self):
        """Verify the dimensions of the collision avoidance limit constraints."""
        g1 = get_body_geom_ids(self.model, self.model.body("wrist_2_link").id)
        g2 = get_body_geom_ids(self.model, self.model.body("upper_arm_link").id)

        bound_relaxation = -1e-3
        limit = CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=[(g1, g2)],
            bound_relaxation=bound_relaxation,
        )

        # Filter out non-colliding geoms and calculate expected number of contacts.
        g1_coll = [g for g in g1 if self.model.geom_conaffinity[g] != 0 and self.model.geom_contype[g] != 0]
        g2_coll = [g for g in g2 if self.model.geom_conaffinity[g] != 0 and self.model.geom_contype[g] != 0]
        expected_max_contacts = len(list(itertools.product(g1_coll, g2_coll)))
        self.assertEqual(limit.max_num_contacts, expected_max_contacts)

        G, h = limit.compute_qp_inequalities(self.configuration, 1e-3)

        # Validate the upper bound and constraint dimensions.
        self.assertTrue(np.all(h >= bound_relaxation), "All elements of h should be greater than or equal to the bound relaxation.")
        self.assertEqual(G.shape, (expected_max_contacts, self.model.nv), "G matrix shape should match the expected number of contacts and degrees of freedom.")
        self.assertEqual(h.shape, (expected_max_contacts,), "h vector shape should match the expected number of contacts.")

    def test_contact_normal_jac_matches_mujoco(self):
        """Ensure the computed contact normal Jacobian matches MuJoCo's output."""
        model = load_robot_description("ur5e_mj_description")
        nv = model.nv

        # Configure model options for contact normal computation.
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EQUALITY | mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS | mujoco.mjtDisableBit.mjDSBL_LIMIT
        model.geom_condim[:] = 1

        data = mujoco.MjData(model)

        # Set a configuration with multiple contacts.
        qpos_with_contacts = np.array([-1.5708, -1.5708, 3.01632, -1.5708, -1.5708, 0])
        data.qpos = qpos_with_contacts
        mujoco.mj_forward(model, data)
        self.assertGreater(data.ncon, 1, "There should be multiple contacts.")

        for i in range(data.ncon):
            # Retrieve MuJoCo's contact normal Jacobian.
            contact = data.contact[i]
            start_idx = contact.efc_address * nv
            end_idx = start_idx + nv
            mujoco_jacobian = data.efc_J[start_idx:end_idx]

            # Manually compute the contact Jacobian.
            normal = contact.frame[:3]
            dist = contact.dist
            fromto = np.empty((6,), dtype=np.float64)
            fromto[3:] = contact.pos - 0.5 * dist * normal
            fromto[:3] = contact.pos + 0.5 * dist * normal
            contact_info = Contact(
                dist=dist,
                fromto=fromto,
                geom1=contact.geom1,
                geom2=contact.geom2,
                distmax=np.inf,
            )
            computed_jacobian = compute_contact_normal_jacobian(model, data, contact_info)

            # Compare the computed Jacobian with MuJoCo's.
            np.testing.assert_allclose(computed_jacobian, mujoco_jacobian, atol=1e-7, err_msg="Computed Jacobian does not match MuJoCo's Jacobian.")


if __name__ == "__main__":
    absltest.main()