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
        geom_ids_wrist_2 = get_body_geom_ids(self.model, self.model.body("wrist_2_link").id)
        geom_ids_upper_arm = get_body_geom_ids(self.model, self.model.body("upper_arm_link").id)

        bound_relaxation = -1e-3
        collision_limit = CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=[(geom_ids_wrist_2, geom_ids_upper_arm)],
            bound_relaxation=bound_relaxation,
        )

        # Filter out non-colliding geoms and calculate expected number of contacts.
        colliding_geoms_wrist_2 = [
            geom_id
            for geom_id in geom_ids_wrist_2
            if self.model.geom_conaffinity[geom_id] != 0 and self.model.geom_contype[geom_id] != 0
        ]
        colliding_geoms_upper_arm = [
            geom_id
            for geom_id in geom_ids_upper_arm
            if self.model.geom_conaffinity[geom_id] != 0 and self.model.geom_contype[geom_id] != 0
        ]
        expected_max_contacts = len(list(itertools.product(colliding_geoms_wrist_2, colliding_geoms_upper_arm)))
        self.assertEqual(collision_limit.max_num_contacts, expected_max_contacts)

        G_matrix, h_vector = collision_limit.compute_qp_inequalities(self.configuration, 1e-3)

        # Validate the upper bound and constraint dimensions.
        self.assertTrue(np.all(h_vector >= bound_relaxation))
        self.assertEqual(G_matrix.shape, (expected_max_contacts, self.model.nv))
        self.assertEqual(h_vector.shape, (expected_max_contacts,))

    def test_contact_normal_jacobian_matches_mujoco(self):
        """Ensure the computed contact normal Jacobian matches MuJoCo's output."""
        model = load_robot_description("ur5e_mj_description")
        num_dof = model.nv

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
        self.assertGreater(data.ncon, 1)

        for contact_index in range(data.ncon):
            # Retrieve MuJoCo's contact normal Jacobian.
            contact = data.contact[contact_index]
            start_idx = contact.efc_address * num_dof
            end_idx = start_idx + num_dof
            mujoco_jacobian = data.efc_J[start_idx:end_idx]

            # Manually compute the contact Jacobian.
            normal_vector = contact.frame[:3]
            distance = contact.dist
            from_to = np.empty((6,), dtype=np.float64)
            from_to[3:] = contact.pos - 0.5 * distance * normal_vector
            from_to[:3] = contact.pos + 0.5 * distance * normal_vector
            contact_info = Contact(
                dist=distance,
                fromto=from_to,
                geom1=contact.geom1,
                geom2=contact.geom2,
                distmax=np.inf,
            )
            computed_jacobian = compute_contact_normal_jacobian(model, data, contact_info)

            # Compare the computed Jacobian with MuJoCo's.
            np.testing.assert_allclose(computed_jacobian, mujoco_jacobian, atol=1e-7)


if __name__ == "__main__":
    absltest.main()