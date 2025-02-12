"""Tests for velocity_limit.py, incorporating posture control and collision avoidance mechanisms."""

import mujoco
import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import Configuration, PostureController
from mink.limits import LimitDefinitionError, VelocityLimit, CollisionAvoidance
from mink.utils import get_freejoint_dims

class TestVelocityLimit(absltest.TestCase):
    """Test velocity limit with enhancements for posture control and collision avoidance."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("g1_mj_description")
        cls.collision_avoidance = CollisionAvoidance(cls.model)

    def setUp(self):
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("stand")
        # NOTE(kevin): These velocities are arbitrary and do not match real hardware.
        self.velocities = {
            self.model.joint(i).name: 3.14 for i in range(1, self.model.njnt)
        }
        self.posture_controller = PostureController(self.model, target_posture=self.configuration.qpos)

    def test_dimensions(self):
        limit = VelocityLimit(self.model, self.velocities)
        nv = self.configuration.nv
        nb = nv - len(get_freejoint_dims(self.model)[1])
        self.assertEqual(len(limit.indices), nb)
        self.assertEqual(limit.projection_matrix.shape, (nb, nv))

    def test_indices(self):
        limit = VelocityLimit(self.model, self.velocities)
        expected = np.arange(6, self.model.nv)  # Freejoint (0-5) is not limited.
        self.assertTrue(np.allclose(limit.indices, expected))

    def test_model_with_no_limit(self):
        empty_model = mujoco.MjModel.from_xml_string("<mujoco></mujoco>")
        empty_bounded = VelocityLimit(empty_model)
        self.assertEqual(len(empty_bounded.indices), 0)
        self.assertIsNone(empty_bounded.projection_matrix)
        G, h = empty_bounded.compute_qp_inequalities(self.configuration, 1e-3)
        self.assertIsNone(G)
        self.assertIsNone(h)

    def test_model_with_subset_of_velocities_limited(self):
        partial_velocities = {}
        for i, (key, value) in enumerate(self.velocities.items()):
            if i > 2:
                break
            partial_velocities[key] = value
        limit = VelocityLimit(self.model, partial_velocities)
        nb = 3
        nv = self.model.nv
        self.assertEqual(limit.projection_matrix.shape, (nb, nv))
        self.assertEqual(len(limit.indices), nb)
        expected_limit = np.asarray([3.14] * nb)
        np.testing.assert_allclose(limit.limit, expected_limit)

    def test_model_with_ball_joint(self):
        xml_str = """\n        <mujoco>\n          <worldbody>\n            <body>\n              <joint type="ball" name="ball"/>\n              <geom type="sphere" size=".1" mass=".1"/>\n              <body>\n                <joint type="hinge" name="hinge" range="0 1.57"/>\n                <geom type="sphere" size=".1" mass=".1"/>\n              </body>\n            </body>\n          </worldbody>\n        </mujoco>\n        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        velocities = {
            "ball": (np.pi, np.pi / 2, np.pi / 4),
            "hinge": (0.5,),
        }
        limit = VelocityLimit(model, velocities)
        nb = 3 + 1
        self.assertEqual(len(limit.indices), nb)
        self.assertEqual(limit.projection_matrix.shape, (nb, model.nv))

    def test_ball_joint_invalid_limit_shape(self):
        xml_str = """\n        <mujoco>\n          <worldbody>\n            <body>\n              <joint type="ball" name="ball"/>\n              <geom type="sphere" size=".1" mass=".1"/>\n              <body>\n                <joint type="hinge" name="hinge" range="0 1.57"/>\n                <geom type="sphere" size=".1" mass=".1"/>\n              </body>\n            </body>\n          </worldbody>\n        </mujoco>\n        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        velocities = {
            "ball": (np.pi, np.pi / 2),
        }
        with self.assertRaises(LimitDefinitionError) as cm:
            VelocityLimit(model, velocities)
        expected_error_message = "Joint ball must have a limit of shape (3,). Got: (2,)"
        self.assertEqual(str(cm.exception), expected_error_message)

    def test_that_freejoint_raises_error(self):
        xml_str = """\n        <mujoco>\n          <worldbody>\n            <body>\n              <joint type="free" name="floating"/>\n              <geom type="sphere" size=".1" mass=".1"/>\n              <body>\n                <joint type="hinge" name="hinge" range="0 1.57" limited="true"/>\n                <geom type="sphere" size=".1" mass=".1"/>\n              </body>\n            </body>\n          </worldbody>\n        </mujoco>\n        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        velocities = {
            "floating": np.pi,
            "hinge": np.pi,
        }
        with self.assertRaises(LimitDefinitionError) as cm:
            VelocityLimit(model, velocities)
        expected_error_message = "Free joint floating is not supported"
        self.assertEqual(str(cm.exception), expected_error_message)

    def test_posture_control_integration(self):
        """Test integration of posture control with velocity limits."""
        limit = VelocityLimit(self.model, self.velocities)
        posture_control_signal = self.posture_controller.compute_control_signal(self.configuration)
        self.assertTrue(np.allclose(posture_control_signal.shape, (self.configuration.nv,)))

    def test_collision_avoidance_integration(self):
        """Test integration of collision avoidance with velocity limits."""
        limit = VelocityLimit(self.model, self.velocities)
        collision_avoidance_signal = self.collision_avoidance.compute_avoidance_signal(self.configuration)
        self.assertTrue(np.allclose(collision_avoidance_signal.shape, (self.configuration.nv,)))

if __name__ == "__main__":
    absltest.main()