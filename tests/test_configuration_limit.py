"""Tests for configuration_limit.py.\n\nThis module contains tests for the ConfigurationLimit class, which manages\nconfiguration limits in a robot model. It ensures that the class correctly\nidentifies and applies limits to the robot's configuration space, ignoring\nfloating base joints and handling various edge cases such as invalid gains\nand models with no limits.\n"""

import mujoco
import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import Configuration
from mink.limits import ConfigurationLimit, VelocityLimit
from mink.limits.exceptions import LimitDefinitionError
from mink.utils import get_freejoint_dims


class TestConfigurationLimit(absltest.TestCase):
    """Test suite for the ConfigurationLimit class.\n\n    This test suite verifies the functionality of the ConfigurationLimit class,\n    ensuring it correctly applies configuration limits to a robot model, handles\n    various model configurations, and raises appropriate errors for invalid inputs.\n    """

    @classmethod
    def setUpClass(cls):
        """Load a robot model for testing."""
        cls.model = load_robot_description("g1_mj_description")

    def setUp(self):
        """Initialize common test objects and configurations."""
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("stand")
        # NOTE(kevin): These velocities are arbitrary and do not match real hardware.
        self.velocities = {
            self.model.joint(i).name: 3.14 for i in range(1, self.model.njnt)
        }
        self.vel_limit = VelocityLimit(self.model, self.velocities)

    def test_invalid_gain_raises_error(self):
        """Test that invalid gain values raise a LimitDefinitionError."""
        with self.assertRaises(LimitDefinitionError):
            ConfigurationLimit(self.model, gain=-1)
        with self.assertRaises(LimitDefinitionError):
            ConfigurationLimit(self.model, gain=1.1)

    def test_dimensionality(self):
        """Test that the dimensions of indices and projection matrix are correct."""
        limit = ConfigurationLimit(self.model)
        nv = self.configuration.nv
        nb = nv - len(get_freejoint_dims(self.model)[1])
        self.assertEqual(len(limit.indices), nb)
        self.assertEqual(limit.projection_matrix.shape, (nb, nv))

    def test_index_computation(self):
        """Test that the indices of velocity-limited joints are correctly computed."""
        limit = ConfigurationLimit(self.model)
        expected_indices = np.arange(6, self.model.nv)
        self.assertTrue(np.allclose(limit.indices, expected_indices))

    def test_no_limits_model(self):
        """Test behavior with a model that has no configuration limits."""
        empty_model = mujoco.MjModel.from_xml_string("<mujoco></mujoco>")
        empty_bounded = ConfigurationLimit(empty_model)
        self.assertEqual(len(empty_bounded.indices), 0)
        self.assertIsNone(empty_bounded.projection_matrix)

    def test_partial_limits_model(self):
        """Test behavior with a model that has a subset of joints with limits."""
        xml_str = """\n        <mujoco>\n          <worldbody>\n            <body>\n              <joint type="hinge" name="hinge_unlimited"/>\n              <geom type="sphere" size=".1" mass=".1"/>\n              <body>\n                <joint type="hinge" name="hinge_limited" limited="true" range="0 1.57"/>\n                <geom type="sphere" size=".1" mass=".1"/>\n              </body>\n            </body>\n          </worldbody>\n        </mujoco>\n        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        limit = ConfigurationLimit(model)
        nb = 1  # 1 limited joint.
        nv = model.nv
        self.assertEqual(limit.projection_matrix.shape, (nb, nv))
        self.assertEqual(len(limit.indices), nb)

    def test_free_joint_ignored(self):
        """Test that free joints are ignored in the computation of limits."""
        xml_str = """\n        <mujoco>\n          <worldbody>\n            <body>\n              <joint type="free" name="floating"/>\n              <geom type="sphere" size=".1" mass=".1"/>\n              <body>\n                <joint type="hinge" name="hinge" range="0 1.57" limited="true"/>\n                <geom type="sphere" size=".1" mass=".1"/>\n              </body>\n            </body>\n          </worldbody>\n        </mujoco>\n        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        limit = ConfigurationLimit(model)
        nb = 1  # 1 limited joint.
        nv = model.nv
        self.assertEqual(limit.projection_matrix.shape, (nb, nv))
        self.assertEqual(len(limit.indices), nb)

    def test_far_from_limit(self, tol=1e-10):
        """Test that the limit has no effect when the configuration is far away."""
        dt = 1e-3  # [s]
        model = load_robot_description("ur5e_mj_description")
        configuration = Configuration(model)
        limit = ConfigurationLimit(model)
        G, h = limit.compute_qp_inequalities(configuration, dt=dt)
        velocities = {
            "shoulder_pan_joint": np.pi,
            "shoulder_lift_joint": np.pi,
            "elbow_joint": np.pi,
            "wrist_1_joint": np.pi,
            "wrist_2_joint": np.pi,
            "wrist_3_joint": np.pi,
        }
        vel_limit = VelocityLimit(model, velocities)
        self.assertLess(np.max(+G @ vel_limit.limit * dt - h), -tol)
        self.assertLess(np.max(-G @ vel_limit.limit * dt - h), -tol)

    def test_configuration_limit_repulsion(self, tol=1e-10):
        """Test that velocities are scaled down when close to a configuration limit."""
        dt = 1e-3  # [s]
        slack_vel = 5e-4  # [rad] / [s]
        limit = ConfigurationLimit(self.model, gain=0.5)
        # Override configuration limits to `q +/- slack_vel * dt`.
        limit.lower = self.configuration.integrate(
            -slack_vel * np.ones((self.configuration.nv,)), dt
        )
        limit.upper = self.configuration.integrate(
            +slack_vel * np.ones((self.configuration.nv,)), dt
        )
        _, h = limit.compute_qp_inequalities(self.configuration, dt)
        self.assertLess(np.max(h), slack_vel * dt + tol)
        self.assertGreater(np.min(h), -slack_vel * dt - tol)


if __name__ == "__main__":
    absltest.main()