"""Tests for configuration_limit.py.

This module contains tests for the ConfigurationLimit class, which defines
inequality constraints on joint configurations in a robot model. The tests
verify the correct initialization, dimensionality, and behavior of the
ConfigurationLimit object under various conditions.
"""

import mujoco
import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import Configuration
from mink.limits import ConfigurationLimit, VelocityLimit
from mink.limits.exceptions import LimitDefinitionError
from mink.utils import get_freejoint_dims


class TestConfigurationLimit(absltest.TestCase):
    """Test suite for the ConfigurationLimit class.

    This class contains several test methods to verify the functionality of the
    ConfigurationLimit class, including initialization, dimensionality checks,
    and behavior with different model configurations.
    """

    @classmethod
    def setUpClass(cls):
        """Load a robot model for use in the tests."""
        cls.model = load_robot_description("g1_mj_description")

    def setUp(self):
        """Initialize a Configuration object and a VelocityLimit object for testing."""
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("stand")
        # NOTE: These velocities are arbitrary and do not match real hardware.
        self.velocities = {
            self.model.joint(i).name: 3.14 for i in range(1, self.model.njnt)
        }
        self.vel_limit = VelocityLimit(self.model, self.velocities)

    def test_throws_error_if_gain_invalid(self):
        """Test that an error is raised for invalid gain values."""
        with self.assertRaises(LimitDefinitionError):
            ConfigurationLimit(self.model, gain=-1)
        with self.assertRaises(LimitDefinitionError):
            ConfigurationLimit(self.model, gain=1.1)

    def test_dimensions(self):
        """Test that the dimensions of the indices and projection matrix are correct."""
        limit = ConfigurationLimit(self.model)
        nv = self.configuration.nv
        nb = nv - len(get_freejoint_dims(self.model)[1])
        self.assertEqual(len(limit.indices), nb)
        self.assertEqual(limit.projection_matrix.shape, (nb, nv))

    def test_indices(self):
        """Test that the indices of the velocity-limited joints are correct."""
        limit = ConfigurationLimit(self.model)
        expected_indices = np.arange(6, self.model.nv)
        self.assertTrue(np.allclose(limit.indices, expected_indices))

    def test_model_with_no_limit(self):
        """Test behavior with a model that has no velocity limits."""
        empty_model = mujoco.MjModel.from_xml_string("<mujoco></mujoco>")
        empty_bounded = ConfigurationLimit(empty_model)
        self.assertEqual(len(empty_bounded.indices), 0)
        self.assertIsNone(empty_bounded.projection_matrix)

    def test_model_with_subset_of_velocities_limited(self):
        """Test behavior with a model that has a subset of velocity-limited joints."""
        xml_str = """
        <mujoco>
          <worldbody>
            <body>
              <joint type="hinge" name="hinge_unlimited"/>
              <geom type="sphere" size=".1" mass=".1"/>
              <body>
                <joint type="hinge" name="hinge_limited" limited="true" range="0 1.57"/>
                <geom type="sphere" size=".1" mass=".1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        limit = ConfigurationLimit(model)
        nb = 1  # Only one limited joint.
        nv = model.nv
        self.assertEqual(limit.projection_matrix.shape, (nb, nv))
        self.assertEqual(len(limit.indices), nb)

    def test_freejoint_ignored(self):
        """Test that free joints are ignored in the velocity limits."""
        xml_str = """
        <mujoco>
          <worldbody>
            <body>
              <joint type="free" name="floating"/>
              <geom type="sphere" size=".1" mass=".1"/>
              <body>
                <joint type="hinge" name="hinge" range="0 1.57" limited="true"/>
                <geom type="sphere" size=".1" mass=".1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        limit = ConfigurationLimit(model)
        nb = 1  # Only one limited joint.
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