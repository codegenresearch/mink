"""Tests for the VelocityLimit class in velocity_limit.py."""

import mujoco
import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import Configuration
from mink.limits import LimitDefinitionError, VelocityLimit


class TestVelocityLimit(absltest.TestCase):
    """Tests the functionality of the VelocityLimit class."""

    @classmethod
    def setUpClass(cls):
        """Load a model for use in tests."""
        cls.model = load_robot_description("ur5e_mj_description")

    def setUp(self):
        """Initialize a configuration and define joint velocities."""
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("home")
        self.velocities = {
            "shoulder_pan_joint": np.pi,
            "shoulder_lift_joint": np.pi,
            "elbow_joint": np.pi,
            "wrist_1_joint": np.pi,
            "wrist_2_joint": np.pi,
            "wrist_3_joint": np.pi,
        }

    def test_projection_matrix_and_indices_dimensions(self):
        """Test the dimensions of the projection matrix and indices."""
        limit = VelocityLimit(self.model, self.velocities)
        nv = self.configuration.nv
        expected_projection_shape = (nv, nv)
        expected_indices_length = nv
        self.assertEqual(limit.projection_matrix.shape, expected_projection_shape)
        self.assertEqual(len(limit.indices), expected_indices_length)

    def test_no_velocity_limits(self):
        """Test the case where no velocity limits are defined."""
        empty_model = mujoco.MjModel.from_xml_string("<mujoco></mujoco>")
        empty_bounded = VelocityLimit(empty_model)
        self.assertEqual(len(empty_bounded.indices), 0)
        self.assertIsNone(empty_bounded.projection_matrix)
        G, h = empty_bounded.compute_qp_inequalities(self.configuration, 1e-3)
        self.assertIsNone(G)
        self.assertIsNone(h)

    def test_partial_velocity_limits(self):
        """Test the case where only a subset of joints have velocity limits."""
        velocities = {
            "wrist_1_joint": np.pi,
            "wrist_2_joint": np.pi,
            "wrist_3_joint": np.pi,
        }
        limit = VelocityLimit(self.model, velocities)
        nb = 3
        nv = self.model.nv
        expected_projection_shape = (nb, nv)
        expected_indices_length = nb
        self.assertEqual(limit.projection_matrix.shape, expected_projection_shape)
        self.assertEqual(len(limit.indices), expected_indices_length)

    def test_ball_joint_velocity_limits(self):
        """Test velocity limits for a ball joint."""
        xml_str = """\n        <mujoco>\n          <worldbody>\n            <body>\n              <joint type="ball" name="ball"/>\n              <geom type="sphere" size=".1" mass=".1"/>\n              <body>\n                <joint type="hinge" name="hinge" range="0 1.57"/>\n                <geom type="sphere" size=".1" mass=".1"/>\n              </body>\n            </body>\n          </worldbody>\n        </mujoco>\n        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        velocities = {
            "ball": (np.pi, np.pi / 2, np.pi / 4),
            "hinge": (0.5,),
        }
        limit = VelocityLimit(model, velocities)
        nb = 3 + 1
        expected_indices_length = nb
        expected_projection_shape = (nb, model.nv)
        self.assertEqual(len(limit.indices), expected_indices_length)
        self.assertEqual(limit.projection_matrix.shape, expected_projection_shape)

    def test_ball_joint_invalid_velocity_limit_shape(self):
        """Test that an error is raised when a ball joint has an invalid velocity limit shape."""
        xml_str = """\n        <mujoco>\n          <worldbody>\n            <body>\n              <joint type="ball" name="ball"/>\n              <geom type="sphere" size=".1" mass=".1"/>\n              <body>\n                <joint type="hinge" name="hinge" range="0 1.57"/>\n                <geom type="sphere" size=".1" mass=".1"/>\n              </body>\n            </body>\n          </worldbody>\n        </mujoco>\n        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        velocities = {
            "ball": (np.pi, np.pi / 2),
        }
        with self.assertRaises(LimitDefinitionError) as context_manager:
            VelocityLimit(model, velocities)
        expected_error_message = "Joint ball must have a limit of shape (3,). Got: (2,)"
        self.assertEqual(str(context_manager.exception), expected_error_message)

    def test_free_joint_raises_error(self):
        """Test that an error is raised when a free joint is included in velocity limits."""
        xml_str = """\n        <mujoco>\n          <worldbody>\n            <body>\n              <joint type="free" name="floating"/>\n              <geom type="sphere" size=".1" mass=".1"/>\n              <body>\n                <joint type="hinge" name="hinge" range="0 1.57" limited="true"/>\n                <geom type="sphere" size=".1" mass=".1"/>\n              </body>\n            </body>\n          </worldbody>\n        </mujoco>\n        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        velocities = {
            "floating": np.pi,
            "hinge": np.pi,
        }
        with self.assertRaises(LimitDefinitionError) as context_manager:
            VelocityLimit(model, velocities)
        expected_error_message = "Free joint floating is not supported"
        self.assertEqual(str(context_manager.exception), expected_error_message)


if __name__ == "__main__":
    absltest.main()