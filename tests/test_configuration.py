"""Tests for configuration.py."""

import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

import mink


class TestConfiguration(absltest.TestCase):
    """Test task various configuration methods work as intended."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_robot_description("ur5e_mj_description")

    def setUp(self) -> None:
        self.q_ref: np.ndarray = self.model.key("home").qpos

    def test_nq_nv(self) -> None:
        configuration: mink.Configuration = mink.Configuration(self.model)
        self.assertEqual(configuration.nq, self.model.nq)
        self.assertEqual(configuration.nv, self.model.nv)

    def test_initialize_from_keyframe(self) -> None:
        """Test that keyframe initialization correctly updates the configuration."""
        configuration: mink.Configuration = mink.Configuration(self.model)
        np.testing.assert_array_equal(configuration.q, np.zeros(self.model.nq))
        configuration.update_from_keyframe("home")
        np.testing.assert_array_equal(configuration.q, self.q_ref)

    def test_site_transform_world_frame(self) -> None:
        site_name: str = "attachment_site"
        configuration: mink.Configuration = mink.Configuration(self.model)

        # Randomly sample a joint configuration.
        np.random.seed(12345)
        configuration.data.qpos = np.random.uniform(*configuration.model.jnt_range.T)
        configuration.update()

        world_T_site = configuration.get_transform_frame_to_world(site_name, "site")

        expected_translation: np.ndarray = configuration.data.site(site_name).xpos
        np.testing.assert_array_equal(world_T_site.translation(), expected_translation)

        expected_xmat: np.ndarray = configuration.data.site(site_name).xmat.reshape(3, 3)
        np.testing.assert_almost_equal(
            world_T_site.rotation().as_matrix(), expected_xmat
        )

    def test_site_transform_raises_error_if_frame_name_is_invalid(self) -> None:
        """Raise an error when the requested frame does not exist."""
        configuration: mink.Configuration = mink.Configuration(self.model)
        with self.assertRaises(mink.InvalidFrame):
            configuration.get_transform_frame_to_world("invalid_name", "site")

    def test_site_transform_raises_error_if_frame_type_is_invalid(self) -> None:
        """Raise an error when the requested frame type is invalid."""
        configuration: mink.Configuration = mink.Configuration(self.model)
        with self.assertRaises(mink.UnsupportedFrame):
            configuration.get_transform_frame_to_world("name_does_not_matter", "joint")

    def test_update_raises_error_if_keyframe_is_invalid(self) -> None:
        """Raise an error when the request keyframe does not exist."""
        configuration: mink.Configuration = mink.Configuration(self.model)
        with self.assertRaises(mink.InvalidKeyframe):
            configuration.update_from_keyframe("invalid_keyframe")

    def test_inplace_integration(self) -> None:
        configuration: mink.Configuration = mink.Configuration(self.model, self.q_ref)

        dt: float = 1e-3
        qvel: np.ndarray = np.ones((self.model.nv,))
        # We can use this formula because the ur5e only has hinge joints.
        expected_qpos: np.ndarray = self.q_ref + dt * qvel

        # Regular integration should not modify the underlying q.
        qpos: np.ndarray = configuration.integrate(qvel, dt)
        np.testing.assert_almost_equal(qpos, expected_qpos)
        np.testing.assert_equal(configuration.q, self.q_ref)

        # Inplace integration should change qpos.
        configuration.integrate_inplace(qvel, dt)
        np.testing.assert_almost_equal(configuration.q, expected_qpos)

    def test_check_limits(self) -> None:
        """Check that an error is raised iff a joint limit is exceeded."""
        configuration: mink.Configuration = mink.Configuration(self.model, q=self.q_ref)
        configuration.check_limits()
        self.q_ref[0] += 1e4  # Move configuration out of bounds.
        configuration.update(q=self.q_ref)
        with self.assertRaises(mink.NotWithinConfigurationLimits):
            configuration.check_limits()