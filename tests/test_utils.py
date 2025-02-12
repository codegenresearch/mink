"""Tests for utils.py."""

import mujoco
import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import utils
from mink.exceptions import InvalidKeyframe, InvalidMocapBody


class TestUtils(absltest.TestCase):
    """Test utility functions."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("g1_mj_description")

    def setUp(self):
        self.data = mujoco.MjData(self.model)
        self.initial_configuration = self.data.qpos.copy()

    def test_custom_configuration_vector_raises_error_for_invalid_keyframe(self):
        with self.assertRaises(InvalidKeyframe):
            utils.custom_configuration_vector(self.model, "invalid_keyframe")

    def test_custom_configuration_vector_from_valid_keyframe(self):
        configuration_vector = utils.custom_configuration_vector(self.model, "stand")
        np.testing.assert_allclose(configuration_vector, self.model.key("stand").qpos)

    def test_custom_configuration_vector_raises_error_for_invalid_joint_shape(self):
        with self.assertRaises(ValueError):
            utils.custom_configuration_vector(
                self.model,
                "stand",
                left_ankle_pitch_joint=(0.1, 0.1),
            )

    def test_custom_configuration_vector_with_custom_joints(self):
        custom_joints = {
            "left_ankle_pitch_joint": 0.2,  # Hinge.
            "right_ankle_roll_joint": 0.1,  # Slide.
        }
        configuration_vector = utils.custom_configuration_vector(self.model, **custom_joints)
        expected_configuration_vector = self.initial_configuration.copy()
        for joint_name, value in custom_joints.items():
            joint_id = self.model.joint(joint_name).id
            qpos_index = self.model.jnt_qposadr[joint_id]
            expected_configuration_vector[qpos_index] = value
        np.testing.assert_array_almost_equal(configuration_vector, expected_configuration_vector)

    def test_move_mocap_to_frame_raises_error_for_non_mocap_body(self):
        with self.assertRaises(InvalidMocapBody):
            utils.move_mocap_to_frame(
                self.model,
                self.data,
                "non_mocap_body",
                "unused_frame_name",
                "unused_frame_type",
            )

    def test_move_mocap_to_frame_with_valid_arguments(self):
        xml_str = """\n        <mujoco>\n          <worldbody>\n            <body pos=".1 -.1 0">\n              <joint type="free" name="floating"/>\n              <geom type="sphere" size=".1" mass=".1"/>\n              <body name="test">\n                <joint type="hinge" name="hinge" range="0 1.57" limited="true"/>\n                <geom type="sphere" size=".1" mass=".1"/>\n              </body>\n            </body>\n            <body name="mocap" mocap="true" pos=".5 1 5" quat="1 1 0 0">\n              <geom type="sphere" size=".1" mass=".1"/>\n            </body>\n          </worldbody>\n        </mujoco>\n        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        target_body_position = data.body("test").xpos
        target_body_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(target_body_quaternion, data.body("test").xmat)

        # Initially, the mocap body position and quaternion should not match the target body's.\n        with np.testing.assert_raises(AssertionError):\n            np.testing.assert_allclose(data.body("mocap").xpos, target_body_position)\n        with np.testing.assert_raises(AssertionError):\n            np.testing.assert_allclose(data.body("mocap").xquat, target_body_quaternion)\n\n        utils.move_mocap_to_frame(model, data, "mocap", "test", "body")\n        mujoco.mj_forward(model, data)\n\n        # After moving the mocap body, its position and quaternion should match the target body's.
        np.testing.assert_allclose(data.body("mocap").xpos, target_body_position)
        np.testing.assert_allclose(data.body("mocap").xquat, target_body_quaternion)

    def test_get_freejoint_dimensions(self):
        q_indices, v_indices = utils.get_freejoint_dims(self.model)
        expected_q_indices = list(range(7))
        expected_v_indices = list(range(6))
        np.testing.assert_allclose(q_indices, expected_q_indices)
        np.testing.assert_allclose(v_indices, expected_v_indices)

    def test_get_subtree_geom_ids(self):
        xml_str = """\n        <mujoco>\n          <worldbody>\n            <body name="b1" pos=".1 -.1 0">\n              <joint type="free"/>\n              <geom name="b1/g1" type="sphere" size=".1" mass=".1"/>\n              <geom name="b1/g2" type="sphere" size=".1" mass=".1" pos="0 0 .5"/>\n              <body name="b2">\n                <joint type="hinge" range="0 1.57" limited="true"/>\n                <geom name="b2/g1" type="sphere" size=".1" mass=".1"/>\n              </body>\n            </body>\n            <body name="b3" pos="1 1 1">\n              <joint type="free"/>\n              <geom name="b3/g1" type="sphere" size=".1" mass=".1"/>\n              <body name="b4">\n                <joint type="hinge" range="0 1.57" limited="true"/>\n                <geom name="b4/g1" type="sphere" size=".1" mass=".1"/>\n              </body>\n            </body>\n          </worldbody>\n        </mujoco>\n        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        body_id = model.body("b1").id
        actual_geom_ids = utils.get_subtree_geom_ids(model, body_id)
        expected_geom_names = ["b1/g1", "b1/g2", "b2/g1"]
        expected_geom_ids = [model.geom(name).id for name in expected_geom_names]
        self.assertListEqual(actual_geom_ids, expected_geom_ids)


if __name__ == "__main__":
    absltest.main()