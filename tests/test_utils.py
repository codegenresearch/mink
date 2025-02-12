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
        self.q0 = self.data.qpos.copy()

    def test_custom_configuration_vector_throws_error_if_keyframe_invalid(self):
        with self.assertRaises(InvalidKeyframe):
            utils.custom_configuration_vector(self.model, "stand123")

    def test_custom_configuration_vector_from_keyframe(self):
        q = utils.custom_configuration_vector(self.model, "stand")
        np.testing.assert_allclose(q, self.model.key("stand").qpos)

    def test_custom_configuration_vector_raises_error_if_jnt_shape_invalid(self):
        with self.assertRaises(ValueError):
            utils.custom_configuration_vector(
                self.model,
                "stand",
                left_ankle_pitch_joint=(0.1, 0.1),
            )

    def test_custom_configuration_vector(self):
        custom_joints = dict(
            left_ankle_pitch_joint=0.2,  # Hinge.
            right_ankle_roll_joint=0.1,  # Slide.
        )
        q = utils.custom_configuration_vector(self.model, **custom_joints)
        q_expected = self.q0.copy()
        for name, value in custom_joints.items():
            qid = self.model.jnt_qposadr[self.model.joint(name).id]
            q_expected[qid] = value
        np.testing.assert_array_almost_equal(q, q_expected)

    def test_move_mocap_to_frame_throws_error_if_body_not_mocap(self):
        with self.assertRaises(InvalidMocapBody):
            utils.move_mocap_to_frame(
                self.model,
                self.data,
                "left_ankle_roll_link",
                "unused_frame_name",
                "unused_frame_type",
            )

    def test_move_mocap_to_frame(self):
        xml_str = """\n        <mujoco>\n          <worldbody>\n            <body pos=".1 -.1 0">\n              <joint type="free" name="floating"/>\n              <geom type="sphere" size=".1" mass=".1"/>\n              <body name="test">\n                <joint type="hinge" name="hinge" range="0 1.57" limited="true"/>\n                <geom type="sphere" size=".1" mass=".1"/>\n              </body>\n            </body>\n            <body name="mocap" mocap="true" pos=".5 1 5" quat="1 1 0 0">\n              <geom type="sphere" size=".1" mass=".1"/>\n            </body>\n          </worldbody>\n        </mujoco>\n        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        body_pos = data.body("test").xpos
        body_quat = np.empty(4)
        mujoco.mju_mat2Quat(body_quat, data.body("test").xmat)

        # Initially not the same.
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_allclose(data.body("mocap").xpos, body_pos)
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_allclose(data.body("mocap").xquat, body_quat)

        utils.move_mocap_to_frame(model, data, "mocap", "test", "body")
        mujoco.mj_forward(model, data)

        # Should now be the same.
        np.testing.assert_allclose(data.body("mocap").xpos, body_pos)
        np.testing.assert_allclose(data.body("mocap").xquat, body_quat)

    def test_get_freejoint_dims(self):
        q_ids, v_ids = utils.get_freejoint_dims(self.model)
        np.testing.assert_allclose(
            np.asarray(q_ids),
            np.asarray(list(range(0, 7))),
        )
        np.testing.assert_allclose(
            np.asarray(v_ids),
            np.asarray(list(range(0, 6))),
        )

    def test_get_subtree_geom_ids(self):
        xml_str = """\n        <mujoco>\n          <worldbody>\n            <body name="b1" pos=".1 -.1 0">\n              <joint type="free"/>\n              <geom name="b1/g1" type="sphere" size=".1" mass=".1"/>\n              <geom name="b1/g2" type="sphere" size=".1" mass=".1" pos="0 0 .5"/>\n              <body name="b2">\n                <joint type="hinge" range="0 1.57" limited="true"/>\n                <geom name="b2/g1" type="sphere" size=".1" mass=".1"/>\n              </body>\n            </body>\n            <body name="b3" pos="1 1 1">\n              <joint type="free"/>\n              <geom name="b3/g1" type="sphere" size=".1" mass=".1"/>\n              <body name="b4">\n                <joint type="hinge" range="0 1.57" limited="true"/>\n                <geom name="b4/g1" type="sphere" size=".1" mass=".1"/>\n              </body>\n            </body>\n          </worldbody>\n        </mujoco>\n        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        b1_id = model.body("b1").id
        actual_geom_ids = utils.get_subtree_geom_ids(model, b1_id)
        geom_names = ["b1/g1", "b1/g2", "b2/g1"]
        expected_geom_ids = [model.geom(g).id for g in geom_names]
        self.assertListEqual(actual_geom_ids, expected_geom_ids)


if __name__ == "__main__":
    absltest.main()