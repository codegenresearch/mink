"""Tests for utils.py."""

import mujoco
import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import utils
from mink.exceptions import InvalidKeyframe, InvalidMocapBody


def get_direct_and_descendant_geoms(model, body_id):
    """Retrieve direct and descendant geoms for a given body.

    Args:
        model: Mujoco model.
        body_id: ID of the body.

    Returns:
        A tuple of lists: (direct_geoms, descendant_geoms).
    """
    direct_geoms = []
    descendant_geoms = []

    # Collect direct geoms
    for geom_id in range(model.body_geomadr[body_id], model.body_geomadr[body_id + 1]):
        direct_geoms.append(geom_id)

    # Collect descendant geoms using depth-first traversal
    stack = [body_id]
    while stack:
        current_body_id = stack.pop()
        for child_id in range(model.body_childadr[current_body_id], model.body_childadr[current_body_id + 1]):
            stack.append(child_id)
            for geom_id in range(model.body_geomadr[child_id], model.body_geomadr[child_id + 1]):
                descendant_geoms.append(geom_id)

    return direct_geoms, descendant_geoms


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
        np.testing.assert_allclose(q, self.model.key("stand").qpos, atol=1e-7)

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
        np.testing.assert_array_almost_equal(q, q_expected, decimal=7)

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
        xml_str = """
        <mujoco>
          <worldbody>
            <body pos=".1 -.1 0">
              <joint type="free" name="floating"/>
              <geom type="sphere" size=".1" mass=".1"/>
              <body name="test">
                <joint type="hinge" name="hinge" range="0 1.57" limited="true"/>
                <geom type="sphere" size=".1" mass=".1"/>
              </body>
            </body>
            <body name="mocap" mocap="true" pos=".5 1 5" quat="1 1 0 0">
              <geom type="sphere" size=".1" mass=".1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        body_pos = data.body("test").xpos
        body_quat = np.empty(4)
        mujoco.mju_mat2Quat(body_quat, data.body("test").xmat)

        # Initially not the same.
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_allclose(data.body("mocap").xpos, body_pos, atol=1e-7)
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_allclose(data.body("mocap").xquat, body_quat, atol=1e-7)

        utils.move_mocap_to_frame(model, data, "mocap", "test", "body")
        mujoco.mj_forward(model, data)

        # Should now be the same.
        np.testing.assert_allclose(data.body("mocap").xpos, body_pos, atol=1e-7)
        np.testing.assert_allclose(data.body("mocap").xquat, body_quat, atol=1e-7)

    def test_get_freejoint_dims(self):
        q_ids, v_ids = utils.get_freejoint_dims(self.model)
        np.testing.assert_allclose(
            np.asarray(q_ids),
            np.asarray(list(range(0, 7))),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            np.asarray(v_ids),
            np.asarray(list(range(0, 6))),
            atol=1e-7,
        )

    def test_get_subtree_geom_ids(self):
        xml_str = """
        <mujoco>
          <worldbody>
            <body name="b1" pos=".1 -.1 0">
              <joint type="free"/>
              <geom name="b1/g1" type="sphere" size=".1" mass=".1"/>
              <geom name="b1/g2" type="sphere" size=".1" mass=".1" pos="0 0 .5"/>
              <body name="b2">
                <joint type="hinge" range="0 1.57" limited="true"/>
                <geom name="b2/g1" type="sphere" size=".1" mass=".1"/>
              </body>
            </body>
            <body name="b3" pos="1 1 1">
              <joint type="free"/>
              <geom name="b3/g1" type="sphere" size=".1" mass=".1"/>
              <body name="b4">
                <joint type="hinge" range="0 1.57" limited="true"/>
                <geom name="b4/g1" type="sphere" size=".1" mass=".1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        b1_id = model.body("b1").id
        actual_geom_ids = utils.get_subtree_geom_ids(model, b1_id)
        geom_names = ["b1/g1", "b1/g2", "b2/g1"]
        expected_geom_ids = [model.geom(g).id for g in geom_names]
        self.assertListEqual(actual_geom_ids, expected_geom_ids)

    def test_get_direct_and_descendant_geoms(self):
        xml_str = """
        <mujoco>
          <worldbody>
            <body name="b1" pos=".1 -.1 0">
              <joint type="free"/>
              <geom name="b1/g1" type="sphere" size=".1" mass=".1"/>
              <geom name="b1/g2" type="sphere" size=".1" mass=".1" pos="0 0 .5"/>
              <body name="b2">
                <joint type="hinge" range="0 1.57" limited="true"/>
                <geom name="b2/g1" type="sphere" size=".1" mass=".1"/>
              </body>
            </body>
            <body name="b3" pos="1 1 1">
              <joint type="free"/>
              <geom name="b3/g1" type="sphere" size=".1" mass=".1"/>
              <body name="b4">
                <joint type="hinge" range="0 1.57" limited="true"/>
                <geom name="b4/g1" type="sphere" size=".1" mass=".1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        b1_id = model.body("b1").id
        direct_geoms, descendant_geoms = get_direct_and_descendant_geoms(model, b1_id)
        direct_geom_names = ["b1/g1", "b1/g2"]
        descendant_geom_names = ["b1/g1", "b1/g2", "b2/g1"]
        expected_direct_geom_ids = [model.geom(g).id for g in direct_geom_names]
        expected_descendant_geom_ids = [model.geom(g).id for g in descendant_geom_names]
        self.assertListEqual(direct_geoms, expected_direct_geom_ids)
        self.assertListEqual(descendant_geoms, expected_descendant_geom_ids)


if __name__ == "__main__":
    absltest.main()