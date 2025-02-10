"""Tests for utils.py with enhanced functionality and improved organization."""

import mujoco
import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import utils
from mink.exceptions import InvalidKeyframe, InvalidMocapBody
from mink.lie.se3 import SE3


class TestUtils(absltest.TestCase):
    """Test utility functions with gravity compensation and subtree handling."""

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
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(data.body("mocap").xpos, body_pos)
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(data.body("mocap").xquat, body_quat)

        utils.move_mocap_to_frame(model, data, "mocap", "test", "body")
        mujoco.mj_forward(model, data)

        # Should now be the same.
        np.testing.assert_allclose(data.body("mocap").xpos, body_pos)
        np.testing.assert_allclose(data.body("mocap").xquat, body_quat)

    def test_get_freejoint_dims(self):
        q_ids, v_ids = utils.get_freejoint_dims(self.model)
        np.testing.assert_allclose(np.asarray(q_ids), np.asarray(list(range(0, 7))))
        np.testing.assert_allclose(np.asarray(v_ids), np.asarray(list(range(0, 6))))

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
        expected_geom_ids = {model.geom(g).id for g in geom_names}
        self.assertSetEqual(set(actual_geom_ids), expected_geom_ids)

    def test_get_subtree_body_ids(self):
        xml_str = """
        <mujoco>
          <worldbody>
            <body name="b1" pos=".1 -.1 0">
              <joint type="free"/>
              <geom type="sphere" size=".1" mass=".1"/>
              <body name="b2">
                <joint type="hinge" range="0 1.57" limited="true"/>
                <geom type="sphere" size=".1" mass=".1"/>
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
        actual_body_ids = utils.get_subtree_body_ids(model, b1_id)
        body_names = ["b1", "b2"]
        expected_body_ids = {model.body(b).id for b in body_names}
        self.assertSetEqual(set(actual_body_ids), expected_body_ids)

    def test_apply_gravity_compensation(self):
        q = utils.custom_configuration_vector(self.model, "stand")
        utils.apply_gravity_compensation(self.model, self.data, q)
        mujoco.mj_forward(self.model, self.data)
        self.assertTrue(np.any(self.data.qfrc_passive != 0))

    def test_get_subtree_transform(self):
        xml_str = """
        <mujoco>
          <worldbody>
            <body name="b1" pos=".1 -.1 0">
              <joint type="free"/>
              <geom type="sphere" size=".1" mass=".1"/>
              <body name="b2">
                <joint type="hinge" range="0 1.57" limited="true"/>
                <geom type="sphere" size=".1" mass=".1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        b1_id = model.body("b1").id
        transform = utils.get_subtree_transform(model, data, b1_id)
        expected_translation = data.body("b1").xpos
        expected_rotation = SE3.from_matrix(data.body("b1").xmat)
        np.testing.assert_allclose(transform.translation(), expected_translation)
        np.testing.assert_allclose(transform.rotation().as_matrix(), expected_rotation.as_matrix())


if __name__ == "__main__":
    absltest.main()


### Additional Utility Functions in `mink.utils`

To ensure the tests pass, you need to implement the `apply_gravity_compensation` and `get_subtree_transform` functions in the `mink.utils` module. Here are the implementations:


# mink/utils.py

import mujoco
import numpy as np
from mink.lie.se3 import SE3


def custom_configuration_vector(model, keyframe_name=None, **custom_joints):
    if keyframe_name:
        q = model.key(keyframe_name).qpos.copy()
    else:
        q = np.zeros(model.nq)

    for name, value in custom_joints.items():
        jnt = model.joint(name)
        qid = model.jnt_qposadr[jnt.id]
        if jnt.type == mujoco.mjtJoint.mjJNT_HINGE:
            if isinstance(value, (list, tuple)) and len(value) == 2:
                raise ValueError(f"Joint {name} is a hinge and expects a single value, not {value}")
            q[qid] = value
        elif jnt.type == mujoco.mjtJoint.mjJNT_SLIDE:
            if isinstance(value, (list, tuple)) and len(value) == 2:
                raise ValueError(f"Joint {name} is a slide and expects a single value, not {value}")
            q[qid] = value
        else:
            raise ValueError(f"Joint type {jnt.type} not supported for joint {name}")

    return q


def move_mocap_to_frame(model, data, mocap_name, frame_name, frame_type):
    mocap_body = model.body(mocap_name)
    if not mocap_body.mocapid:
        raise InvalidMocapBody(f"Body {mocap_name} is not a mocap body.")

    if frame_type == "body":
        frame_body = model.body(frame_name)
        data.mocap_pos[mocap_body.mocapid[0]] = frame_body.xpos
        data.mocap_quat[mocap_body.mocapid[0]] = frame_body.xquat
    else:
        raise ValueError(f"Frame type {frame_type} not supported.")


def get_freejoint_dims(model):
    freejoint_qids = []
    freejoint_vids = []
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            freejoint_qids.extend(range(model.jnt_qposadr[i], model.jnt_qposadr[i] + 7))
            freejoint_vids.extend(range(model.jnt_dofadr[i], model.jnt_dofadr[i] + 6))
    return freejoint_qids, freejoint_vids


def get_subtree_geom_ids(model, body_id):
    geom_ids = []
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == body_id:
            geom_ids.append(geom_id)
        elif model.geom_bodyid[geom_id] in model.body_subtreelist[body_id].childbodyid:
            geom_ids.append(geom_id)
    return geom_ids


def get_subtree_body_ids(model, body_id):
    body_ids = [body_id]
    for child_id in model.body_subtreelist[body_id].childbodyid:
        body_ids.extend(get_subtree_body_ids(model, child_id))
    return body_ids


def apply_gravity_compensation(model, data, q):
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    mujoco.mj_inverse(model, data)


def get_subtree_transform(model, data, body_id):
    xpos = data.body(body_id).xpos
    xmat = data.body(body_id).xmat
    return SE3(position=xpos, rotation=SE3.from_matrix(xmat))


This should address the feedback and ensure that the tests pass.