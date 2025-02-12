from typing import Optional, List, Tuple

import mujoco
import numpy as np

from . import constants as consts
from .exceptions import InvalidKeyframe, InvalidMocapBody, MinkError


def move_mocap_to_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mocap_name: str,
    frame_name: str,
    frame_type: str,
) -> None:
    """Initialize mocap body pose at a desired frame.\n\n    Args:\n        model: Mujoco model.\n        data: Mujoco data.\n        mocap_name: The name of the mocap body.\n        frame_name: The desired frame name.\n        frame_type: The desired frame type. Can be "body", "geom", or "site".\n    """
    mocap_body = model.body(mocap_name)
    if mocap_body.mocapid[0] == -1:
        raise InvalidMocapBody(mocap_name, model)

    frame_id = mujoco.mj_name2id(model, consts.FRAME_TO_ENUM[frame_type], frame_name)
    if frame_id == -1:
        raise InvalidFrame(frame_name, frame_type, model)

    frame_pos = getattr(data, consts.FRAME_TO_POS_ATTR[frame_type])[frame_id]
    frame_xmat = getattr(data, consts.FRAME_TO_XMAT_ATTR[frame_type])[frame_id]

    data.mocap_pos[mocap_body.mocapid[0]] = frame_pos.copy()
    mujoco.mju_mat2Quat(data.mocap_quat[mocap_body.mocapid[0]], frame_xmat)


def get_freejoint_dims(model: mujoco.MjModel) -> Tuple[List[int], List[int]]:
    """Get all floating joint configuration and tangent indices.\n\n    Args:\n        model: Mujoco model.\n\n    Returns:\n        A (q_ids, v_ids) pair containing all floating joint indices in the\n        configuration and tangent spaces respectively.\n    """
    q_ids: List[int] = []
    v_ids: List[int] = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            qadr = model.jnt_qposadr[j]
            vadr = model.jnt_dofadr[j]
            q_ids.extend(range(qadr, qadr + 7))
            v_ids.extend(range(vadr, vadr + 6))
    return q_ids, v_ids


def custom_configuration_vector(
    model: mujoco.MjModel,
    key_name: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    """Generate a configuration vector where named joints have specific values.\n\n    Args:\n        model: Mujoco model.\n        key_name: Optional keyframe name to initialize the configuration vector from.\n            Otherwise, the default pose `qpos0` is used.\n        kwargs: Custom values for joint coordinates.\n\n    Returns:\n        Configuration vector where named joints have the values specified in\n            keyword arguments, and other joints have their neutral value or value\n            defined in the keyframe if provided.\n    """
    data = mujoco.MjData(model)
    if key_name:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        if key_id == -1:
            raise InvalidKeyframe(key_name, model)
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)

    q = data.qpos.copy()
    for name, value in kwargs.items():
        joint = model.joint(name)
        qid = model.jnt_qposadr[joint.id]
        value = np.atleast_1d(value)
        expected_dim = consts.qpos_width(model.jnt_type[joint.id])
        if value.shape != (expected_dim,):
            raise ValueError(
                f"Joint {name} should have a qpos value of shape {expected_dim} but "
                f"got {value.shape}"
            )
        q[qid : qid + expected_dim] = value
    return q


def get_subtree_geom_ids(model: mujoco.MjModel, body_id: int) -> List[int]:
    """Get all geoms belonging to subtree starting at a given body.\n\n    Args:\n        model: Mujoco model.\n        body_id: ID of body where subtree starts.\n\n    Returns:\n        A list containing all subtree geom ids.\n    """

    def gather_geoms(current_body_id: int) -> List[int]:
        geoms: List[int] = []
        geom_start = model.body_geomadr[current_body_id]
        geom_end = geom_start + model.body_geomnum[current_body_id]
        geoms.extend(range(geom_start, geom_end))
        children = [
            i for i in range(model.nbody) if model.body_parentid[i] == current_body_id
        ]
        for child_id in children:
            geoms.extend(gather_geoms(child_id))
        return geoms

    return gather_geoms(body_id)


def get_body_geom_ids(model: mujoco.MjModel, body_id: int) -> List[int]:
    """Get all geoms belonging to a given body.\n\n    Args:\n        model: Mujoco model.\n        body_id: ID of body.\n\n    Returns:\n        A list containing all body geom ids.\n    """
    geom_start = model.body_geomadr[body_id]
    geom_end = geom_start + model.body_geomnum[body_id]
    return list(range(geom_start, geom_end))


def apply_gravity_compensation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gravity: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Calculate and apply gravity compensation torques.\n\n    Args:\n        model: Mujoco model.\n        data: Mujoco data.\n        gravity: Optional custom gravity vector (3 elements). If None, uses the model's gravity.\n\n    Returns:\n        Array of gravity compensation torques.\n    """
    if gravity is None:
        gravity = model.opt.gravity

    # Ensure the gravity vector has the correct shape
    if gravity.shape != (3,):
        raise ValueError("Gravity vector must have 3 elements.")

    mujoco.mj_kinematics(model, data)
    mujoco.mj_crb(model, data)

    # Calculate the gravity compensation torques
    mujoco.mj_rne(model, data, gravity, np.zeros(6), data.qfrc_bias)

    # Extract the bias forces (gravity compensation torques)
    gravity_compensation_torques = data.qfrc_bias.copy()

    # Apply the gravity compensation torques
    data.qfrc_applied += gravity_compensation_torques

    return gravity_compensation_torques