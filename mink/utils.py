from typing import Optional, List, Tuple

import mujoco
import numpy as np

from . import constants as consts
from .exceptions import InvalidKeyframe, InvalidMocapBody


def move_mocap_to_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mocap_name: str,
    frame_name: str,
    frame_type: str,
) -> None:
    """Set mocap body pose to match a specified frame.

    Args:
        model: Mujoco model.
        data: Mujoco data.
        mocap_name: Name of the mocap body.
        frame_name: Name of the target frame.
        frame_type: Type of the target frame ("body", "geom", or "site").

    Raises:
        InvalidMocapBody: If the mocap body does not exist.
        ValueError: If the frame type is unsupported or the frame does not exist.
    """
    mocap_id = model.body(mocap_name).mocapid[0]
    if mocap_id == -1:
        raise InvalidMocapBody(mocap_name, model)

    if frame_type not in consts.SUPPORTED_FRAMES:
        raise ValueError(f"Unsupported frame type: {frame_type}")

    obj_id = mujoco.mj_name2id(model, consts.FRAME_TO_ENUM[frame_type], frame_name)
    if obj_id == -1:
        raise ValueError(f"Frame '{frame_name}' of type '{frame_type}' not found.")

    xpos = getattr(data, consts.FRAME_TO_POS_ATTR[frame_type])[obj_id]
    xmat = getattr(data, consts.FRAME_TO_XMAT_ATTR[frame_type])[obj_id]

    data.mocap_pos[mocap_id] = xpos.copy()
    mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], xmat)


def get_freejoint_dims(model: mujoco.MjModel) -> Tuple[List[int], List[int]]:
    """Get indices of all floating joints in configuration and tangent spaces.

    Args:
        model: Mujoco model.

    Returns:
        Tuple of lists (q_ids, v_ids) for configuration and tangent space indices.
    """
    q_ids = []
    v_ids = []
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
    """Create a configuration vector with specific joint values.

    Args:
        model: Mujoco model.
        key_name: Optional keyframe name to initialize the configuration vector.
            If None, the default pose `qpos0` is used.
        **kwargs: Custom values for joint coordinates.

    Returns:
        Configuration vector with specified joint values.

    Raises:
        InvalidKeyframe: If the keyframe does not exist.
        ValueError: If a joint value does not match the expected dimension.
    """
    data = mujoco.MjData(model)
    if key_name is not None:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        if key_id == -1:
            raise InvalidKeyframe(key_name, model)
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)

    q = data.qpos.copy()
    for name, value in kwargs.items():
        jid = model.joint(name).id
        jnt_dim = consts.qpos_width(model.jnt_type[jid])
        qid = model.jnt_qposadr[jid]
        value = np.atleast_1d(value)
        if value.shape != (jnt_dim,):
            raise ValueError(f"Joint {name} should have a qpos value of shape {jnt_dim,}")
        q[qid : qid + jnt_dim] = value
    return q


def get_subtree_geom_ids(model: mujoco.MjModel, body_id: int) -> List[int]:
    """Get all geom IDs in the subtree starting at a given body.

    Args:
        model: Mujoco model.
        body_id: ID of the body where the subtree starts.

    Returns:
        List of geom IDs in the subtree.
    """
    geom_ids = []
    stack = [body_id]
    while stack:
        current_body_id = stack.pop()
        geom_start = model.body_geomadr[current_body_id]
        geom_end = geom_start + model.body_geomnum[current_body_id]
        geom_ids.extend(range(geom_start, geom_end))
        stack.extend(
            child_id
            for child_id in range(model.nbody)
            if model.body_parentid[child_id] == current_body_id
        )
    return geom_ids


def get_body_geom_ids(model: mujoco.MjModel, body_id: int) -> List[int]:
    """Get all geom IDs for a specific body.

    Args:
        model: Mujoco model.
        body_id: ID of the body.

    Returns:
        List of geom IDs for the body.
    """
    geom_start = model.body_geomadr[body_id]
    geom_end = geom_start + model.body_geomnum[body_id]
    return list(range(geom_start, geom_end))


def get_subtree_body_ids(model: mujoco.MjModel, body_id: int) -> List[int]:
    """Get all body IDs in the subtree starting at a given body.

    Args:
        model: Mujoco model.
        body_id: ID of the body where the subtree starts.

    Returns:
        List of body IDs in the subtree.
    """
    body_ids = []
    stack = [body_id]
    while stack:
        current_body_id = stack.pop()
        body_ids.append(current_body_id)
        stack.extend(
            child_id
            for child_id in range(model.nbody)
            if model.body_parentid[child_id] == current_body_id
        )
    return body_ids


def apply_gravity_compensation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q: np.ndarray,
    qdot: np.ndarray,
) -> np.ndarray:
    """Compute gravity compensation torques.

    Args:
        model: Mujoco model.
        data: Mujoco data.
        q: Configuration vector.
        qdot: Velocity vector.

    Returns:
        Gravity compensation torques.
    """
    data.qpos[:] = q
    data.qvel[:] = qdot
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_rneUnconstrained(model, data)
    return data.qfrc_bias.copy()