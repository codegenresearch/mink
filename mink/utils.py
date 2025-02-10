from typing import Optional, List, Tuple, Dict

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
    """Set the mocap body pose to match a specified frame.

    Args:
        model: Mujoco model.
        data: Mujoco data.
        mocap_name: Name of the mocap body.
        frame_name: Name of the target frame.
        frame_type: Type of the target frame ("body", "geom", or "site").
    """
    mocap_id = model.body(mocap_name).mocapid[0]
    if mocap_id == -1:
        raise InvalidMocapBody(mocap_name, model)

    frame_id = mujoco.mj_name2id(model, consts.FRAME_TO_ENUM[frame_type], frame_name)
    if frame_id == -1:
        raise ValueError(f"Frame '{frame_name}' of type '{frame_type}' not found in the model.")

    frame_pos = getattr(data, consts.FRAME_TO_POS_ATTR[frame_type])[frame_id]
    frame_xmat = getattr(data, consts.FRAME_TO_XMAT_ATTR[frame_type])[frame_id]

    data.mocap_pos[mocap_id] = frame_pos.copy()
    mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], frame_xmat)


def get_freejoint_dims(model: mujoco.MjModel) -> Tuple[List[int], List[int]]:
    """Retrieve indices of all floating joint configuration and tangent spaces.

    Args:
        model: Mujoco model.

    Returns:
        A tuple (q_ids, v_ids) with lists of indices for floating joints in configuration and tangent spaces.
    """
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
    **kwargs: float,
) -> np.ndarray:
    """Create a configuration vector with specific joint values.

    Args:
        model: Mujoco model.
        key_name: Optional keyframe name to initialize the configuration vector.
        **kwargs: Custom values for joint coordinates.

    Returns:
        Configuration vector with specified joint values and default values for others.
    """
    data = mujoco.MjData(model)
    if key_name:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        if key_id == -1:
            raise InvalidKeyframe(key_name, model)
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)

    q = data.qpos.copy()
    for joint_name, value in kwargs.items():
        joint = model.joint(joint_name)
        qid = joint.qposadr
        q[qid:qid + joint.dof] = np.atleast_1d(value)[:joint.dof]
    return q


def get_subtree_geom_ids(model: mujoco.MjModel, body_id: int) -> List[int]:
    """Collect all geom IDs in the subtree starting from a given body.

    Args:
        model: Mujoco model.
        body_id: ID of the starting body.

    Returns:
        List of geom IDs in the subtree.
    """
    def gather_geoms(current_body_id: int) -> List[int]:
        geoms = list(range(model.body_geomadr[current_body_id], model.body_geomadr[current_body_id] + model.body_geomnum[current_body_id]))
        for child_id in [i for i in range(model.nbody) if model.body_parentid[i] == current_body_id]:
            geoms.extend(gather_geoms(child_id))
        return geoms

    return gather_geoms(body_id)


def get_body_geom_ids(model: mujoco.MjModel, body_id: int) -> List[int]:
    """Retrieve all geom IDs associated with a specific body.

    Args:
        model: Mujoco model.
        body_id: ID of the body.

    Returns:
        List of geom IDs for the body.
    """
    geom_start = model.body_geomadr[body_id]
    geom_end = geom_start + model.body_geomnum[body_id]
    return list(range(geom_start, geom_end))


def apply_gravity_compensation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q: np.ndarray,
    qdot: np.ndarray,
) -> np.ndarray:
    """Compensate for gravity in the given configuration.

    Args:
        model: Mujoco model.
        data: Mujoco data.
        q: Configuration vector.
        qdot: Velocity vector.

    Returns:
        Gravity-compensated torque vector.
    """
    data.qpos[:] = q
    data.qvel[:] = qdot
    mujoco.mj_kinematics(model, data)
    mujoco.mj_crb(model, data)
    mujoco.mj_rneunreduced(model, data)
    return data.qfrc_bias.copy()


def get_joint_limits(model: mujoco.MjModel) -> Dict[str, Tuple[float, float]]:
    """Retrieve the limits for each joint in the model.

    Args:
        model: Mujoco model.

    Returns:
        Dictionary mapping joint names to their (lower, upper) limits.
    """
    joint_limits = {}
    for j in range(model.njnt):
        joint_name = model.joint(j).name
        if model.jnt_limited[j]:
            lower_limit = model.jnt_range[j, 0]
            upper_limit = model.jnt_range[j, 1]
            joint_limits[joint_name] = (lower_limit, upper_limit)
    return joint_limits