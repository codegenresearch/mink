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
    """Set the mocap body pose to match a specified frame.\n\n    Args:\n        model: Mujoco model.\n        data: Mujoco data.\n        mocap_name: Name of the mocap body.\n        frame_name: Name of the target frame.\n        frame_type: Type of the target frame ("body", "geom", or "site").\n    """
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
    """Retrieve indices of all floating joints in configuration and tangent spaces.\n\n    Args:\n        model: Mujoco model.\n\n    Returns:\n        A tuple (q_ids, v_ids) where q_ids are indices of floating joints in the configuration space,\n        and v_ids are indices in the tangent space.\n    """
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
    """Create a configuration vector with specific joint values.\n\n    Args:\n        model: Mujoco model.\n        key_name: Optional keyframe name to initialize the configuration vector.\n            If None, the default pose `qpos0` is used.\n        **kwargs: Custom values for joint coordinates.\n\n    Returns:\n        Configuration vector with specified joint values and default values for others.\n    """
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
            raise ValueError(
                f"Joint {name} should have a qpos value of shape {jnt_dim,} but got {value.shape}"
            )
        q[qid : qid + jnt_dim] = value
    return q


def get_subtree_geom_ids(model: mujoco.MjModel, body_id: int) -> List[int]:
    """Retrieve all geom IDs in the subtree starting from a given body.\n\n    Args:\n        model: Mujoco model.\n        body_id: ID of the starting body.\n\n    Returns:\n        List of geom IDs in the subtree.\n    """

    def gather_geoms(current_body_id: int) -> List[int]:
        geoms: List[int] = []
        geom_start = model.body_geomadr[current_body_id]
        geom_end = geom_start + model.body_geomnum[current_body_id]
        geoms.extend(range(geom_start, geom_end))
        children = [i for i in range(model.nbody) if model.body_parentid[i] == current_body_id]
        for child_id in children:
            geoms.extend(gather_geoms(child_id))
        return geoms

    return gather_geoms(body_id)


def get_body_geom_ids(model: mujoco.MjModel, body_id: int) -> List[int]:
    """Retrieve all geom IDs associated with a specific body.\n\n    Args:\n        model: Mujoco model.\n        body_id: ID of the body.\n\n    Returns:\n        List of geom IDs for the specified body.\n    """
    geom_start = model.body_geomadr[body_id]
    geom_end = geom_start + model.body_geomnum[body_id]
    return list(range(geom_start, geom_end))


def apply_gravity_compensation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gravity: np.ndarray = np.array([0.0, 0.0, -9.81]),
) -> np.ndarray:
    """Compute gravity compensation torques for the model.\n\n    Args:\n        model: Mujoco model.\n        data: Mujoco data.\n        gravity: Gravity vector (default is Earth's gravity).\n\n    Returns:\n        Array of gravity compensation torques.\n    """
    mujoco.mj_setConst(model, data)
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_rnePostConstraint(model, data)
    mujoco.mj_crb(model, data)
    mujoco.mj_passive(model, data)

    # Set gravity
    model.opt.gravity = gravity

    # Compute gravity compensation torques
    mujoco.mj_rne(model, data)
    return data.qfrc_passive.copy()


def get_joint_limits(model: mujoco.MjModel) -> Dict[str, Tuple[float, float]]:
    """Retrieve the position limits for all joints in the model.\n\n    Args:\n        model: Mujoco model.\n\n    Returns:\n        Dictionary mapping joint names to their (lower, upper) position limits.\n    """
    joint_limits = {}
    for j in range(model.njnt):
        joint_name = model.joint(j).name
        if model.jnt_limited[j]:
            lower_limit = model.jnt_range[j, 0]
            upper_limit = model.jnt_range[j, 1]
            joint_limits[joint_name] = (lower_limit, upper_limit)
        else:
            joint_limits[joint_name] = (float('-inf'), float('inf'))
    return joint_limits