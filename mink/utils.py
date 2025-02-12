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
    mocap_id = model.body(mocap_name).mocapid[0]
    if mocap_id == -1:
        raise InvalidMocapBody(mocap_name, model)

    obj_id = mujoco.mj_name2id(model, consts.FRAME_TO_ENUM[frame_type], frame_name)
    if obj_id == -1:
        raise MinkError(f"Frame '{frame_name}' of type '{frame_type}' not found in the model.")

    xpos = getattr(data, consts.FRAME_TO_POS_ATTR[frame_type])[obj_id]
    xmat = getattr(data, consts.FRAME_TO_XMAT_ATTR[frame_type])[obj_id]

    data.mocap_pos[mocap_id] = xpos.copy()
    mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], xmat)


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
        jid = model.joint(name).id
        jnt_dim = consts.qpos_width(model.jnt_type[jid])
        qid = model.jnt_qposadr[jid]
        value = np.atleast_1d(value)
        if value.shape != (jnt_dim,):
            raise ValueError(
                f"Joint {name} should have a qpos value of {jnt_dim} but got {value.shape}"
            )
        q[qid : qid + jnt_dim] = value
    return q


def get_subtree_geom_ids(model: mujoco.MjModel, body_id: int) -> List[int]:
    """Get all geoms belonging to subtree starting at a given body.\n\n    Args:\n        model: Mujoco model.\n        body_id: ID of body where subtree starts.\n\n    Returns:\n        A list containing all subtree geom ids.\n    """
    def gather_geoms(body_id: int) -> List[int]:
        geoms: List[int] = []
        geom_start = model.body_geomadr[body_id]
        geom_end = geom_start + model.body_geomnum[body_id]
        geoms.extend(range(geom_start, geom_end))
        children = [i for i in range(model.nbody) if model.body_parentid[i] == body_id]
        for child_id in children:
            geoms.extend(gather_geoms(child_id))
        return geoms

    return gather_geoms(body_id)


def get_body_geom_ids(model: mujoco.MjModel, body_id: int) -> List[int]:
    """Get all geoms belonging to a given body.\n\n    Args:\n        model: Mujoco model.\n        body_id: ID of body.\n\n    Returns:\n        A list containing all body geom ids.\n    """
    geom_start = model.body_geomadr[body_id]
    geom_end = geom_start + model.body_geomnum[body_id]
    return list(range(geom_start, geom_end))


def integrate_with_gravity(
    model: mujoco.MjModel,
    q: np.ndarray,
    v: np.ndarray,
    dt: float,
    gravity_compensation: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate a velocity starting from the current configuration with optional gravity compensation.\n\n    Args:\n        model: Mujoco model.\n        q: The current configuration vector.\n        v: The current velocity vector.\n        dt: Integration duration in [s].\n        gravity_compensation: If True, apply gravity compensation.\n\n    Returns:\n        The new configuration and velocity after integration.\n    """
    q_next = q.copy()
    v_next = v.copy()
    mujoco.mj_step1(model, data)
    if gravity_compensation:
        # Apply gravity compensation
        mujoco.mj_inverse(model, data)
        v_next += data.qfrc_bias / model.opt.timestep
    mujoco.mj_step2(model, data)
    mujoco.mj_integratePos(model, q_next, v_next, dt)
    return q_next, v_next