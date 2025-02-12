from typing import Optional, Set

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
    """Initialize mocap body pose at a desired frame.\n\n    Args:\n        model: Mujoco model.\n        data: Mujoco data.\n        mocap_name: The name of the mocap body.\n        frame_name: The desired frame name.\n        frame_type: The desired frame type. Can be "body", "geom" or "site".\n    """
    mocap_id = model.body(mocap_name).mocapid[0]
    if mocap_id == -1:
        raise InvalidMocapBody(mocap_name, model)

    obj_id = mujoco.mj_name2id(model, consts.FRAME_TO_ENUM[frame_type], frame_name)
    xpos = getattr(data, consts.FRAME_TO_POS_ATTR[frame_type])[obj_id]
    xmat = getattr(data, consts.FRAME_TO_XMAT_ATTR[frame_type])[obj_id]

    data.mocap_pos[mocap_id] = xpos.copy()
    mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], xmat)


def get_freejoint_dims(model: mujoco.MjModel) -> tuple[Set[int], Set[int]]:
    """Get all floating joint configuration and tangent indices.\n\n    Args:\n        model: Mujoco model.\n\n    Returns:\n        A (q_ids, v_ids) pair containing all floating joint indices in the\n        configuration and tangent spaces respectively.\n    """
    q_ids: Set[int] = set()
    v_ids: Set[int] = set()
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            qadr = model.jnt_qposadr[j]
            vadr = model.jnt_dofadr[j]
            q_ids.update(range(qadr, qadr + 7))
            v_ids.update(range(vadr, vadr + 6))
    return q_ids, v_ids


def custom_configuration_vector(
    model: mujoco.MjModel,
    key_name: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    """Generate a configuration vector where named joints have specific values.\n\n    Args:\n        model: Mujoco model.\n        key_name: Optional keyframe name to initialize the configuration vector from.\n            Otherwise, the default pose `qpos0` is used.\n        kwargs: Custom values for joint coordinates.\n\n    Returns:\n        Configuration vector where named joints have the values specified in\n            keyword arguments, and other joints have their neutral value or value\n            defined in the keyframe if provided.\n    """
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
                f"Joint {name} should have a qpos value of {jnt_dim,} but "
                f"got {value.shape}"
            )
        q[qid : qid + jnt_dim] = value
    return q


def get_subtree_geom_ids(model: mujoco.MjModel, body_id: int) -> Set[int]:
    """Get all geoms belonging to subtree starting at a given body.\n\n    Args:\n        model: Mujoco model.\n        body_id: ID of body where subtree starts.\n\n    Returns:\n        A set containing all subtree geom ids.\n    """

    def gather_geoms(body_id: int) -> Set[int]:
        geoms: Set[int] = set()
        geom_start = model.body_geomadr[body_id]
        geom_end = geom_start + model.body_geomnum[body_id]
        geoms.update(range(geom_start, geom_end))
        children = {i for i in range(model.nbody) if model.body_parentid[i] == body_id}
        for child_id in children:
            geoms.update(gather_geoms(child_id))
        return geoms

    return gather_geoms(body_id)


def get_body_geom_ids(model: mujoco.MjModel, body_id: int) -> Set[int]:
    """Get all geoms belonging to a given body.\n\n    Args:\n        model: Mujoco model.\n        body_id: ID of body.\n\n    Returns:\n        A set containing all body geom ids.\n    """
    geom_start = model.body_geomadr[body_id]
    geom_end = geom_start + model.body_geomnum[body_id]
    geom_ids = set(range(geom_start, geom_end))
    if not geom_ids:
        raise ValueError(f"No geometries found for body with ID {body_id}.")
    return geom_ids


def apply_gravity_compensation(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Apply gravity compensation to the model.\n\n    Args:\n        model: Mujoco model.\n        data: Mujoco data.\n    """
    mujoco.mj_step1(model, data)
    mujoco.mj_inverse(model, data)
    mujoco.mj_com(model, data)
    mujoco.mj_tendon(model, data)
    mujoco.mj_crb(model, data)
    mujoco.mj_factorM(model, data)
    mujoco.mj_mulM(model, data.qfrc_passive, data.qfrc_passive)
    data.qfrc_applied += data.qfrc_passive


def check_multiple_bodies_for_geometry(model: mujoco.MjModel, body_names: Set[str]) -> None:
    """Check if multiple bodies have associated geometry.\n\n    Args:\n        model: Mujoco model.\n        body_names: Set of body names to check.\n\n    Raises:\n        ValueError: If any body does not have associated geometry.\n    """
    for body_name in body_names:
        body_id = model.body(body_name).id
        geom_ids = get_body_geom_ids(model, body_id)
        if not geom_ids:
            raise ValueError(f"Body '{body_name}' does not have any associated geometry.")