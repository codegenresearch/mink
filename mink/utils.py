from typing import Optional, List, Set

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
    """Initialize mocap body pose at a desired frame.

    Args:
        model: Mujoco model.
        data: Mujoco data.
        mocap_name: The name of the mocap body.
        frame_name: The desired frame name.
        frame_type: The desired frame type. Can be "body", "geom", or "site".
    """
    mocap_id = model.body(mocap_name).mocapid[0]
    if mocap_id == -1:
        raise InvalidMocapBody(mocap_name, model)

    obj_id = mujoco.mj_name2id(model, consts.FRAME_TO_ENUM[frame_type], frame_name)
    xpos = getattr(data, consts.FRAME_TO_POS_ATTR[frame_type])[obj_id]
    xmat = getattr(data, consts.FRAME_TO_XMAT_ATTR[frame_type])[obj_id]

    data.mocap_pos[mocap_id] = xpos.copy()
    mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], xmat)


def get_freejoint_dims(model: mujoco.MjModel) -> tuple[list[int], list[int]]:
    """Get all floating joint configuration and tangent indices.

    Args:
        model: Mujoco model.

    Returns:
        A (q_ids, v_ids) pair containing all floating joint indices in the
        configuration and tangent spaces respectively.
    """
    q_ids: list[int] = []
    v_ids: list[int] = []
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
    """Generate a configuration vector where named joints have specific values.

    Args:
        model: Mujoco model.
        key_name: Optional keyframe name to initialize the configuration vector from.
            Otherwise, the default pose `qpos0` is used.
        kwargs: Custom values for joint coordinates.

    Returns:
        Configuration vector where named joints have the values specified in
            keyword arguments, and other joints have their neutral value or value
            defined in the keyframe if provided.
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
            raise ValueError(
                f"Joint {name} should have a qpos value of {jnt_dim,} but "
                f"got {value.shape}"
            )
        q[qid : qid + jnt_dim] = value
    return q


def get_subtree_geom_ids(model: mujoco.MjModel, body_id: int) -> list[int]:
    """Get all geoms belonging to the subtree starting at a given body.

    Args:
        model: Mujoco model.
        body_id: ID of the body where the subtree starts.

    Returns:
        A list of geom IDs belonging to the subtree.
    """
    geom_ids: list[int] = []
    stack = [body_id]
    visited = set()

    while stack:
        current_body_id = stack.pop()
        if current_body_id in visited:
            continue
        visited.add(current_body_id)

        geom_start = model.body_geomadr[current_body_id]
        geom_end = geom_start + model.body_geomnum[current_body_id]
        geom_ids.extend(range(geom_start, geom_end))

        children = [
            i for i in range(model.nbody) if model.body_parentid[i] == current_body_id
        ]
        stack.extend(children)

    return geom_ids


def get_body_geom_ids(model: mujoco.MjModel, body_id: int) -> list[int]:
    """Get all geoms belonging to a given body.

    Args:
        model: Mujoco model.
        body_id: ID of the body.

    Returns:
        A list of geom IDs belonging to the body.
    """
    geom_start = model.body_geomadr[body_id]
    geom_end = geom_start + model.body_geomnum[body_id]
    return list(range(geom_start, geom_end))


def check_empty_geoms(model: mujoco.MjModel) -> bool:
    """Check if there are any geoms in the model.

    Args:
        model: Mujoco model.

    Returns:
        True if there are no geoms, False otherwise.
    """
    return model.ngeom == 0


def get_multiple_body_geom_ids(model: mujoco.MjModel, body_ids: Set[int]) -> list[int]:
    """Get all geoms belonging to multiple bodies.

    Args:
        model: Mujoco model.
        body_ids: Set of body IDs.

    Returns:
        A list of geom IDs for the specified bodies.
    """
    geom_ids: list[int] = []
    for body_id in body_ids:
        geom_ids.extend(get_body_geom_ids(model, body_id))
    return geom_ids


def apply_gravity_compensation(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Apply gravity compensation to the model.

    Args:
        model: Mujoco model.
        data: Mujoco data.
    """
    mujoco.mj_step1(model, data)
    mujoco.mj_inverse(model, data)
    mujoco.mj_step2(model, data)


def get_subtree_body_ids(model: mujoco.MjModel, body_id: int) -> list[int]:
    """Get all body IDs belonging to the subtree starting at a given body.

    Args:
        model: Mujoco model.
        body_id: ID of the body where the subtree starts.

    Returns:
        A list of body IDs belonging to the subtree.
    """
    body_ids: list[int] = []
    stack = [body_id]
    visited = set()

    while stack:
        current_body_id = stack.pop()
        if current_body_id in visited:
            continue
        visited.add(current_body_id)
        body_ids.append(current_body_id)

        children = [
            i for i in range(model.nbody) if model.body_parentid[i] == current_body_id
        ]
        stack.extend(children)

    return body_ids


def get_body_body_ids(model: mujoco.MjModel, body_id: int) -> list[int]:
    """Get all immediate child body IDs of a given body.

    Args:
        model: Mujoco model.
        body_id: ID of the body.

    Returns:
        A list of immediate child body IDs.
    """
    return [
        i for i in range(model.nbody) if model.body_parentid[i] == body_id and i != body_id
    ]


### Key Changes Made:
1. **Syntax Error Fix**: Removed any extraneous text or improperly formatted comments/documentation strings that could cause syntax errors.
2. **Function Documentation**: Ensured that docstrings are consistent in style and detail.
3. **Variable Naming**: Reviewed and ensured variable names are clear and concise.
4. **Logic and Structure**: Streamlined logic in `get_body_body_ids` and `get_subtree_body_ids` functions.
5. **Error Handling**: Ensured consistent error handling across functions.
6. **Type Hinting Consistency**: Used `list[int]` consistently for type hints.
7. **Redundant Code**: Looked for and removed any redundant code.
8. **Formatting**: Ensured code adheres to PEP 8 standards, including line lengths and spacing.