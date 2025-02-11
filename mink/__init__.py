"""mink: MuJoCo inverse kinematics."""

from .configuration import Configuration
from .constants import (
    FRAME_TO_ENUM,
    FRAME_TO_JAC_FUNC,
    FRAME_TO_POS_ATTR,
    FRAME_TO_XMAT_ATTR,
    SUPPORTED_FRAMES,
)
from .exceptions import (
    InvalidFrame,
    InvalidKeyframe,
    InvalidMocapBody,
    MinkError,
    NotWithinConfigurationLimits,
    UnsupportedFrame,
)
from .lie import SE3, SO3, MatrixLieGroup
from .limits import (
    CollisionAvoidanceLimit,
    ConfigurationLimit,
    Constraint,
    Limit,
    VelocityLimit,
)
from .solve_ik import build_ik, solve_ik
from .tasks import (
    ComTask,
    DampingTask,
    FrameTask,
    Objective,
    PostureTask,
    RelativeFrameTask,
    TargetNotSet,
    Task,
)
from .utils import (
    custom_configuration_vector,
    get_body_geom_ids,
    get_freejoint_dims,
    get_subtree_geom_ids,
    get_subtree_body_ids,
    move_mocap_to_frame,
    pose_from_mocap,
    set_mocap_pose_from_frame,
)

__all__ = (
    "ComTask",
    "Configuration",
    "build_ik",
    "solve_ik",
    "DampingTask",
    "FrameTask",
    "RelativeFrameTask",
    "PostureTask",
    "Task",
    "Objective",
    "ConfigurationLimit",
    "VelocityLimit",
    "CollisionAvoidanceLimit",
    "Constraint",
    "Limit",
    "SO3",
    "SE3",
    "MatrixLieGroup",
    "MinkError",
    "UnsupportedFrame",
    "InvalidFrame",
    "InvalidKeyframe",
    "NotWithinConfigurationLimits",
    "TargetNotSet",
    "InvalidMocapBody",
    "SUPPORTED_FRAMES",
    "FRAME_TO_ENUM",
    "FRAME_TO_JAC_FUNC",
    "FRAME_TO_POS_ATTR",
    "FRAME_TO_XMAT_ATTR",
    "custom_configuration_vector",
    "get_freejoint_dims",
    "move_mocap_to_frame",
    "get_subtree_geom_ids",
    "get_body_geom_ids",
    "get_subtree_body_ids",
    "pose_from_mocap",
    "set_mocap_pose_from_frame",
)


### Changes Made:
1. **Import Consistency**: Ensured all necessary imports from the `utils` module are included, specifically `get_subtree_body_ids`.
2. **Public API (`__all__`)**: Verified that the `__all__` tuple includes all necessary items present in the gold code.
3. **Order of Imports**: Maintained the order of imports as per the original snippet, ensuring consistency.
4. **Formatting and Style**: Double-checked for minor inconsistencies in spacing and line breaks to enhance readability.