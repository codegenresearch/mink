from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import mink
from typing import List, Dict, Tuple, Sequence, Optional

_HERE = Path(__file__).parent
_XML = _HERE / "aloha" / "scene.xml"

# Single arm joint names.
_JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]

# Single arm velocity limits, taken from:
# https://github.com/Interbotix/interbotix_ros_manipulators/blob/main/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/vx300s.urdf.xacro
_VELOCITY_LIMITS = {k: np.pi for k in _JOINT_NAMES}

# Position and orientation thresholds
_POSITION_THRESHOLD = 1e-4
_ORIENTATION_THRESHOLD = 1e-4


def get_joint_and_actuator_ids(model: mujoco.MjModel, joint_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the DOF and actuator IDs for the specified joint names.

    Parameters:
    - model (mujoco.MjModel): The MuJoCo model.
    - joint_names (List[str]): List of joint names.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Arrays of DOF and actuator IDs.
    """
    dof_ids = [model.joint(name).id for name in joint_names]
    actuator_ids = [model.actuator(name).id for name in joint_names]
    return np.array(dof_ids), np.array(actuator_ids)


def setup_collision_avoidance(model: mujoco.MjModel) -> mink.CollisionAvoidanceLimit:
    """
    Setup collision avoidance for specified geom pairs.

    Parameters:
    - model (mujoco.MjModel): The MuJoCo model.

    Returns:
    - mink.CollisionAvoidanceLimit: Collision avoidance limit object.
    """
    left_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("left/wrist_link").id)
    right_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("right/wrist_link").id)
    left_geoms = mink.get_subtree_geom_ids(model, model.body("left/upper_arm_link").id)
    right_geoms = mink.get_subtree_geom_ids(model, model.body("right/upper_arm_link").id)
    frame_geoms = mink.get_body_geom_ids(model, model.body("metal_frame").id)
    collision_pairs = [
        (left_wrist_geoms, right_wrist_geoms),
        (left_geoms + right_geoms, frame_geoms + ["table"]),
    ]
    return mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,
        minimum_distance_from_collisions=0.05,
        collision_detection_distance=0.1,
    )


def setup_limits(model: mujoco.MjModel, velocity_limits: Dict[str, float], collision_avoidance_limit: mink.CollisionAvoidanceLimit) -> List[mink.Limit]:
    """
    Setup the limits for the configuration, velocity, and collision avoidance.

    Parameters:
    - model (mujoco.MjModel): The MuJoCo model.
    - velocity_limits (Dict[str, float]): Dictionary of joint names and their velocity limits.
    - collision_avoidance_limit (mink.CollisionAvoidanceLimit): Collision avoidance limit object.

    Returns:
    - List[mink.Limit]: List of limit objects.
    """
    return [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(model, velocity_limits),
        collision_avoidance_limit,
    ]


def initialize_mocap_targets(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """
    Initialize mocap targets at the end-effector sites.

    Parameters:
    - model (mujoco.MjModel): The MuJoCo model.
    - data (mujoco.MjData): The MuJoCo data.
    """
    mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
    mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")


def update_task_targets(model: mujoco.MjModel, data: mujoco.MjData, left_ee_task: mink.FrameTask, right_ee_task: mink.FrameTask) -> None:
    """
    Update the task targets for the left and right end-effectors.

    Parameters:
    - model (mujoco.MjModel): The MuJoCo model.
    - data (mujoco.MjData): The MuJoCo data.
    - left_ee_task (mink.FrameTask): Left end-effector task.
    - right_ee_task (mink.FrameTask): Right end-effector task.
    """
    left_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
    right_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))


def compute_velocity_and_integrate(
    configuration: mink.Configuration,
    tasks: List[mink.FrameTask],
    rate: RateLimiter,
    solver: str,
    limits: List[mink.Limit],
    damping: float,
    pos_threshold: float,
    ori_threshold: float,
    max_iters: int,
) -> bool:
    """
    Compute the velocity and integrate it into the next configuration.

    Parameters:
    - configuration (mink.Configuration): The current configuration.
    - tasks (List[mink.FrameTask]): List of frame tasks.
    - rate (RateLimiter): Rate limiter for controlling the loop frequency.
    - solver (str): Solver method for IK.
    - limits (List[mink.Limit]): List of limits for IK.
    - damping (float): Damping factor for IK.
    - pos_threshold (float): Position threshold for task achievement.
    - ori_threshold (float): Orientation threshold for task achievement.
    - max_iters (int): Maximum number of iterations for IK.

    Returns:
    - bool: True if tasks are achieved, False otherwise.
    """
    for _ in range(max_iters):
        vel = mink.solve_ik(
            configuration,
            tasks,
            rate.dt,
            solver,
            limits=limits,
            damping=damping,
        )
        configuration.integrate_inplace(vel, rate.dt)

        left_err = tasks[0].compute_error(configuration)
        right_err = tasks[1].compute_error(configuration)

        if (np.linalg.norm(left_err[:3]) <= pos_threshold and np.linalg.norm(left_err[3:]) <= ori_threshold and
            np.linalg.norm(right_err[:3]) <= pos_threshold and np.linalg.norm(right_err[3:]) <= ori_threshold):
            return True
    return False


def compensate_gravity(model: mujoco.MjModel, data: mujoco.MjData, left_subtree_id: int, right_subtree_id: int, qfrc_applied: Optional[np.ndarray] = None) -> None:
    """
    Compensate for gravity by computing the necessary forces for specified subtrees.

    Parameters:
    - model (mujoco.MjModel): The MuJoCo model.
    - data (mujoco.MjData): The MuJoCo data.
    - left_subtree_id (int): ID of the left wrist subtree.
    - right_subtree_id (int): ID of the right wrist subtree.
    - qfrc_applied (Optional[np.ndarray]): Array to store applied forces.
    """
    if qfrc_applied is None:
        qfrc_applied = np.zeros(model.nv)

    for subtree_id in [left_subtree_id, right_subtree_id]:
        wrist_geoms = mink.get_subtree_geom_ids(model, subtree_id)
        total_mass = sum(model.geom(geom_id).mass for geom_id in wrist_geoms)
        com_pos = np.mean([model.geom(geom_id).pos for geom_id in wrist_geoms], axis=0)
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jacSubtreeCom(model, data, jacobian, com_pos, subtree_id)
        qfrc_applied += total_mass * np.dot(jacobian.T, model.opt.gravity)

    data.qfrc_applied += qfrc_applied


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    joint_names = [f"{prefix}/{n}" for prefix in ["left", "right"] for n in _JOINT_NAMES]
    velocity_limits = {name: _VELOCITY_LIMITS[name.split('/')[-1]] for name in joint_names}

    dof_ids, actuator_ids = get_joint_and_actuator_ids(model, joint_names)

    configuration = mink.Configuration(model)
    tasks = [
        mink.FrameTask(
            frame_name="left/gripper",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        mink.FrameTask(
            frame_name="right/gripper",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        mink.PostureTask(model, cost=1e-4),
    ]

    collision_avoidance_limit = setup_collision_avoidance(model)
    limits = setup_limits(model, velocity_limits, collision_avoidance_limit)

    solver = "quadprog"
    max_iters = 2

    left_subtree_id = model.body("left/wrist_link").id
    right_subtree_id = model.body("right/wrist_link").id

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        tasks[-1].set_target_from_configuration(configuration)

        initialize_mocap_targets(model, data)

        rate = RateLimiter(frequency=200.0)
        while viewer.is_running():
            update_task_targets(model, data, tasks[0], tasks[1])

            if compute_velocity_and_integrate(
                configuration, tasks, rate, solver, limits, 1e-5, _POSITION_THRESHOLD, _ORIENTATION_THRESHOLD, max_iters
            ):
                break

            # Compensate for gravity
            compensate_gravity(model, data, left_subtree_id, right_subtree_id)

            data.ctrl[actuator_ids] = configuration.q[dof_ids] + data.qfrc_applied[actuator_ids]

            mujoco.mj_step(model, data)

            viewer.sync()
            rate.sleep()