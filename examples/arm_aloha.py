from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import mink
from typing import List, Dict, Tuple, Sequence

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
_POSITION_THRESHOLD = 1e-2
_ORIENTATION_THRESHOLD = 1e-2


def get_joint_and_actuator_ids(model: mujoco.MjModel, joint_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the degree of freedom (DOF) and actuator IDs for the specified joint names.

    Args:
        model (mujoco.MjModel): The MuJoCo model.
        joint_names (List[str]): List of joint names.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of DOF and actuator IDs.
    """
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])
    return dof_ids, actuator_ids


def initialize_tasks(model: mujoco.MjModel) -> List[mink.FrameTask]:
    """
    Initialize the tasks for the left and right end-effectors and posture.

    Args:
        model (mujoco.MjModel): The MuJoCo model.

    Returns:
        List[mink.FrameTask]: List of initialized tasks.
    """
    return [
        (l_ee_task := mink.FrameTask(
            frame_name="left/gripper",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )),
        (r_ee_task := mink.FrameTask(
            frame_name="right/gripper",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )),
        (posture_task := mink.PostureTask(model, cost=1e-4)),
    ]


def setup_collision_avoidance(model: mujoco.MjModel) -> mink.CollisionAvoidanceLimit:
    """
    Set up collision avoidance for the specified geoms.

    Args:
        model (mujoco.MjModel): The MuJoCo model.

    Returns:
        mink.CollisionAvoidanceLimit: Configured collision avoidance limit.
    """
    l_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("left/wrist_link").id)
    r_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("right/wrist_link").id)
    l_geoms = mink.get_subtree_geom_ids(model, model.body("left/upper_arm_link").id)
    r_geoms = mink.get_subtree_geom_ids(model, model.body("right/upper_arm_link").id)
    frame_geoms = mink.get_body_geom_ids(model, model.body("metal_frame").id)
    collision_pairs = [
        (l_wrist_geoms, r_wrist_geoms),
        (l_geoms + r_geoms, frame_geoms + ["table"]),
    ]
    return mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,
        minimum_distance_from_collisions=0.05,
        collision_detection_distance=0.1,
    )


def setup_limits(model: mujoco.MjModel, velocity_limits: Dict[str, float], collision_avoidance_limit: mink.CollisionAvoidanceLimit) -> List[mink.Limit]:
    """
    Set up the configuration limits, velocity limits, and collision avoidance limits.

    Args:
        model (mujoco.MjModel): The MuJoCo model.
        velocity_limits (Dict[str, float]): Dictionary of joint names and their velocity limits.
        collision_avoidance_limit (mink.CollisionAvoidanceLimit): Configured collision avoidance limit.

    Returns:
        List[mink.Limit]: List of configuration limits.
    """
    return [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(model, velocity_limits),
        collision_avoidance_limit,
    ]


def initialize_mocap_targets(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """
    Initialize the mocap targets at the end-effector sites.

    Args:
        model (mujoco.MjModel): The MuJoCo model.
        data (mujoco.MjData): The MuJoCo data.
    """
    mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
    mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")


def update_task_targets(model: mujoco.MjModel, data: mujoco.MjData, l_ee_task: mink.FrameTask, r_ee_task: mink.FrameTask) -> None:
    """
    Update the task targets for the left and right end-effectors.

    Args:
        model (mujoco.MjModel): The MuJoCo model.
        data (mujoco.MjData): The MuJoCo data.
        l_ee_task (mink.FrameTask): Left end-effector task.
        r_ee_task (mink.FrameTask): Right end-effector task.
    """
    l_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
    r_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))


def compute_velocity_and_integrate(
    configuration: mink.Configuration,
    tasks: List[mink.FrameTask],
    rate: RateLimiter,
    solver: str,
    limits: List[mink.Limit],
    damping: float,
    pos_threshold: float,
    ori_threshold: float,
    max_iters: int
) -> None:
    """
    Compute the velocity and integrate it into the next configuration.

    Args:
        configuration (mink.Configuration): The current configuration.
        tasks (List[mink.FrameTask]): List of tasks.
        rate (RateLimiter): Rate limiter for controlling the loop frequency.
        solver (str): Solver to use for IK.
        limits (List[mink.Limit]): List of limits.
        damping (float): Damping factor for IK.
        pos_threshold (float): Position threshold for task achievement.
        ori_threshold (float): Orientation threshold for task achievement.
        max_iters (int): Maximum number of iterations for IK.
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

        l_err = tasks[0].compute_error(configuration)
        r_err = tasks[1].compute_error(configuration)

        if (np.linalg.norm(l_err[:3]) <= pos_threshold and np.linalg.norm(l_err[3:]) <= ori_threshold and
            np.linalg.norm(r_err[:3]) <= pos_threshold and np.linalg.norm(r_err[3:]) <= ori_threshold):
            break


def compensate_gravity(model: mujoco.MjModel, data: mujoco.MjData, subtree_ids: Sequence[int]) -> None:
    """
    Compensate for gravity by computing the necessary forces for each subtree.

    Args:
        model (mujoco.MjModel): The MuJoCo model.
        data (mujoco.MjData): The MuJoCo data.
        subtree_ids (Sequence[int]): List of subtree IDs for which to compute gravity compensation.
    """
    for subtree_id in subtree_ids:
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSubtreeCom(model, data, jacp, None, subtree_id)
        jacp = jacp[:, data.qvel_start:model.nv]
        gravity_compensation = jacp.T @ model.opt.gravity
        data.qfrc_applied[data.qfrc_applied_start:data.qfrc_applied_start + model.nu] += gravity_compensation


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Initialize joint names and velocity limits
    joint_names = []
    velocity_limits = {}
    for prefix in ["left", "right"]:
        for name in _JOINT_NAMES:
            joint_name = f"{prefix}/{name}"
            joint_names.append(joint_name)
            velocity_limits[joint_name] = _VELOCITY_LIMITS[name]

    dof_ids, actuator_ids = get_joint_and_actuator_ids(model, joint_names)

    configuration = mink.Configuration(model)
    tasks = initialize_tasks(model)
    collision_avoidance_limit = setup_collision_avoidance(model)
    limits = setup_limits(model, velocity_limits, collision_avoidance_limit)

    solver = "quadprog"
    max_iters = 2

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        tasks[2].set_target_from_configuration(configuration)

        initialize_mocap_targets(model, data)

        rate = RateLimiter(frequency=200.0)
        while viewer.is_running():
            update_task_targets(model, data, tasks[0], tasks[1])

            compute_velocity_and_integrate(
                configuration, tasks, rate, solver, limits, 1e-5, _POSITION_THRESHOLD, _ORIENTATION_THRESHOLD, max_iters
            )

            # Compensate for gravity
            compensate_gravity(model, data, [model.body("left/wrist_link").id, model.body("right/wrist_link").id])

            data.ctrl[actuator_ids] = configuration.q[dof_ids]
            mujoco.mj_step(model, data)

            viewer.sync()
            rate.sleep()


### Key Changes:
1. **Function Signature and Return Types**: Adjusted the `compensate_gravity` function to return `None` and directly modify `data.qfrc_applied` within the function.
2. **Variable Initialization**: Used a loop to build `joint_names` and `velocity_limits`, similar to the gold code, to enhance readability and maintain consistency.
3. **Task Initialization**: Used the assignment expression (walrus operator) to initialize tasks, making the code more concise and consistent with the gold code.
4. **Collision Avoidance Setup**: Ensured that the setup for collision avoidance is structured similarly to the gold code, including the definition of collision pairs and parameters.
5. **Loop Structure and Logic**: Reviewed and adjusted the main loop structure to closely follow the gold code, particularly how velocities are integrated and task achievement conditions are checked.
6. **Use of Constants**: Ensured that constants like position and orientation thresholds are defined and used consistently throughout the code.
7. **Comments and Documentation**: Reviewed and refined comments and docstrings for clarity and relevance, ensuring they accurately describe the purpose and functionality of each function.