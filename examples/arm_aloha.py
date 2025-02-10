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
    """Get the DOF and actuator IDs for the specified joint names."""
    dof_ids = []
    actuator_ids = []
    for name in joint_names:
        dof_ids.append(model.joint(name).id)
        actuator_ids.append(model.actuator(name).id)
    return np.array(dof_ids), np.array(actuator_ids)


def setup_collision_avoidance(model: mujoco.MjModel) -> mink.CollisionAvoidanceLimit:
    """Setup collision avoidance for specified geom pairs."""
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
    """Setup the limits for the configuration, velocity, and collision avoidance."""
    return [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(model, velocity_limits),
        collision_avoidance_limit,
    ]


def initialize_mocap_targets(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Initialize mocap targets at the end-effector sites."""
    mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
    mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")


def update_task_targets(model: mujoco.MjModel, data: mujoco.MjData, l_ee_task: mink.FrameTask, r_ee_task: mink.FrameTask) -> None:
    """Update the task targets for the left and right end-effectors."""
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
    max_iters: int,
) -> bool:
    """Compute the velocity and integrate it into the next configuration."""
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
            return True
    return False


def compensate_gravity(model: mujoco.MjModel, data: mujoco.MjData, subtree_ids: Sequence[int], qfrc_applied: Optional[np.ndarray] = None) -> None:
    """Compensate for gravity by computing the necessary forces for specified subtrees."""
    if qfrc_applied is None:
        qfrc_applied = np.zeros(model.nv)

    for subtree_id in subtree_ids:
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

    joint_names = []
    velocity_limits = {}
    for prefix in ["left", "right"]:
        for n in _JOINT_NAMES:
            name = f"{prefix}/{n}"
            joint_names.append(name)
            velocity_limits[name] = _VELOCITY_LIMITS[n]

    dof_ids, actuator_ids = get_joint_and_actuator_ids(model, joint_names)

    configuration = mink.Configuration(model)
    tasks = [
        l_ee_task := mink.FrameTask(
            frame_name="left/gripper",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        r_ee_task := mink.FrameTask(
            frame_name="right/gripper",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        posture_task := mink.PostureTask(model, cost=1e-4),
    ]

    collision_avoidance_limit = setup_collision_avoidance(model)
    limits = setup_limits(model, velocity_limits, collision_avoidance_limit)

    solver = "quadprog"
    max_iters = 2

    subtree_ids = [model.body("left/wrist_link").id, model.body("right/wrist_link").id]

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        posture_task.set_target_from_configuration(configuration)

        initialize_mocap_targets(model, data)

        rate = RateLimiter(frequency=200.0)
        while viewer.is_running():
            update_task_targets(model, data, l_ee_task, r_ee_task)

            if compute_velocity_and_integrate(
                configuration, tasks, rate, solver, limits, 1e-5, _POSITION_THRESHOLD, _ORIENTATION_THRESHOLD, max_iters
            ):
                break

            # Compensate for gravity
            compensate_gravity(model, data, subtree_ids)

            data.ctrl[actuator_ids] = configuration.q[dof_ids] + data.qfrc_applied[actuator_ids]

            mujoco.mj_step(model, data)

            viewer.sync()
            rate.sleep()