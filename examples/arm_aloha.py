from pathlib import Path
from typing import Optional, Sequence, Dict, List

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "aloha" / "scene.xml"

# Joint names for a single arm.
_JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]

# Velocity limits for each joint.
_VELOCITY_LIMITS: Dict[str, float] = {joint: np.pi for joint in _JOINT_NAMES}


def compensate_gravity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    subtree_ids: Sequence[int],
    qfrc_applied: Optional[np.ndarray] = None,
) -> None:
    """Compute forces to counteract gravity for the specified subtrees.

    Args:
        model: The Mujoco model.
        data: The Mujoco data.
        subtree_ids: A list of subtree IDs. Gravity compensation forces will be applied to all bodies in these subtrees.
        qfrc_applied: An optional array to store the computed forces. If not provided, the applied forces in `data` are used.
    """
    qfrc_applied = data.qfrc_applied if qfrc_applied is None else qfrc_applied
    qfrc_applied[:] = 0.0  # Reset forces from previous calls.
    jac = np.empty((3, model.nv))
    for subtree_id in subtree_ids:
        total_mass = model.body_subtreemass[subtree_id]
        mujoco.mj_jacSubtreeCom(model, data, jac, subtree_id)
        qfrc_applied[:] -= model.opt.gravity @ (total_mass * jac)


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(str(_XML))
    data = mujoco.MjData(model)

    # Subtree IDs for gravity compensation.
    left_subtree_id = model.body("left/base_link").id
    right_subtree_id = model.body("right/base_link").id

    # Collect joint and actuator IDs for both arms.
    joint_names: List[str] = []
    velocity_limits: Dict[str, float] = {}
    for prefix in ["left", "right"]:
        for joint in _JOINT_NAMES:
            name = f"{prefix}/{joint}"
            joint_names.append(name)
            velocity_limits[name] = _VELOCITY_LIMITS[joint]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    configuration = mink.Configuration(model)

    # Define tasks for both arms and posture.
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

    # Set up collision avoidance for specified geometries.
    left_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("left/wrist_link").id)
    right_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("right/wrist_link").id)
    left_upper_arm_geoms = mink.get_subtree_geom_ids(model, model.body("left/upper_arm_link").id)
    right_upper_arm_geoms = mink.get_subtree_geom_ids(model, model.body("right/upper_arm_link").id)
    frame_geoms = mink.get_body_geom_ids(model, model.body("metal_frame").id)
    collision_pairs = [
        (left_wrist_geoms, right_wrist_geoms),
        (left_upper_arm_geoms + right_upper_arm_geoms, frame_geoms + ["table"]),
    ]
    collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,
        minimum_distance_from_collisions=0.05,
        collision_detection_distance=0.1,
    )

    # Define limits for configuration, velocity, and collision avoidance.
    config_limit = mink.ConfigurationLimit(model=model)
    velocity_limit = mink.VelocityLimit(model, velocity_limits)
    limits = [config_limit, velocity_limit, collision_avoidance_limit]

    # Mocap IDs for left and right targets.
    left_target_id = model.body("left/target").mocapid[0]
    right_target_id = model.body("right/target").mocapid[0]

    # IK solver settings.
    solver = "quadprog"
    pos_threshold = 5e-3
    ori_threshold = 5e-3
    max_iters = 5

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the neutral pose keyframe.
        mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        tasks[-1].set_target_from_configuration(configuration)

        # Position mocap targets at the end-effector sites.
        mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
        mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Update task targets.
            tasks[0].set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
            tasks[1].set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

            # Solve IK and integrate the solution.
            for _ in range(max_iters):
                vel = mink.solve_ik(
                    configuration,
                    tasks,
                    rate.dt,
                    solver,
                    limits=limits,
                    damping=1e-5,
                )
                configuration.integrate_inplace(vel, rate.dt)

                # Check if both arms have reached their targets.
                l_err = tasks[0].compute_error(configuration)
                l_pos_achieved = np.linalg.norm(l_err[:3]) <= pos_threshold
                l_ori_achieved = np.linalg.norm(l_err[3:]) <= ori_threshold
                r_err = tasks[1].compute_error(configuration)
                r_pos_achieved = np.linalg.norm(r_err[:3]) <= pos_threshold
                r_ori_achieved = np.linalg.norm(r_err[3:]) <= ori_threshold
                if l_pos_achieved and l_ori_achieved and r_pos_achieved and r_ori_achieved:
                    break

            # Apply computed joint positions and gravity compensation.
            data.ctrl[actuator_ids] = configuration.q[dof_ids]
            compensate_gravity(model, data, [left_subtree_id, right_subtree_id])
            mujoco.mj_step(model, data)

            # Update the viewer and sleep to maintain the desired rate.
            viewer.sync()
            rate.sleep()