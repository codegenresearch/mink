from pathlib import Path
from typing import Optional, Sequence

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "aloha" / "scene.xml"

# Joint names for a single arm.
SINGLE_ARM_JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]

# Velocity limits for each joint, sourced from:
# https://github.com/Interbotix/interbotix_ros_manipulators/blob/main/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/vx300s.urdf.xacro
VELOCITY_LIMITS = {joint: np.pi for joint in SINGLE_ARM_JOINT_NAMES}


def compensate_gravity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    subtree_ids: Sequence[int],
    qfrc_applied: Optional[np.ndarray] = None,
) -> None:
    """Apply forces to counteract gravity for specified subtrees.\n\n    Args:\n        model: The Mujoco model.\n        data: The Mujoco data.\n        subtree_ids: A list of subtree IDs to apply gravity compensation to.\n        qfrc_applied: An optional array to store computed forces. If not provided,\n            the applied forces in `data` are used.\n    """
    qfrc_applied = data.qfrc_applied if qfrc_applied is None else qfrc_applied
    qfrc_applied[:] = 0.0  # Reset forces from previous calls.
    jac = np.empty((3, model.nv))
    for subtree_id in subtree_ids:
        total_mass = model.body_subtreemass[subtree_id]
        mujoco.mj_jacSubtreeCom(model, data, jac, subtree_id)
        qfrc_applied[:] -= model.opt.gravity * total_mass * jac


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(str(_XML_PATH))
    data = mujoco.MjData(model)

    # Subtree IDs for which gravity compensation is applied.
    LEFT_SUBTREE_ID = model.body("left/base_link").id
    RIGHT_SUBTREE_ID = model.body("right/base_link").id

    # Collect joint and actuator IDs for all controlled joints.
    joint_names = [f"{prefix}/{name}" for prefix in ["left", "right"] for name in SINGLE_ARM_JOINT_NAMES]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    configuration = mink.Configuration(model)

    # Define control tasks.
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

    # Configure collision avoidance between specific geoms.
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

    # Define control limits.
    limits = [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(model, VELOCITY_LIMITS),
        collision_avoidance_limit,
    ]

    # Mocap target IDs.
    LEFT_MOCAP_ID = model.body("left/target").mocapid[0]
    RIGHT_MOCAP_ID = model.body("right/target").mocapid[0]

    # IK solver parameters.
    SOLVER = "quadprog"
    POSITION_THRESHOLD = 5e-3
    ORIENTATION_THRESHOLD = 5e-3
    MAX_ITERATIONS = 5

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

        rate_limiter = RateLimiter(frequency=200.0)
        while viewer.is_running():
            # Update task targets to follow mocap positions.
            tasks[0].set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
            tasks[1].set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

            # Solve IK to compute velocity commands.
            for _ in range(MAX_ITERATIONS):
                velocity = mink.solve_ik(
                    configuration,
                    tasks,
                    rate_limiter.dt,
                    SOLVER,
                    limits=limits,
                    damping=1e-5,
                )
                configuration.integrate_inplace(velocity, rate_limiter.dt)

                # Check if both end-effectors have reached their targets.
                left_error = tasks[0].compute_error(configuration)
                left_position_reached = np.linalg.norm(left_error[:3]) <= POSITION_THRESHOLD
                left_orientation_reached = np.linalg.norm(left_error[3:]) <= ORIENTATION_THRESHOLD

                right_error = tasks[1].compute_error(configuration)
                right_position_reached = np.linalg.norm(right_error[:3]) <= POSITION_THRESHOLD
                right_orientation_reached = np.linalg.norm(right_error[3:]) <= ORIENTATION_THRESHOLD

                if left_position_reached and left_orientation_reached and right_position_reached and right_orientation_reached:
                    break

            # Apply computed joint torques and compensate for gravity.
            data.ctrl[actuator_ids] = configuration.q[dof_ids]
            compensate_gravity(model, data, [LEFT_SUBTREE_ID, RIGHT_SUBTREE_ID])

            # Step the simulation.
            mujoco.mj_step(model, data)

            # Synchronize the viewer and sleep to maintain the desired update rate.
            viewer.sync()
            rate_limiter.sleep()