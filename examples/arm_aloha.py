from pathlib import Path

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

# Velocity limits for each joint, sourced from:
# https://github.com/Interbotix/interbotix_ros_manipulators/blob/main/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/vx300s.urdf.xacro
_JOINT_VELOCITY_LIMITS = {k: np.pi for k in _JOINT_NAMES}


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Collect joint and actuator IDs for both arms.
    joint_ids = {}
    actuator_ids = {}
    for prefix in ["left", "right"]:
        for joint_name in _JOINT_NAMES:
            full_joint_name = f"{prefix}/{joint_name}"
            joint_ids[full_joint_name] = model.joint(full_joint_name).id
            actuator_ids[full_joint_name] = model.actuator(full_joint_name).id

    configuration = mink.Configuration(model)

    # Define tasks for both left and right end-effectors.
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
    ]

    # Define collision pairs for collision avoidance.
    left_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("left/wrist_link").id)
    right_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("right/wrist_link").id)
    frame_geoms = mink.get_body_geom_ids(model, model.body("metal_frame").id)
    collision_pairs = [
        (left_wrist_geoms, right_wrist_geoms),
        (left_wrist_geoms + right_wrist_geoms, frame_geoms + ["table"]),
    ]

    # Set up limits for configuration, velocity, and collision avoidance.
    limits = [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(model, _JOINT_VELOCITY_LIMITS),
        mink.CollisionAvoidanceLimit(
            model=model,
            geom_pairs=collision_pairs,
            minimum_distance_from_collisions=0.05,
            collision_detection_distance=0.1,
        ),
    ]

    # Mocap IDs for left and right targets.
    left_target_mid = model.body("left/target").mocapid[0]
    right_target_mid = model.body("right/target").mocapid[0]

    # IK settings.
    solver = "quadprog"
    position_threshold = 1e-4
    orientation_threshold = 1e-4
    max_iterations = 20

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the neutral keyframe.
        mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)

        # Position mocap targets at the end-effector sites.
        mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
        mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")

        rate = RateLimiter(frequency=200.0)
        while viewer.is_running():
            # Update task targets.
            tasks[0].set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
            tasks[1].set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

            # Compute velocity and integrate into the next configuration.
            for iteration in range(max_iterations):
                velocity = mink.solve_ik(
                    configuration,
                    tasks,
                    rate.dt,
                    solver,
                    limits=limits,
                    damping=1e-3,
                )
                configuration.integrate_inplace(velocity, rate.dt)

                left_error = tasks[0].compute_error(configuration)
                left_position_achieved = np.linalg.norm(left_error[:3]) <= position_threshold
                left_orientation_achieved = np.linalg.norm(left_error[3:]) <= orientation_threshold

                right_error = tasks[1].compute_error(configuration)
                right_position_achieved = np.linalg.norm(right_error[:3]) <= position_threshold
                right_orientation_achieved = np.linalg.norm(right_error[3:]) <= orientation_threshold

                if (
                    left_position_achieved
                    and left_orientation_achieved
                    and right_position_achieved
                    and right_orientation_achieved
                ):
                    print(f"Exiting after {iteration} iterations.")
                    break

            # Apply control to the actuators.
            data.ctrl[list(actuator_ids.values())] = configuration.q[list(joint_ids.values())]
            mujoco.mj_step(model, data)

            # Sync viewer and sleep to maintain frame rate.
            viewer.sync()
            rate.sleep()