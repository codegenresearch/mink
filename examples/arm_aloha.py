from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

# This script sets up a simulation environment for a dual-arm robot using the MuJoCo physics engine.
# It configures inverse kinematics (IK) for both arms, sets up collision avoidance, and visualizes the simulation.

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

# Velocity limits for each joint, sourced from the Interbotix vx300s URDF.
_VELOCITY_LIMITS = {k: np.pi for k in _JOINT_NAMES}


if __name__ == "__main__":
    # Load the MuJoCo model and data.
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Collect joint and actuator IDs for both arms.
    joint_names = [f"{prefix}/{name}" for prefix in ["left", "right"] for name in _JOINT_NAMES]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    # Initialize the configuration for IK.
    configuration = mink.Configuration(model)

    # Define IK tasks for both arms' end effectors.
    left_ee_task = mink.FrameTask(
        frame_name="left/gripper",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    right_ee_task = mink.FrameTask(
        frame_name="right/gripper",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    tasks = [left_ee_task, right_ee_task]

    # Enable collision avoidance between the wrists and the table, and between the two wrists.
    left_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("left/wrist_link").id)
    right_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("right/wrist_link").id)
    frame_geoms = mink.get_body_geom_ids(model, model.body("metal_frame").id)
    collision_pairs = [
        (left_wrist_geoms, right_wrist_geoms),
        (left_wrist_geoms + right_wrist_geoms, frame_geoms + ["table"]),
    ]
    collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,
        minimum_distance_from_collisions=0.05,
        collision_detection_distance=0.1,
    )

    # Define the limits for IK.
    limits = [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(model, _VELOCITY_LIMITS),
        collision_avoidance_limit,
    ]

    # Get mocap IDs for both arms' targets.
    left_mid = model.body("left/target").mocapid[0]
    right_mid = model.body("right/target").mocapid[0]
    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    # Launch the MuJoCo viewer.
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)

        # Initialize mocap targets at the end-effector sites.
        mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
        mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")

        rate = RateLimiter(frequency=200.0)
        while viewer.is_running():
            # Update task targets.
            left_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
            right_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

            # Compute velocity and integrate into the next configuration.
            for i in range(max_iters):
                vel = mink.solve_ik(
                    configuration,
                    tasks,
                    rate.dt,
                    solver,
                    limits=limits,
                    damping=1e-3,
                )
                configuration.integrate_inplace(vel, rate.dt)

                # Check if both end effectors have reached their targets.
                left_err = left_ee_task.compute_error(configuration)
                left_pos_achieved = np.linalg.norm(left_err[:3]) <= pos_threshold
                left_ori_achieved = np.linalg.norm(left_err[3:]) <= ori_threshold
                right_err = right_ee_task.compute_error(configuration)
                right_pos_achieved = np.linalg.norm(right_err[:3]) <= pos_threshold
                right_ori_achieved = np.linalg.norm(right_err[3:]) <= ori_threshold
                if left_pos_achieved and left_ori_achieved and right_pos_achieved and right_ori_achieved:
                    print(f"Exiting after {i} iterations.")
                    break

            # Apply the computed velocities to the actuators.
            data.ctrl[actuator_ids] = configuration.q[dof_ids]
            mujoco.mj_step(model, data)

            # Update the viewer.
            viewer.sync()
            rate.sleep()