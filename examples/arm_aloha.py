from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "aloha" / "scene.xml"

# Joint names for a single arm.
JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]

# Velocity limits for each joint, sourced from:
# https://github.com/Interbotix/interbotix_ros_manipulators/blob/main/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/vx300s.urdf.xacro
VELOCITY_LIMITS = {joint: np.pi for joint in JOINT_NAMES}

def get_dof_and_actuator_ids(model, joint_names):
    """Retrieve DOF and actuator IDs for the specified joint names."""
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])
    return dof_ids, actuator_ids

if __name__ == "__main__":
    # Load the model and data.
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Generate joint names for both left and right arms.
    joint_names = [f"{prefix}/{joint}" for prefix in ["left", "right"] for joint in JOINT_NAMES]
    dof_ids, actuator_ids = get_dof_and_actuator_ids(model, joint_names)

    # Configure the control system.
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

    # Set up collision avoidance between the arms and the table.
    l_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("left/wrist_link").id)
    r_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("right/wrist_link").id)
    frame_geoms = mink.get_body_geom_ids(model, model.body("metal_frame").id)
    collision_pairs = [
        (l_wrist_geoms, r_wrist_geoms),
        (l_wrist_geoms + r_wrist_geoms, frame_geoms + ["table"]),
    ]
    collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,
        minimum_distance_from_collisions=0.05,
        collision_detection_distance=0.1,
    )

    # Define limits for the configuration.
    limits = [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(model, VELOCITY_LIMITS),
        collision_avoidance_limit,
    ]

    # Get mocap IDs for the left and right targets.
    left_target_id = model.body("left/target").mocapid[0]
    right_target_id = model.body("right/target").mocapid[0]

    # Set up the solver and error thresholds.
    solver = "quadprog"
    position_threshold = 1e-4
    orientation_threshold = 1e-4
    max_iterations = 20

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        # Set up the default camera view.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the neutral pose keyframe.
        mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)

        # Initialize mocap targets at the end-effector sites.
        mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
        mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")

        # Set up the rate limiter for the control loop.
        rate = RateLimiter(frequency=200.0)

        while viewer.is_running():
            # Update task targets to match mocap positions.
            tasks[0].set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
            tasks[1].set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

            # Solve the IK problem and integrate the solution.
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

                # Check if the end-effectors have reached the desired position and orientation.
                left_error = tasks[0].compute_error(configuration)
                left_position_achieved = np.linalg.norm(left_error[:3]) <= position_threshold
                left_orientation_achieved = np.linalg.norm(left_error[3:]) <= orientation_threshold

                right_error = tasks[1].compute_error(configuration)
                right_position_achieved = np.linalg.norm(right_error[:3]) <= position_threshold
                right_orientation_achieved = np.linalg.norm(right_error[3:]) <= orientation_threshold

                if (left_position_achieved and left_orientation_achieved and
                    right_position_achieved and right_orientation_achieved):
                    break

            # Apply the computed joint velocities to the actuators.
            data.ctrl[actuator_ids] = configuration.q[dof_ids]

            # Step the simulation.
            mujoco.mj_step(model, data)

            # Update the viewer at the desired frame rate.
            viewer.sync()
            rate.sleep()