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
_VELOCITY_LIMITS = {joint: np.pi for joint in _JOINT_NAMES}


def construct_model(xml_path):
    """
    Constructs the Mujoco model from the provided XML file.

    Args:
        xml_path (str): Path to the XML file defining the model.

    Returns:
        mujoco.MjModel: The constructed Mujoco model.
    """
    return mujoco.MjModel.from_xml_path(xml_path)


def get_dof_and_actuator_ids(model, joint_names):
    """
    Retrieves the degree of freedom (DOF) and actuator IDs for the specified joint names.

    Args:
        model (mujoco.MjModel): The Mujoco model.
        joint_names (list[str]): List of joint names.

    Returns:
        tuple: A tuple containing the DOF IDs and actuator IDs as numpy arrays.
    """
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])
    return dof_ids, actuator_ids


def initialize_mocap_targets(model, data):
    """
    Initializes the mocap targets at the end-effector sites.

    Args:
        model (mujoco.MjModel): The Mujoco model.
        data (mujoco.MjData): The Mujoco data.
    """
    mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
    mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")


def setup_collision_avoidance(model):
    """
    Sets up collision avoidance between specified geoms.

    Args:
        model (mujoco.MjModel): The Mujoco model.

    Returns:
        mink.CollisionAvoidanceLimit: The collision avoidance limit.
    """
    l_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("left/wrist_link").id)
    r_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("right/wrist_link").id)
    frame_geoms = mink.get_body_geom_ids(model, model.body("metal_frame").id)
    table_geom = ["table"]
    collision_pairs = [
        (l_wrist_geoms, r_wrist_geoms),
        (l_wrist_geoms, frame_geoms + table_geom),
        (r_wrist_geoms, frame_geoms + table_geom),
    ]
    return mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,
        minimum_distance_from_collisions=0.05,
        collision_detection_distance=0.1,
    )


def main():
    model = construct_model(_XML.as_posix())
    data = mujoco.MjData(model)

    # Generate joint names and velocity limits for both arms.
    joint_names = [f"{prefix}/{joint}" for prefix in ["left", "right"] for joint in _JOINT_NAMES]
    dof_ids, actuator_ids = get_dof_and_actuator_ids(model, joint_names)

    configuration = mink.Configuration(model)

    # Define tasks for the end-effectors and posture.
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
        posture_task := mink.PostureTask(
            posture_cost=0.1,  # Adjusted to match the gold code
            lm_damping=1.0,
        ),
    ]

    # Set the posture task target from the current configuration.
    posture_task.set_target(configuration.q)

    # Set up collision avoidance.
    collision_avoidance_limit = setup_collision_avoidance(model)

    # Define configuration limits.
    limits = [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(model, _VELOCITY_LIMITS),
        collision_avoidance_limit,
    ]

    # Mocap target IDs.
    left_target_id = model.body("left/target").mocapid[0]
    right_target_id = model.body("right/target").mocapid[0]

    # Solver and error thresholds.
    solver = "quadprog"
    position_threshold = 1e-3  # Adjusted to match the gold code
    orientation_threshold = 1e-3  # Adjusted to match the gold code
    max_iterations = 10  # Adjusted to match the gold code
    damping_value = 1e-2  # Adjusted to match the gold code

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Reset to the neutral pose keyframe.
        mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)

        # Initialize mocap targets.
        initialize_mocap_targets(model, data)

        rate = RateLimiter(frequency=200.0)
        while viewer.is_running():
            # Update task targets.
            l_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
            r_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

            # Compute velocity and integrate into the next configuration.
            for _ in range(max_iterations):
                velocity = mink.solve_ik(
                    configuration,
                    tasks,
                    rate.dt,
                    solver,
                    limits=limits,
                    damping=damping_value,
                )
                configuration.integrate_inplace(velocity, rate.dt)

                # Check if the tasks are achieved.
                left_error = l_ee_task.compute_error(configuration)
                left_position_achieved = np.linalg.norm(left_error[:3]) <= position_threshold
                left_orientation_achieved = np.linalg.norm(left_error[3:]) <= orientation_threshold

                right_error = r_ee_task.compute_error(configuration)
                right_position_achieved = np.linalg.norm(right_error[:3]) <= position_threshold
                right_orientation_achieved = np.linalg.norm(right_error[3:]) <= orientation_threshold

                if (left_position_achieved and left_orientation_achieved and
                    right_position_achieved and right_orientation_achieved):
                    break

            # Apply control signals.
            data.ctrl[actuator_ids] = configuration.q[dof_ids]
            mujoco.mj_step(model, data)

            # Update the viewer.
            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()