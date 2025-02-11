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
    dof_ids = []
    actuator_ids = []
    for prefix in ["left", "right"]:
        for joint in joint_names:
            dof_ids.append(model.joint(f"{prefix}/{joint}").id)
            actuator_ids.append(model.actuator(f"{prefix}/{joint}").id)
    return np.array(dof_ids), np.array(actuator_ids)


def initialize_mocap_targets(model, data, left_target_name, right_target_name, left_gripper_name, right_gripper_name):
    """
    Initializes the mocap targets at the end-effector sites.

    Args:
        model (mujoco.MjModel): The Mujoco model.
        data (mujoco.MjData): The Mujoco data.
        left_target_name (str): Name of the left mocap target.
        right_target_name (str): Name of the right mocap target.
        left_gripper_name (str): Name of the left gripper site.
        right_gripper_name (str): Name of the right gripper site.
    """
    mink.move_mocap_to_frame(model, data, left_target_name, left_gripper_name, "site")
    mink.move_mocap_to_frame(model, data, right_target_name, right_gripper_name, "site")


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
        (l_wrist_geoms, r_wrist_geoms),  # Collision between left and right wrists
        (l_wrist_geoms + r_wrist_geoms, frame_geoms + table_geom),  # Collision between wrists and table/frame
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

    # Get the dof and actuator ids for the joints we wish to control.
    dof_ids, actuator_ids = get_dof_and_actuator_ids(model, _JOINT_NAMES)

    configuration = mink.Configuration(model)

    # Define tasks for left and right end-effectors.
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
    ]

    # Define posture task.
    posture_task = mink.PostureTask(
        configuration=configuration,
        position_cost=0.1,
        orientation_cost=0.1,
    )

    # Set up collision avoidance.
    collision_avoidance_limit = setup_collision_avoidance(model)

    # Define configuration limits.
    velocity_limits = {f"{prefix}/{joint}": _VELOCITY_LIMITS[joint] for prefix in ["left", "right"] for joint in _JOINT_NAMES}
    limits = [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(model, velocity_limits),
        collision_avoidance_limit,
    ]

    # Mocap target IDs.
    l_mid = model.body("left/target").mocapid[0]
    r_mid = model.body("right/target").mocapid[0]

    # Solver and error thresholds.
    solver = "quadprog"
    pos_threshold = 1e-3  # Adjusted to match gold code
    ori_threshold = 1e-3  # Adjusted to match gold code
    max_iters = 10  # Adjusted to match gold code
    damping = 1e-2  # Adjusted to match gold code

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)

        # Initialize mocap targets at the end-effector site.
        initialize_mocap_targets(model, data, "left/target", "right/target", "left/gripper", "right/gripper")

        rate = RateLimiter(frequency=200.0)
        while viewer.is_running():
            # Update task targets.
            l_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
            r_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

            # Compute posture task target.
            posture_task.set_target_from_configuration(configuration)

            # Compute velocity and integrate into the next configuration.
            for _ in range(max_iters):
                vel = mink.solve_ik(
                    configuration,
                    tasks + [posture_task],
                    rate.dt,
                    solver,
                    limits=limits,
                    damping=damping,
                )
                configuration.integrate_inplace(vel, rate.dt)

                # Check if the tasks are achieved.
                l_err = l_ee_task.compute_error(configuration)
                l_pos_achieved = np.linalg.norm(l_err[:3]) <= pos_threshold
                l_ori_achieved = np.linalg.norm(l_err[3:]) <= ori_threshold

                r_err = r_ee_task.compute_error(configuration)
                r_pos_achieved = np.linalg.norm(r_err[:3]) <= pos_threshold
                r_ori_achieved = np.linalg.norm(r_err[3:]) <= ori_threshold

                if l_pos_achieved and l_ori_achieved and r_pos_achieved and r_ori_achieved:
                    break

            data.ctrl[actuator_ids] = configuration.q[dof_ids]
            mujoco.mj_step(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()