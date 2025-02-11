from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

# Path to the XML file defining the model.
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

# Velocity limits for each joint, based on the Interbotix VX300S specifications.
VELOCITY_LIMITS = {k: np.pi for k in JOINT_NAMES}


if __name__ == "__main__":
    # Load the model and data from the XML file.
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Collect degrees of freedom (DOF) and actuator IDs for both arms.
    dof_ids = []
    actuator_ids = []
    for prefix in ["left", "right"]:
        for name in JOINT_NAMES:
            full_name = f"{prefix}/{name}"
            dof_ids.append(model.joint(full_name).id)
            actuator_ids.append(model.actuator(full_name).id)
    dof_ids = np.array(dof_ids)
    actuator_ids = np.array(actuator_ids)

    # Initialize the configuration object.
    configuration = mink.Configuration(model)

    # Define tasks for the left and right end-effectors using the walrus operator.
    tasks = [
        left_ee_task := mink.FrameTask(
            frame_name="left/gripper",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        right_ee_task := mink.FrameTask(
            frame_name="right/gripper",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        posture_task := mink.PostureTask(
            configuration=configuration,
            position_cost=0.1,
            orientation_cost=0.1,
        ),
    ]

    # Define collision pairs to avoid collisions between the arms and the table.
    left_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("left/wrist_link").id)
    right_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("right/wrist_link").id)
    frame_geoms = mink.get_body_geom_ids(model, model.body("metal_frame").id)
    collision_pairs = [
        (left_wrist_geoms, right_wrist_geoms),
        (left_wrist_geoms, frame_geoms + ["table"]),
        (right_wrist_geoms, frame_geoms + ["table"]),
    ]
    collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,
        minimum_distance_from_collisions=0.05,
        collision_detection_distance=0.1,
    )

    # Define limits for the configuration, including velocity and collision avoidance.
    limits = [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(model, VELOCITY_LIMITS),
        collision_avoidance_limit,
    ]

    # Get mocap IDs for the left and right targets.
    left_target_id = model.body("left/target").mocapid[0]
    right_target_id = model.body("right/target").mocapid[0]
    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    # Launch the viewer.
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Reset the simulation to the neutral pose.
        mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)

        # Initialize mocap targets at the end-effector sites.
        mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
        mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")

        rate = RateLimiter(frequency=200.0)
        while viewer.is_running():
            # Update task targets to the current mocap positions.
            left_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
            right_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

            # Solve inverse kinematics and integrate the solution.
            for _ in range(max_iters):
                vel = mink.solve_ik(
                    configuration,
                    tasks,
                    rate.dt,
                    solver,
                    limits=limits,
                    damping=1e-2,  # Adjusted damping value to match gold code
                )
                configuration.integrate_inplace(vel, rate.dt)

                # Check if the tasks are achieved within the thresholds.
                left_err = left_ee_task.compute_error(configuration)
                left_pos_achieved = np.linalg.norm(left_err[:3]) <= pos_threshold
                left_ori_achieved = np.linalg.norm(left_err[3:]) <= ori_threshold
                right_err = right_ee_task.compute_error(configuration)
                right_pos_achieved = np.linalg.norm(right_err[:3]) <= pos_threshold
                right_ori_achieved = np.linalg.norm(right_err[3:]) <= ori_threshold
                if left_pos_achieved and left_ori_achieved and right_pos_achieved and right_ori_achieved:
                    break

            # Apply the computed velocities to the actuators.
            data.ctrl[actuator_ids] = configuration.q[dof_ids]
            mujoco.mj_step(model, data)

            # Update the viewer.
            viewer.sync()
            rate.sleep()