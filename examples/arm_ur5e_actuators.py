from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "universal_robots_ur5e" / "scene.xml"

# This script sets up and runs inverse kinematics for the UR5e robotic arm using Mujoco and Mink.
# It includes tasks for end-effector position and orientation control, collision avoidance, and joint velocity limits.

if __name__ == "__main__":
    # Load the Mujoco model from the XML file.
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # =================== #
    # Setup Inverse Kinematics
    # =================== #

    # Initialize the configuration using the loaded model.
    configuration = mink.Configuration(model)

    # Define the task for end-effector control.
    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
    ]

    # Define collision avoidance between the wrist_3_link and specified obstacles (floor and wall).
    wrist_3_geoms = mink.get_body_geom_ids(model, model.body("wrist_3_link").id)
    collision_pairs = [
        (wrist_3_geoms, ["floor", "wall"]),
    ]

    # Initialize limits for configuration, collision avoidance, and velocity.
    limits = [
        mink.ConfigurationLimit(model=configuration.model),
        mink.CollisionAvoidanceLimit(
            model=configuration.model,
            geom_pairs=collision_pairs,
            minimum_distance_from_collisions=0.05,
            collision_detection_distance=0.1,
        ),
    ]

    # Define maximum joint velocities for each joint.
    max_velocities = {
        "shoulder_pan": np.pi,
        "shoulder_lift": np.pi,
        "elbow": np.pi,
        "wrist_1": np.pi,
        "wrist_2": np.pi,
        "wrist_3": np.pi,
    }
    velocity_limit = mink.VelocityLimit(model, max_velocities)
    limits.append(velocity_limit)

    # =================== #

    # ID for the target body in the model.
    mid = model.body("target").mocapid[0]

    # IK solver settings.
    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    # Launch the Mujoco viewer in passive mode.
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Reset the simulation to the home keyframe and update the configuration.
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

        # Set up the rate limiter for controlling the loop frequency.
        rate = RateLimiter(frequency=500.0)
        while viewer.is_running():
            # Update the task target to the current position of the mocap target.
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            # Compute the inverse kinematics and integrate the result into the configuration.
            for i in range(max_iters):
                vel = mink.solve_ik(
                    configuration, tasks, rate.dt, solver, damping=1e-3, limits=limits
                )
                configuration.integrate_inplace(vel, rate.dt)
                err = end_effector_task.compute_error(configuration)
                pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
                ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
                if pos_achieved and ori_achieved:
                    print(f"Task achieved after {i} iterations.")
                    break

            # Update the control input to the model.
            data.ctrl = configuration.q
            mujoco.mj_step(model, data)

            # Sync the viewer to visualize the simulation at a fixed frame rate.
            viewer.sync()
            rate.sleep()