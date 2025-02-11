"""
KUKA IIWA 14 Robot Manipulation Script

This script sets up and runs an inverse kinematics (IK) controller for the KUKA IIWA 14 robot.
It uses the MuJoCo physics engine to simulate the robot and the Mink library for IK calculations.
The script initializes the robot in a home position, sets up tasks for end-effector positioning and posture,
and continuously updates the robot's configuration to achieve the desired tasks.
"""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

# Define the path to the XML file containing the robot model
_HERE = Path(__file__).parent
_XML_PATH = _HERE / "kuka_iiwa_14" / "scene.xml"

def main():
    # Load the MuJoCo model and data
    model = mujoco.MjModel.from_xml_path(_XML_PATH.as_posix())
    data = mujoco.MjData(model)

    # Setup IK configuration
    configuration = mink.Configuration(model)

    # Define tasks for the IK solver
    tasks = [
        mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        mink.PostureTask(model=model, cost=1e-2),
    ]

    # Define IK settings
    solver = "quadprog"
    position_threshold = 1e-4
    orientation_threshold = 1e-4
    max_iterations = 20

    # Launch the MuJoCo viewer
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        # Set up the default free camera view
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Reset the robot to the home position
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        tasks[1].set_target_from_configuration(configuration)  # Set posture target
        mujoco.mj_forward(model, data)

        # Initialize the mocap target at the end-effector site
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

        # Set up the rate limiter for controlling the loop frequency
        rate_limiter = RateLimiter(frequency=500.0)

        # Main simulation loop
        while viewer.is_running():
            # Update the task target based on the mocap position
            target_pose = mink.SE3.from_mocap_name(model, data, "target")
            tasks[0].set_target(target_pose)

            # Solve IK and update the robot configuration
            for iteration in range(max_iterations):
                velocity = mink.solve_ik(configuration, tasks, rate_limiter.dt, solver, 1e-3)
                configuration.integrate_inplace(velocity, rate_limiter.dt)
                error = tasks[0].compute_error(configuration)
                position_achieved = np.linalg.norm(error[:3]) <= position_threshold
                orientation_achieved = np.linalg.norm(error[3:]) <= orientation_threshold
                if position_achieved and orientation_achieved:
                    print(f"Task achieved after {iteration + 1} iterations.")
                    break

            # Apply the updated configuration to the robot's actuators
            data.ctrl = configuration.q
            mujoco.mj_step(model, data)

            # Synchronize the viewer and sleep to maintain the desired loop rate
            viewer.sync()
            rate_limiter.sleep()

if __name__ == "__main__":
    main()