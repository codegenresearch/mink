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
_XML = _HERE / "kuka_iiwa_14" / "scene.xml"

if __name__ == "__main__":
    # Load the MuJoCo model and data
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    ## =================== ##
    ## Setup IK.
    ## =================== ##

    configuration = mink.Configuration(model)

    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        posture_task := mink.PostureTask(model=model, cost=1e-2),
    ]

    ## =================== ##

    # IK settings.
    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    # Launch the MuJoCo viewer
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        # Set up the default free camera view
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Reset the robot to the home position
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration()  # Set posture target
        mujoco.mj_forward(model, data)

        # Initialize the mocap target at the end-effector site
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

        # Set up the rate limiter for controlling the loop frequency
        rate = RateLimiter(frequency=500.0)

        # Main simulation loop
        while viewer.is_running():
            # Update task target
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            # Compute the IK solution and update the robot configuration
            for i in range(max_iters):
                vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3)
                configuration.integrate_inplace(vel, rate.dt)
                err = end_effector_task.compute_error(configuration)
                pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
                ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
                if pos_achieved and ori_achieved:
                    break

            # Apply the computed configuration to the robot's actuators
            data.ctrl = configuration.q
            mujoco.mj_step(model, data)

            # Synchronize the viewer and sleep to maintain the desired loop rate
            viewer.sync()
            rate.sleep()