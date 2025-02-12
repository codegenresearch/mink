from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import mink

# Define the path to the XML file for the Boston Dynamics Spot model
_HERE = Path(__file__).parent
_XML = _HERE / "boston_dynamics_spot" / "scene.xml"

if __name__ == "__main__":
    # Load the Mujoco model and data from the XML file
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Initialize the inverse kinematics configuration
    configuration = mink.Configuration(model)

    # Define the feet names
    feet = ["FL", "FR", "HR", "HL"]

    # Define the base task for the body frame
    base_task = mink.FrameTask(
        frame_name="body",
        frame_type="body",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    # Define the posture task
    posture_task = mink.PostureTask(model, cost=1e-5)

    # Define tasks for each foot
    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="geom",
            position_cost=1.0,
            orientation_cost=0.0,
        )
        feet_tasks.append(task)

    # Define the end-effector task for the EE site
    eef_task = mink.FrameTask(
        frame_name="EE",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    # Combine all tasks into a single list
    tasks = [base_task, posture_task, *feet_tasks, eef_task]

    # Define mocap IDs for the base, feet, and end-effector
    base_mid = model.body("body_target").mocapid[0]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]
    eef_mid = model.body("EE_target").mocapid[0]

    # Define inverse kinematics settings
    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    # Launch the Mujoco viewer
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Reset data to the home keyframe and update configuration
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)

        # Set initial targets for tasks
        posture_task.set_target_from_configuration(configuration)
        for foot in feet:
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "geom")
        mink.move_mocap_to_frame(model, data, "body_target", "body", "body")
        mink.move_mocap_to_frame(model, data, "EE_target", "EE", "site")

        # Initialize the rate limiter for fixed FPS
        rate = RateLimiter(frequency=500.0)
        while viewer.is_running():
            # Update task targets based on mocap positions
            base_task.set_target(mink.SE3.from_mocap_id(data, base_mid))
            for i, task in enumerate(feet_tasks):
                task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))
            eef_task.set_target(mink.SE3.from_mocap_id(data, eef_mid))

            # Solve IK and integrate velocity
            pos_achieved = True
            ori_achieved = True
            for i in range(max_iters):
                vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3)
                configuration.integrate_inplace(vel, rate.dt)

                # Check if position and orientation goals are achieved
                for task in [eef_task, base_task, *feet_tasks]:
                    err = task.compute_error(configuration)
                    pos_achieved &= np.linalg.norm(err[:3]) <= pos_threshold
                    ori_achieved &= np.linalg.norm(err[3:]) <= ori_threshold
                if pos_achieved and ori_achieved:
                    print(f"Exiting after {i + 1} iterations.")
                    break

            # Update control signals and step the simulation
            data.ctrl = configuration.q[7:]
            mujoco.mj_step(model, data)

            # Sync the viewer with the simulation data
            viewer.sync()
            rate.sleep()