from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "unitree_go1" / "scene.xml"

if __name__ == "__main__":
    # Load the model from the XML file.
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    # Create a configuration object.
    configuration = mink.Configuration(model)

    # Define the feet names.
    feet = ["FL", "FR", "RR", "RL"]

    # Create tasks for the base and posture.
    base_task = mink.FrameTask(
        frame_name="trunk",
        frame_type="body",
        position_cost=0.0,
        orientation_cost=10.0,
    )

    posture_task = mink.PostureTask(model, cost=1.0)

    # Create tasks for each foot.
    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=200.0,
            orientation_cost=10.0,
            lm_damping=1.0,
        )
        feet_tasks.append(task)

    # Combine all tasks.
    tasks = [base_task, posture_task, *feet_tasks]

    # Get the mocap IDs for the base and feet.
    base_mid = model.body("trunk_target").mocapid[0]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]

    # Get the model and data from the configuration.
    model = configuration.model
    data = configuration.data
    solver = "quadprog"

    # Launch the viewer.
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("stand")
        posture_task.set_target_from_configuration(configuration)
        base_task.set_target_from_configuration(configuration)

        # Initialize mocap bodies at their respective sites.
        for foot in feet:
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
        mink.move_mocap_to_frame(model, data, "trunk_target", "trunk", "body")

        # Set up the rate limiter.
        rate = RateLimiter(frequency=200.0, warn=False)

        # Main loop.
        while viewer.is_running():
            # Update task targets.
            base_task.set_target(mink.SE3.from_mocap_id(data, base_mid))
            for i, task in enumerate(feet_tasks):
                task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))

            # Compute velocity, integrate, and set control signal.
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-1)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()