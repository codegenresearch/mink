from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "unitree_g1" / "scene.xml"

if __name__ == "__main__":
    # Load the model from the XML file.
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    # Create a configuration object.
    configuration = mink.Configuration(model)

    # Define the feet and hands.
    feet = ["right_foot", "left_foot"]
    hands = ["right_palm", "left_palm"]

    # Initialize tasks.
    tasks = [
        mink.FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=10.0,
        ),
        mink.PostureTask(model, cost=1.0),
        mink.ComTask(cost=200.0),
    ]

    # Add tasks for each foot.
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=200.0,
            orientation_cost=10.0,
            lm_damping=1.0,
        )
        tasks.append(task)

    # Add tasks for each hand.
    for hand in hands:
        task = mink.FrameTask(
            frame_name=hand,
            frame_type="site",
            position_cost=200.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        tasks.append(task)

    # Get the mocap IDs for the center of mass and feet/hands.
    com_mid = model.body("com_target").mocapid[0]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]
    hands_mid = [model.body(f"{hand}_target").mocapid[0] for hand in hands]

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
        posture_task = tasks[1]
        pelvis_orientation_task = tasks[0]
        posture_task.set_target_from_configuration(configuration)
        pelvis_orientation_task.set_target_from_configuration(configuration)

        # Initialize mocap bodies at their respective sites.
        for hand, foot in zip(hands, feet):
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
            mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")
        data.mocap_pos[com_mid] = data.subtree_com[1]

        # Set up the rate limiter.
        rate = RateLimiter(frequency=200.0, warn=False)

        # Main loop.
        while viewer.is_running():
            # Update task targets.
            com_task = tasks[2]
            com_task.set_target(data.mocap_pos[com_mid])
            for i, (hand_task, foot_task) in enumerate(zip(tasks[3:5], tasks[5:7])):
                foot_task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))
                hand_task.set_target(mink.SE3.from_mocap_id(data, hands_mid[i]))

            # Solve inverse kinematics and integrate.
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-1)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()