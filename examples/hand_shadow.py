from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "shadow_hand" / "scene_left.xml"

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    configuration = mink.Configuration(model)

    # Define fingers
    fingers = ["thumb", "first", "middle", "ring", "little"]

    # Initialize posture task
    posture_task = mink.PostureTask(model, cost=1e-2)

    # Initialize finger tasks
    finger_tasks = []
    for finger in fingers:
        task = mink.FrameTask(
            frame_name=finger,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        finger_tasks.append(task)

    # Combine tasks
    tasks = [posture_task] + finger_tasks

    model = configuration.model
    data = configuration.data
    solver = "quadprog"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize mocap bodies at their respective sites.
        for finger in fingers:
            mink.move_mocap_to_frame(model, data, f"{finger}_target", finger, "site")

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("grasp hard")
        posture_task.set_target_from_configuration(configuration)

        rate = RateLimiter(frequency=500.0, warn=False)  # Adjusted frequency for better performance
        dt = rate.dt
        t = 0  # Track elapsed time

        while viewer.is_running():
            # Update task targets.
            for finger, task in zip(fingers, finger_tasks):
                task.set_target(mink.SE3.from_mocap_name(model, data, f"{finger}_target"))

            # Solve inverse kinematics and integrate velocity.
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-5)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
            t += dt  # Increment elapsed time