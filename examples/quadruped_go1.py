from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

HERE = Path(__file__).parent
XML_PATH = HERE / "unitree_go1" / "scene.xml"

def initialize_tasks(model, feet):
    base_task = mink.FrameTask(
        frame_name="trunk",
        frame_type="body",
        position_cost=1.0,
        orientation_cost=1.0,
    )
    
    posture_task = mink.PostureTask(model, cost=1e-5)
    
    feet_tasks = [
        mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
        ) for foot in feet
    ]
    
    return [base_task, posture_task] + feet_tasks

def initialize_mocap_positions(model, data, feet):
    for foot in feet:
        mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
    mink.move_mocap_to_frame(model, data, "trunk_target", "trunk", "body")

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(XML_PATH.as_posix())
    configuration = mink.Configuration(model)

    FEET = ["FL", "FR", "RR", "RL"]
    tasks = initialize_tasks(model, FEET)

    BASE_MOCAP_ID = model.body("trunk_target").mocapid[0]
    FEET_MOCAP_IDS = [model.body(f"{foot}_target").mocapid[0] for foot in FEET]

    model, data = configuration.model, configuration.data
    SOLVER = "quadprog"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("home")
        tasks[1].set_target_from_configuration(configuration)

        # Initialize mocap bodies at their respective sites.
        initialize_mocap_positions(model, data, FEET)

        rate_limiter = RateLimiter(frequency=500.0)
        while viewer.is_running():
            # Update task targets.
            tasks[0].set_target(mink.SE3.from_mocap_id(data, BASE_MOCAP_ID))
            for i, task in enumerate(tasks[2:], start=2):
                task.set_target(mink.SE3.from_mocap_id(data, FEET_MOCAP_IDS[i-2]))

            # Compute velocity, integrate and set control signal.
            velocity = mink.solve_ik(configuration, tasks, rate_limiter.dt, SOLVER, 1e-5)
            configuration.integrate_inplace(velocity, rate_limiter.dt)
            mujoco.mj_camlight(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate_limiter.sleep()