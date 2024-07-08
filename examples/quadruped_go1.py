import mujoco
import mujoco.viewer
import mink
from pathlib import Path
from loop_rate_limiters import RateLimiter
from mink.utils import set_mocap_pose_from_site, set_mocap_pose_from_body

_HERE = Path(__file__).parent
_XML = _HERE / "unitree_go1" / "scene.xml"


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    configuration = mink.Configuration(model)

    feet = ["FL", "FR", "RR", "RL"]

    base_task = mink.FrameTask(
        frame_name="trunk",
        frame_type="body",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    posture_task = mink.PostureTask(cost=1e-5)

    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
        )
        feet_tasks.append(task)

    tasks = [base_task, posture_task, *feet_tasks]

    limits = [
        mink.ConfigurationLimit(model=model),
    ]

    base_mid = model.body("trunk_target").mocapid[0]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]

    model = configuration.model
    data = configuration.data

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("home")
        posture_task.set_target_from_configuration(configuration)

        # Initialize mocap bodies at their respective sites.
        for foot in feet:
            set_mocap_pose_from_site(model, data, f"{foot}_target", foot)
        set_mocap_pose_from_body(model, data, "trunk_target", "trunk")

        rate = RateLimiter(frequency=500.0)
        solver = "clarabel"
        while viewer.is_running():
            # Update task targets.
            base_task.set_target_from_mocap(data, base_mid)
            for i, task in enumerate(feet_tasks):
                task.set_target_from_mocap(data, feet_mid[i])

            # Compute velocity, integrate and set control signal.
            vel = mink.solve_ik(configuration, tasks, limits, rate.dt, solver, 1e-5)
            configuration.integrate_inplace(vel, rate.dt)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
