from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "boston_dynamics_spot" / "scene.xml"

def setup_tasks(model, feet):
    base_task = mink.FrameTask(
        frame_name="body",
        frame_type="body",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    posture_task = mink.PostureTask(model, cost=1e-5)

    feet_tasks = [
        mink.FrameTask(
            frame_name=foot,
            frame_type="geom",
            position_cost=1.0,
            orientation_cost=0.0,
        )
        for foot in feet
    ]

    eef_task = mink.FrameTask(
        frame_name="EE",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    return [base_task, posture_task, *feet_tasks, eef_task]

def setup_mocap_targets(model, data, feet):
    for foot in feet:
        mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "geom")
    mink.move_mocap_to_frame(model, data, "body_target", "body", "body")
    mink.move_mocap_to_frame(model, data, "EE_target", "EE", "site")

def main():
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    configuration = mink.Configuration(model)
    feet = ["FL", "FR", "HR", "HL"]
    tasks = setup_tasks(model, feet)

    base_mid = model.body("body_target").mocapid[0]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]
    eef_mid = model.body("EE_target").mocapid[0]

    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)

        posture_task.set_target_from_configuration(configuration)
        setup_mocap_targets(model, data, feet)

        rate = RateLimiter(frequency=500.0, warn=True)
        while viewer.is_running():
            base_task.set_target(mink.SE3.from_mocap_id(data, base_mid))
            for i, task in enumerate(tasks[2:-1]):
                task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))
            tasks[-1].set_target(mink.SE3.from_mocap_id(data, eef_mid))

            for _ in range(max_iters):
                vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3)
                configuration.integrate_inplace(vel, rate.dt)

                pos_achieved = all(
                    np.linalg.norm(task.compute_error(configuration)[:3]) <= pos_threshold
                    for task in tasks
                )
                ori_achieved = all(
                    np.linalg.norm(task.compute_error(configuration)[3:]) <= ori_threshold
                    for task in tasks
                )
                if pos_achieved and ori_achieved:
                    break

            data.ctrl = configuration.q[7:]
            mujoco.mj_step(model, data)

            viewer.sync()
            rate.sleep()

if __name__ == "__main__":
    main()