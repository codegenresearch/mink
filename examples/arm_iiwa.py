from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "kuka_iiwa_14" / "scene.xml"

# IK settings
SOLVER = "quadprog"
POSITION_THRESHOLD = 1e-4
ORIENTATION_THRESHOLD = 1e-4
MAX_ITERATIONS = 20
RATE_LIMITER_FREQUENCY = 500.0

def setup_tasks(model):
    end_effector_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    return [end_effector_task, posture_task]

def main():
    model = mujoco.MjModel.from_xml_path(_XML_PATH.as_posix())
    data = mujoco.MjData(model)
    configuration = mink.Configuration(model)
    tasks = setup_tasks(model)

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        tasks[1].set_target_from_configuration(configuration)  # posture_task
        mujoco.mj_forward(model, data)

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

        rate_limiter = RateLimiter(frequency=RATE_LIMITER_FREQUENCY, warn=False)
        while viewer.is_running():
            # Update task target.
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            tasks[0].set_target(T_wt)  # end_effector_task

            # Compute velocity and integrate into the next configuration.
            for _ in range(MAX_ITERATIONS):
                vel = mink.solve_ik(configuration, tasks, rate_limiter.dt, SOLVER, 1e-3)
                configuration.integrate_inplace(vel, rate_limiter.dt)
                err = tasks[0].compute_error(configuration)  # end_effector_task
                pos_achieved = np.linalg.norm(err[:3]) <= POSITION_THRESHOLD
                ori_achieved = np.linalg.norm(err[3:]) <= ORIENTATION_THRESHOLD
                if pos_achieved and ori_achieved:
                    break

            data.ctrl = configuration.q
            mujoco.mj_step(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate_limiter.sleep()

if __name__ == "__main__":
    main()