from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "kuka_iiwa_14" / "scene.xml"

# IK settings
solver = "quadprog"
pos_threshold = 1e-4
ori_threshold = 1e-4
max_iters = 20
rate_limiter_frequency = 500.0

def main():
    model = mujoco.MjModel.from_xml_path(_XML_PATH.as_posix())
    data = mujoco.MjData(model)

    ## =================== ##
    ## Setup IK.
    ## =================== ##

    configuration = mink.Configuration(model)

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

    ## =================== ##

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

        rate_limiter = RateLimiter(frequency=rate_limiter_frequency, warn=False)

        while viewer.is_running():
            # Update task target.
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            tasks[0].set_target(T_wt)  # end_effector_task

            # Compute velocity and integrate into the next configuration.
            for iteration in range(max_iters):
                vel = mink.solve_ik(configuration, tasks, rate_limiter.dt, solver, 1e-3)
                configuration.integrate_inplace(vel, rate_limiter.dt)
                err = tasks[0].compute_error(configuration)  # end_effector_task
                pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
                ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
                if pos_achieved and ori_achieved:
                    break

            data.ctrl = configuration.q
            mujoco.mj_step(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate_limiter.sleep()

if __name__ == "__main__":
    main()