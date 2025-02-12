from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "kuka_iiwa_14" / "scene.xml"

# IK solver settings
SOLVER = "quadprog"
POSITION_THRESHOLD = 1e-4
ORIENTATION_THRESHOLD = 1e-4
MAX_ITERATIONS = 20
DAMPING = 1e-3

# Task costs
END_EFFECTOR_POSITION_COST = 1.0
END_EFFECTOR_ORIENTATION_COST = 1.0
POSTURE_COST = 1e-2

def main():
    model = mujoco.MjModel.from_xml_path(_XML_PATH.as_posix())
    data = mujoco.MjData(model)

    # Setup IK
    configuration = mink.Configuration(model)

    tasks = [
        mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=END_EFFECTOR_POSITION_COST,
            orientation_cost=END_EFFECTOR_ORIENTATION_COST,
            lm_damping=DAMPING,
        ),
        mink.PostureTask(model=model, cost=POSTURE_COST),
    ]

    # Viewer setup
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        tasks[1].set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        # Initialize the mocap target at the end-effector site
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

        rate_limiter = RateLimiter(frequency=500.0, warn=False)
        while viewer.is_running():
            # Update task target
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            tasks[0].set_target(T_wt)

            # Compute velocity and integrate into the next configuration
            for _ in range(MAX_ITERATIONS):
                velocity = mink.solve_ik(configuration, tasks, rate_limiter.dt, SOLVER, DAMPING)
                configuration.integrate_inplace(velocity, rate_limiter.dt)
                error = tasks[0].compute_error(configuration)
                position_achieved = np.linalg.norm(error[:3]) <= POSITION_THRESHOLD
                orientation_achieved = np.linalg.norm(error[3:]) <= ORIENTATION_THRESHOLD
                if position_achieved and orientation_achieved:
                    break

            data.ctrl = configuration.q
            mujoco.mj_step(model, data)

            # Visualize at fixed FPS
            viewer.sync()
            rate_limiter.sleep()

if __name__ == "__main__":
    main()