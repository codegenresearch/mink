from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "universal_robots_ur5e" / "scene.xml"


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    configuration = mink.Configuration(model)
    model = configuration.model
    data = configuration.data

    ## =================== ##
    ## Setup IK.
    ## =================== ##

    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
    ]

    # Enable collision avoidance between wrist_3_link and floor/wall.
    collision_pairs = [
        (mink.get_body_geom_ids(model, model.body("wrist_3_link").id), ["floor", "wall"]),
    ]

    limits = [
        mink.ConfigurationLimit(model=model),
        mink.CollisionAvoidanceLimit(
            model=model,
            geom_pairs=collision_pairs,
        ),
    ]

    max_velocities = {
        "shoulder_pan": np.pi,
        "shoulder_lift": np.pi,
        "elbow": np.pi,
        "wrist_1": np.pi,
        "wrist_2": np.pi,
        "wrist_3": np.pi,
    }
    velocity_limit = mink.VelocityLimit(model, max_velocities)
    limits.append(velocity_limit)

    mid = model.body("target").mocapid[0]

    # IK settings.
    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("home")

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

        rate = RateLimiter(frequency=500.0, warn=False)
        while viewer.is_running():
            # Update task target.
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            # Compute velocity and integrate into the next configuration.
            for i in range(max_iters):
                vel = mink.solve_ik(
                    configuration, tasks, rate.dt, solver, damping=1e-3, limits=limits
                )
                configuration.integrate_inplace(vel, rate.dt)
                err = end_effector_task.compute_error(configuration)
                pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
                ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
                if pos_achieved and ori_achieved:
                    break

            data.ctrl = configuration.q
            mujoco.mj_step(model, data)

            # Visualize camera light and sensor positions.
            mujoco.mj_camlight(model, data)
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_sensorPos(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()


### Adjustments Made:
1. **Model and Data Initialization**: Consistently used `model` and `data` variables throughout the code, assigning them from the `configuration` object after initialization.
2. **Comment Clarity**: Updated the comment to explicitly mention the geoms involved in collision avoidance.
3. **Parameter Order in Function Calls**: Ensured that the parameters are passed in the same order as in the gold code, including the damping value.
4. **Visualization of Camera and Sensors**: Included all relevant visualization calls, such as `mujoco.mj_camlight`, `mujoco.mj_fwdPosition`, and `mujoco.mj_sensorPos`.
5. **Consistency in Variable Usage**: Used the same variable names and structures as in the gold code to maintain consistency and readability.