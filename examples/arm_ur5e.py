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
    configuration = mink.Configuration(model)
    data = configuration.data

    # Define the end-effector task
    tasks = [
        mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
    ]

    # Define collision avoidance pairs
    collision_pairs = [
        ["wrist_3_link", "floor", "wall"],
    ]

    # Define limits
    limits = [
        mink.ConfigurationLimit(model=model),
        mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs),
    ]

    # Define velocity limits
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

    # Initialize the mocap target at the end-effector site
    mid = model.body("target").mocapid[0]
    mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

    # Initialize to the home keyframe
    configuration.update_from_keyframe("home")
    mujoco.mj_forward(model, data)

    # Set up the rate limiter
    rate = RateLimiter(frequency=500.0, warn=False)

    # Define the solver
    solver = "quadprog"

    # Main loop
    while mujoco.viewer.is_running():
        # Update task target
        T_wt = mink.SE3.from_mocap_name(model, data, "target")
        tasks[0].set_target(T_wt)

        # Solve IK and integrate
        vel = mink.solve_ik(
            configuration, tasks, rate.dt, solver=solver, damping=1e-3, limits=limits
        )
        configuration.integrate_inplace(vel, rate.dt)
        mujoco.mj_camlight(model, data)

        # Update positions and sensor data for collision avoidance visualization
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_sensorPos(model, data)

        # Visualize at fixed FPS
        mujoco.viewer.sync()
        rate.sleep()


It seems there was a misunderstanding regarding the `mujoco.viewer.is_running()` and `mujoco.viewer.sync()` calls. These should be part of the viewer context manager. Here is the corrected version:


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
    configuration = mink.Configuration(model)
    data = configuration.data

    # Define the end-effector task
    tasks = [
        mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
    ]

    # Define collision avoidance pairs
    collision_pairs = [
        ["wrist_3_link", "floor", "wall"],
    ]

    # Define limits
    limits = [
        mink.ConfigurationLimit(model=model),
        mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs),
    ]

    # Define velocity limits
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

    # Initialize the mocap target at the end-effector site
    mid = model.body("target").mocapid[0]
    mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

    # Initialize to the home keyframe
    configuration.update_from_keyframe("home")
    mujoco.mj_forward(model, data)

    # Set up the rate limiter
    rate = RateLimiter(frequency=500.0, warn=False)

    # Define the solver
    solver = "quadprog"

    # Initialize the viewer
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Main loop
        while viewer.is_running():
            # Update task target
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            tasks[0].set_target(T_wt)

            # Solve IK and integrate
            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver=solver, damping=1e-3, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Update positions and sensor data for collision avoidance visualization
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_sensorPos(model, data)

            # Visualize at fixed FPS
            viewer.sync()
            rate.sleep()