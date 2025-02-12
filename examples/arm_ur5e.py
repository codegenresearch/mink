from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "universal_robots_ur5e" / "scene.xml"

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML_PATH.as_posix())
    data = mujoco.MjData(model)

    configuration = mink.Configuration(model)

    # Define the end-effector task with specified costs and damping.
    end_effector_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    tasks = [end_effector_task]

    # Set up collision avoidance between wrist_3_link and floor/wall.
    collision_pairs = [
        (mink.get_body_geom_ids(model, model.body("wrist_3_link").id), ["floor", "wall"]),
    ]
    collision_avoidance_limit = mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs)

    # Define velocity limits for each joint.
    max_velocities = {
        "shoulder_pan": np.pi,
        "shoulder_lift": np.pi,
        "elbow": np.pi,
        "wrist_1": np.pi,
        "wrist_2": np.pi,
        "wrist_3": np.pi,
    }
    velocity_limit = mink.VelocityLimit(model, max_velocities)

    # Combine all limits into a single list.
    limits = [
        mink.ConfigurationLimit(model=model),
        collision_avoidance_limit,
        velocity_limit,
    ]

    # Initialize the mocap target at the end-effector site.
    mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

    # Set up the viewer and rate limiter.
    viewer = mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    )
    mujoco.mjv_defaultFreeCamera(model, viewer.cam)
    configuration.update_from_keyframe("home")
    rate_limiter = RateLimiter(frequency=500.0, warn=True)

    # Main simulation loop.
    while viewer.is_running():
        # Update task target position.
        T_wt = mink.SE3.from_mocap_name(model, data, "target")
        end_effector_task.set_target(T_wt)

        # Solve inverse kinematics and integrate the resulting velocity.
        velocity = mink.solve_ik(
            configuration, tasks, rate_limiter.dt, solver="quadprog", damping=1e-3, limits=limits
        )
        configuration.integrate_inplace(velocity, rate_limiter.dt)

        # Update the model and data for visualization.
        mujoco.mj_camlight(model, data)
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_sensorPos(model, data)

        # Render the viewer at a fixed frame rate.
        viewer.sync()
        rate_limiter.sleep()