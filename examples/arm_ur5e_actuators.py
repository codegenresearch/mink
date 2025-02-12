from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "universal_robots_ur5e" / "scene.xml"

# Load the model and data
model = mujoco.MjModel.from_xml_path(_XML_PATH.as_posix())
data = mujoco.MjData(model)

# Setup Inverse Kinematics (IK)
configuration = mink.Configuration(model)

# Define the end-effector task
end_effector_task = mink.FrameTask(
    frame_name="attachment_site",
    frame_type="site",
    position_cost=1.0,
    orientation_cost=1.0,
    lm_damping=1.0,
)
tasks = [end_effector_task]

# Set up collision avoidance between wrist_3_link and floor, wall
wrist_3_geoms = mink.get_body_geom_ids(model, model.body("wrist_3_link").id)
collision_pairs = [(wrist_3_geoms, ["floor", "wall"])]
limits = [
    mink.ConfigurationLimit(model=configuration.model),
    mink.CollisionAvoidanceLimit(
        model=configuration.model,
        geom_pairs=collision_pairs,
    ),
]

# Define velocity limits for each joint
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

# IK settings
solver = "quadprog"
position_threshold = 1e-4
orientation_threshold = 1e-4
max_iterations = 20

# Initialize viewer and reset data to home keyframe
with mujoco.viewer.launch_passive(
    model=model, data=data, show_left_ui=False, show_right_ui=False
) as viewer:
    mujoco.mjv_defaultFreeCamera(model, viewer.cam)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    configuration.update(data.qpos)
    mujoco.mj_forward(model, data)

    # Initialize the mocap target at the end-effector site
    mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

    rate_limiter = RateLimiter(frequency=500.0)
    while viewer.is_running():
        # Update task target
        T_wt = mink.SE3.from_mocap_name(model, data, "target")
        end_effector_task.set_target(T_wt)

        # Solve IK and integrate velocity into the next configuration
        for _ in range(max_iterations):
            velocity = mink.solve_ik(
                configuration, tasks, rate_limiter.dt, solver, damping=1e-3, limits=limits
            )
            configuration.integrate_inplace(velocity, rate_limiter.dt)
            error = end_effector_task.compute_error(configuration)
            position_achieved = np.linalg.norm(error[:3]) <= position_threshold
            orientation_achieved = np.linalg.norm(error[3:]) <= orientation_threshold
            if position_achieved and orientation_achieved:
                break

        # Apply the new configuration to the model
        data.ctrl = configuration.q
        mujoco.mj_step(model, data)

        # Visualize at fixed FPS
        viewer.sync()
        rate_limiter.sleep()