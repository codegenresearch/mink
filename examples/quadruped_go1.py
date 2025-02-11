from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

# Define the path to the XML file
HERE = Path(__file__).parent
XML_PATH = HERE / "unitree_go1" / "scene.xml"

# Load the model from the XML file
model = mujoco.MjModel.from_xml_path(XML_PATH.as_posix())

# Create a configuration object
configuration = mink.Configuration(model)

# Define the feet names
feet_names = ["FL", "FR", "RR", "RL"]

# Create tasks for the base and posture
base_task = mink.FrameTask(
    frame_name="trunk",
    frame_type="body",
    position_cost=1.0,
    orientation_cost=1.0,
)

posture_task = mink.PostureTask(model, cost=1e-5)

# Create tasks for each foot
feet_tasks = [
    mink.FrameTask(
        frame_name=foot,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.0,
    )
    for foot in feet_names
]

# Combine all tasks
tasks = [base_task, posture_task, *feet_tasks]

# Get the mocap IDs for the base and feet
base_mocap_id = model.body("trunk_target").mocapid[0]
feet_mocap_ids = [model.body(f"{foot}_target").mocapid[0] for foot in feet_names]

# Get the model and data from the configuration
model = configuration.model
data = configuration.data
solver = "quadprog"

# Launch the viewer
with mujoco.viewer.launch_passive(
    model=model, data=data, show_left_ui=False, show_right_ui=False
) as viewer:
    mujoco.mjv_defaultFreeCamera(model, viewer.cam)

    # Initialize to the home keyframe
    configuration.update_from_keyframe("home")
    posture_task.set_target_from_configuration(configuration)

    # Initialize mocap bodies at their respective sites
    for foot in feet_names:
        mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
    mink.move_mocap_to_frame(model, data, "trunk_target", "trunk", "body")

    # Set up the rate limiter
    rate_limiter = RateLimiter(frequency=500.0)

    # Main loop
    while viewer.is_running():
        # Update task targets
        base_task.set_target(mink.SE3.from_mocap_id(data, base_mocap_id))
        for task, mocap_id in zip(feet_tasks, feet_mocap_ids):
            task.set_target(mink.SE3.from_mocap_id(data, mocap_id))

        # Solve inverse kinematics, integrate, and set control signal
        velocity = mink.solve_ik(configuration, tasks, rate_limiter.dt, solver, 1e-5)
        configuration.integrate_inplace(velocity, rate_limiter.dt)
        mujoco.mj_camlight(model, data)

        # Visualize at fixed FPS
        viewer.sync()
        rate_limiter.sleep()