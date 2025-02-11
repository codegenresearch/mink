from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

# Define the path to the XML file
HERE = Path(__file__).parent
XML_PATH = HERE / "unitree_g1" / "scene.xml"

# Load the model from the XML file
model = mujoco.MjModel.from_xml_path(XML_PATH.as_posix())

# Create a configuration object
configuration = mink.Configuration(model)

# Define the feet and hands
feet = ["right_foot", "left_foot"]
hands = ["right_palm", "left_palm"]

# Define the tasks
tasks = [
    mink.FrameTask(
        frame_name="pelvis",
        frame_type="body",
        position_cost=0.0,
        orientation_cost=10.0,
    ),
    mink.PostureTask(model, cost=1.0),
    mink.ComTask(cost=200.0),
]

# Add tasks for each foot
for foot in feet:
    tasks.append(
        mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=200.0,
            orientation_cost=10.0,
            lm_damping=1.0,
        )
    )

# Add tasks for each hand
for hand in hands:
    tasks.append(
        mink.FrameTask(
            frame_name=hand,
            frame_type="site",
            position_cost=200.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
    )

# Get the mocap IDs for the center of mass and feet/hands
com_mocap_id = model.body("com_target").mocapid[0]
feet_mocap_ids = [model.body(f"{foot}_target").mocapid[0] for foot in feet]
hands_mocap_ids = [model.body(f"{hand}_target").mocapid[0] for hand in hands]

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
    configuration.update_from_keyframe("stand")
    tasks[1].set_target_from_configuration(configuration)  # Posture task
    tasks[0].set_target_from_configuration(configuration)  # Pelvis orientation task

    # Initialize mocap bodies at their respective sites
    for hand, foot in zip(hands, feet):
        mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
        mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")
    data.mocap_pos[com_mocap_id] = data.subtree_com[1]

    # Set up the rate limiter
    rate_limiter = RateLimiter(frequency=200.0)

    # Main loop
    while viewer.is_running():
        # Update task targets
        tasks[2].set_target(data.mocap_pos[com_mocap_id])  # COM task
        for i, (hand_task, foot_task) in enumerate(zip(tasks[4:], tasks[3:3+len(feet)])):
            foot_task.set_target(mink.SE3.from_mocap_id(data, feet_mocap_ids[i]))
            hand_task.set_target(mink.SE3.from_mocap_id(data, hands_mocap_ids[i]))

        # Solve inverse kinematics and integrate
        velocity = mink.solve_ik(configuration, tasks, rate_limiter.dt, solver, 1e-1)
        configuration.integrate_inplace(velocity, rate_limiter.dt)
        mujoco.mj_camlight(model, data)

        # Visualize at fixed FPS
        viewer.sync()
        rate_limiter.sleep()