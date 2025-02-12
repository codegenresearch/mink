from pathlib import Path
import mujoco
import mujoco.viewer
from dm_control import mjcf
from loop_rate_limiters import RateLimiter
import mink

# Define paths to the model XML files
_HERE = Path(__file__).parent
_ARM_XML = _HERE / "kuka_iiwa_14" / "scene.xml"
_HAND_XML = _HERE / "wonik_allegro" / "left_hand.xml"

# Define finger names and home joint positions
FINGERS = ["rf_tip", "mf_tip", "ff_tip", "th_tip"]
HOME_QPOS = [
    # Kuka IIWA 14 joint positions
    -0.0759329, 0.153982, 0.104381, -1.8971, 0.245996, 0.34972, -0.239115,
    # Allegro Hand joint positions
    -0.0694123, 0.0551428, 0.986832, 0.671424,
    -0.186261, -0.0866821, 1.01374, 0.728192,
    -0.218949, -0.0318307, 1.25156, 0.840648,
    1.0593, 0.638801, 0.391599, 0.57284,
]

def construct_model():
    """Constructs the Mujoco model from XML files and attaches the hand to the arm."""
    arm_mjcf = mjcf.from_path(_ARM_XML.as_posix())
    hand_mjcf = mjcf.from_path(_HAND_XML.as_posix())

    # Remove the default home keyframe from the arm
    arm_mjcf.find("key", "home").remove()

    # Position and orient the hand and attach it to the arm
    palm = hand_mjcf.worldbody.find("body", "palm")
    palm.quat = (1, 0, 0, 0)
    palm.pos = (0, 0, 0.095)
    attach_site = arm_mjcf.worldbody.find("site", "attachment_site")
    attach_site.attach(hand_mjcf)

    # Add a new home keyframe with the specified joint positions
    arm_mjcf.keyframe.add("key", name="home", qpos=HOME_QPOS)

    # Add mocap targets for each finger
    for finger in FINGERS:
        body = arm_mjcf.worldbody.add("body", name=f"{finger}_target", mocap=True)
        body.add(
            "geom",
            type="sphere",
            size=".02",
            contype="0",
            conaffinity="0",
            rgba=".6 .3 .3 .5",
        )

    # Return the constructed model
    return mujoco.MjModel.from_xml_string(
        arm_mjcf.to_xml_string(), arm_mjcf.get_assets()
    )

if __name__ == "__main__":
    # Construct the model
    model = construct_model()
    configuration = mink.Configuration(model)

    # Define tasks for end-effector and fingers
    end_effector_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=5e-2)
    finger_tasks = [
        mink.RelativeFrameTask(
            frame_name=f"allegro_left/{finger}",
            frame_type="site",
            root_name="allegro_left/palm",
            root_type="body",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        for finger in FINGERS
    ]
    tasks = [end_effector_task, posture_task, *finger_tasks]

    # Define configuration limits and IK settings
    limits = [mink.ConfigurationLimit(model=model)]
    solver = "quadprog"
    model, data = configuration.model, configuration.data

    # Launch the viewer
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)

        # Initialize mocap targets
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")
        for finger in FINGERS:
            mink.move_mocap_to_frame(
                model, data, f"{finger}_target", f"allegro_left/{finger}", "site"
            )

        # Initialize previous end-effector transform
        T_eef_prev = configuration.get_transform_frame_to_world("attachment_site", "site")

        # Set up rate limiting for simulation
        rate = RateLimiter(frequency=100.0)

        # Main simulation loop
        while viewer.is_running():
            # Update end-effector task
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            # Update finger tasks
            for finger, task in zip(FINGERS, finger_tasks):
                T_pm = configuration.get_transform(f"{finger}_target", "body", "allegro_left/palm", "body")
                task.set_target(T_pm)

            # Update mocap positions for fingers
            for finger in FINGERS:
                T_eef = configuration.get_transform_frame_to_world("attachment_site", "site")
                T = T_eef @ T_eef_prev.inverse()
                T_w_mocap = mink.SE3.from_mocap_name(model, data, f"{finger}_target")
                T_w_mocap_new = T @ T_w_mocap
                data.mocap_pos[model.body(f"{finger}_target").mocapid[0]] = T_w_mocap_new.translation()
                data.mocap_quat[model.body(f"{finger}_target").mocapid[0]] = T_w_mocap_new.rotation().wxyz

            # Solve IK and update configuration
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3, limits=limits)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Update previous end-effector transform
            T_eef_prev = T_eef.copy()

            # Sync viewer and sleep to maintain frame rate
            viewer.sync()
            rate.sleep()