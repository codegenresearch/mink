from pathlib import Path

import mujoco
import mujoco.viewer
from dm_control import mjcf
from loop_rate_limiters import RateLimiter

import mink

# Define paths to XML files
_HERE = Path(__file__).parent
_ARM_XML = _HERE / "kuka_iiwa_14" / "scene.xml"
_HAND_XML = _HERE / "wonik_allegro" / "left_hand.xml"

# Define finger names and home joint positions
fingers = ["rf_tip", "mf_tip", "ff_tip", "th_tip"]

# fmt: off
HOME_QPOS = [
    # iiwa.
    -0.0759329, 0.153982, 0.104381, -1.8971, 0.245996, 0.34972, -0.239115,
    # allegro.
    -0.0694123, 0.0551428, 0.986832, 0.671424,
    -0.186261, -0.0866821, 1.01374, 0.728192,
    -0.218949, -0.0318307, 1.25156, 0.840648,
    1.0593, 0.638801, 0.391599, 0.57284,
]
# fmt: on

def construct_model():
    """Constructs the Mujoco model by combining the arm and hand MJCF models."""
    arm_mjcf = mjcf.from_path(_ARM_XML.as_posix())
    arm_mjcf.find("key", "home").remove()

    hand_mjcf = mjcf.from_path(_HAND_XML.as_posix())
    palm = hand_mjcf.worldbody.find("body", "palm")
    palm.quat = (1, 0, 0, 0)
    palm.pos = (0, 0, 0.095)
    attach_site = arm_mjcf.worldbody.find("site", "attachment_site")
    attach_site.attach(hand_mjcf)

    arm_mjcf.keyframe.add("key", name="home", qpos=HOME_QPOS)

    for finger in fingers:
        body = arm_mjcf.worldbody.add("body", name=f"{finger}_target", mocap=True)
        body.add(
            "geom",
            type="sphere",
            size=".02",
            contype="0",
            conaffinity="0",
            rgba=".6 .3 .3 .5",
        )

    return mujoco.MjModel.from_xml_string(
        arm_mjcf.to_xml_string(), arm_mjcf.get_assets()
    )

def create_tasks(model):
    """Creates and returns the list of tasks for the IK solver."""
    tasks = []

    end_effector_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    tasks.append(end_effector_task)

    posture_task = mink.PostureTask(model=model, cost=5e-2)
    tasks.append(posture_task)

    for finger in fingers:
        task = mink.RelativeFrameTask(
            frame_name=f"allegro_left/{finger}",
            frame_type="site",
            root_name="allegro_left/palm",
            root_type="body",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        tasks.append(task)

    return tasks

def initialize_mocap_targets(model, data):
    """Initializes the mocap targets at their respective sites."""
    mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")
    for finger in fingers:
        mink.move_mocap_to_frame(
            model, data, f"{finger}_target", f"allegro_left/{finger}", "site"
        )

if __name__ == "__main__":
    model = construct_model()
    configuration = mink.Configuration(model)
    tasks = create_tasks(model)
    limits = [mink.ConfigurationLimit(model=model)]
    solver = "quadprog"

    with mujoco.viewer.launch_passive(
        model=model, data=configuration.data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        mujoco.mj_resetDataKeyframe(model, configuration.data, model.key("home").id)
        configuration.update(configuration.data.qpos)
        posture_task = tasks[1]
        posture_task.set_target_from_configuration(configuration)
        initialize_mocap_targets(model, configuration.data)

        T_eef_prev = configuration.get_transform_frame_to_world("attachment_site", "site")
        rate = RateLimiter(frequency=100.0, warn=False)

        while viewer.is_running():
            # Update end-effector task
            T_wt = mink.SE3.from_mocap_name(model, configuration.data, "target")
            tasks[0].set_target(T_wt)

            # Update finger tasks
            for finger, task in zip(fingers, tasks[2:]):
                T_pm = configuration.get_transform(f"{finger}_target", "body", "allegro_left/palm", "body")
                task.set_target(T_pm)

            # Update mocap positions
            T_eef = configuration.get_transform_frame_to_world("attachment_site", "site")
            T = T_eef @ T_eef_prev.inverse()
            for finger in fingers:
                T_w_mocap = mink.SE3.from_mocap_name(model, configuration.data, f"{finger}_target")
                T_w_mocap_new = T @ T_w_mocap
                body = model.body(f"{finger}_target")
                configuration.data.mocap_pos[body.mocapid[0]] = T_w_mocap_new.translation()
                configuration.data.mocap_quat[body.mocapid[0]] = T_w_mocap_new.rotation().wxyz

            # Solve IK and integrate velocity
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3, limits=limits)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, configuration.data)

            T_eef_prev = T_eef.copy()
            viewer.sync()
            rate.sleep()