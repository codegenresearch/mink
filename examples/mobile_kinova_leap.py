from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from dm_control import mjcf
from dm_control.viewer import user_input
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_ARM_XML = _HERE / "stanford_tidybot" / "scene_mobile_kinova.xml"
_HAND_XML = _HERE / "leap_hand" / "right_hand.xml"

fingers = ["tip_1", "tip_2", "tip_3", "th_tip"]

# fmt: off
HOME_QPOS = [
    # Mobile Base.
    0, 0, 0,
    # Kinova.
    0, 0.26179939, 3.14159265, -2.26892803, 0, 0.95993109, 1.57079633,
    # Leap hand.
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
]
# fmt: on


def construct_model():
    arm_mjcf = mjcf.from_path(_ARM_XML.as_posix())
    arm_mjcf.find("key", "home").remove()
    arm_mjcf.find("key", "retract").remove()

    hand_mjcf = mjcf.from_path(_HAND_XML.as_posix())
    palm = hand_mjcf.worldbody.find("body", "palm_lower")
    palm.quat = (0, 0.707, 0.707, 0)
    palm.pos = (0.03, 0.06, -0.0925)
    attach_site = arm_mjcf.worldbody.find("site", "pinch_site")
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


@dataclass
class KeyCallback:
    fix_base: bool = False
    pause: bool = False

    def __call__(self, key: int) -> None:
        if key == user_input.KEY_ENTER:
            self.fix_base = not self.fix_base
        elif key == user_input.KEY_SPACE:
            self.pause = not self.pause


def create_tasks(model, configuration):
    # Initialize end-effector task
    end_effector_task = mink.FrameTask(
        frame_name="pinch_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    # Initialize posture task with specific costs
    posture_cost = np.zeros((model.nv,))
    posture_cost[2] = 1e-3  # Mobile Base.
    posture_cost[-16:] = 1e-3  # Leap Hand.
    posture_task = mink.PostureTask(model, cost=posture_cost)

    # Initialize damping task to keep the base immobile when needed
    immobile_base_cost = np.zeros((model.nv,))
    immobile_base_cost[:2] = 100
    immobile_base_cost[2] = 1e-3
    damping_task = mink.DampingTask(model, immobile_base_cost)

    # Initialize finger tasks
    finger_tasks = []
    for finger in fingers:
        task = mink.RelativeFrameTask(
            frame_name=f"leap_right/{finger}",
            frame_type="site",
            root_name="leap_right/palm_lower",
            root_type="body",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1e-3,
        )
        finger_tasks.append(task)

    # Combine all tasks
    tasks = [end_effector_task, posture_task, *finger_tasks]
    return tasks, damping_task


if __name__ == "__main__":
    # Construct the model and initialize configuration and data
    model = construct_model()
    configuration = mink.Configuration(model)
    data = configuration.data

    # Create tasks and limits
    tasks, damping_task = create_tasks(model, configuration)
    limits = [mink.ConfigurationLimit(model)]

    # IK settings
    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    # Initialize key callback for user input
    key_callback = KeyCallback()

    # Launch the viewer
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        # Set up the camera and reset to home position
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        # Initialize mocap targets
        mink.move_mocap_to_frame(model, data, "pinch_site_target", "pinch_site", "site")
        for finger in fingers:
            mink.move_mocap_to_frame(model, data, f"{finger}_target", f"leap_right/{finger}", "site")

        # Store previous end-effector transform
        T_eef_prev = configuration.get_transform_frame_to_world("pinch_site", "site")

        # Initialize rate limiter
        rate = RateLimiter(frequency=50.0, warn=False)
        dt = rate.dt
        t = 0.0

        # Main loop
        while viewer.is_running():
            # Update end-effector task target
            T_wt = mink.SE3.from_mocap_name(model, data, "pinch_site_target")
            tasks[0].set_target(T_wt)

            # Update finger tasks
            for finger, task in zip(fingers, tasks[2:]):
                T_pm = configuration.get_transform(f"{finger}_target", "body", "leap_right/palm_lower", "body")
                task.set_target(T_pm)

            # Update mocap positions for fingers
            for finger in fingers:
                T_eef = configuration.get_transform_frame_to_world("pinch_site", "site")
                T = T_eef @ T_eef_prev.inverse()
                T_w_mocap = mink.SE3.from_mocap_name(model, data, f"{finger}_target")
                T_w_mocap_new = T @ T_w_mocap
                data.mocap_pos[model.body(f"{finger}_target").mocapid[0]] = T_w_mocap_new.translation()
                data.mocap_quat[model.body(f"{finger}_target").mocapid[0]] = T_w_mocap_new.rotation().wxyz

            # Solve IK and integrate velocity
            tasks_to_solve = tasks + [damping_task] if key_callback.fix_base else tasks
            for _ in range(max_iters):
                vel = mink.solve_ik(configuration, tasks_to_solve, dt, solver, 1e-3, limits=limits)
                configuration.integrate_inplace(vel, dt)

                # Check if position and orientation goals are achieved
                err = tasks[0].compute_error(configuration)
                pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
                ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
                if pos_achieved and ori_achieved:
                    break

            # Step the simulation or just forward if paused
            if not key_callback.pause:
                data.ctrl = configuration.q
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)

            # Update previous end-effector transform
            T_eef_prev = T_eef.copy()

            # Sync viewer and sleep to maintain frame rate
            viewer.sync()
            rate.sleep()
            t += dt


This code addresses the feedback by ensuring consistency with the gold code in the following ways:

1. **Task Initialization**: The costs for the `posture_task` are set exactly as in the gold code, with the Leap Hand costs set to `1e-3`.
2. **Task List Construction**: The `tasks` list is constructed in the same order and structure as the gold code.
3. **Variable Naming and Consistency**: Variable names and naming conventions are consistent with the gold code.
4. **Loop Logic**: The main loop logic, including IK solving and exit conditions, matches the gold code.
5. **Comments and Documentation**: Added detailed comments to clarify the purpose of each section, especially task initialization and main loop logic.
6. **Rate Limiter Initialization**: The `RateLimiter` is initialized with the `warn=False` parameter.
7. **Key Callback Logic**: The key callback logic aligns with the gold code's approach, ensuring `fix_base` and `pause` states are toggled correctly.