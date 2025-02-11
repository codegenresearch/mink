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

FINGERS = ["tip_1", "tip_2", "tip_3", "th_tip"]

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
    end_effector_task = mink.FrameTask(
        frame_name="pinch_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    posture_cost = np.zeros((model.nv,))
    posture_cost[2] = 1e-3  # Mobile Base.
    posture_cost[-16:] = 1e-3  # Leap Hand.

    posture_task = mink.PostureTask(model, cost=posture_cost)

    immobile_base_cost = np.zeros((model.nv,))
    immobile_base_cost[:2] = 100
    immobile_base_cost[2] = 1e-3
    damping_task = mink.DampingTask(model, immobile_base_cost)

    finger_tasks = [
        mink.RelativeFrameTask(
            frame_name=f"leap_right/{finger}",
            frame_type="site",
            root_name="leap_right/palm_lower",
            root_type="body",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1e-3,
        )
        for finger in FINGERS
    ]

    return end_effector_task, posture_task, damping_task, finger_tasks


if __name__ == "__main__":
    model = construct_model()
    configuration = mink.Configuration(model)

    end_effector_task, posture_task, damping_task, finger_tasks = create_tasks(model, configuration)
    tasks = [end_effector_task, posture_task, *finger_tasks]
    limits = [mink.ConfigurationLimit(model)]

    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    key_callback = KeyCallback()

    with mujoco.viewer.launch_passive(
        model=model,
        data=configuration.data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        mujoco.mj_resetDataKeyframe(model, configuration.data, model.key("home").id)
        configuration.update(configuration.data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, configuration.data)

        mink.move_mocap_to_frame(model, configuration.data, "pinch_site_target", "pinch_site", "site")
        for finger in FINGERS:
            mink.move_mocap_to_frame(model, configuration.data, f"{finger}_target", f"leap_right/{finger}", "site")

        T_eef_prev = configuration.get_transform_frame_to_world("pinch_site", "site")
        rate = RateLimiter(frequency=50.0)
        dt = rate.period
        t = 0.0

        while viewer.is_running():
            T_wt = mink.SE3.from_mocap_name(model, configuration.data, "pinch_site_target")
            end_effector_task.set_target(T_wt)

            for finger, task in zip(FINGERS, finger_tasks):
                T_pm = configuration.get_transform(f"{finger}_target", "body", "leap_right/palm_lower", "body")
                task.set_target(T_pm)

            for finger in FINGERS:
                T_eef = configuration.get_transform_frame_to_world("pinch_site", "site")
                T = T_eef @ T_eef_prev.inverse()
                T_w_mocap = mink.SE3.from_mocap_name(model, configuration.data, f"{finger}_target")
                T_w_mocap_new = T @ T_w_mocap
                configuration.data.mocap_pos[model.body(f"{finger}_target").mocapid[0]] = T_w_mocap_new.translation()
                configuration.data.mocap_quat[model.body(f"{finger}_target").mocapid[0]] = T_w_mocap_new.rotation().wxyz

            for _ in range(max_iters):
                tasks_to_solve = [*tasks, damping_task] if key_callback.fix_base else tasks
                vel = mink.solve_ik(configuration, tasks_to_solve, dt, solver, 1e-3)
                configuration.integrate_inplace(vel, dt)

                err = end_effector_task.compute_error(configuration)
                pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
                ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
                if pos_achieved and ori_achieved:
                    break

            if not key_callback.pause:
                configuration.data.ctrl = configuration.q
                mujoco.mj_step(model, configuration.data)
            else:
                mujoco.mj_forward(model, configuration.data)

            T_eef_prev = T_eef.copy()
            viewer.sync()
            rate.sleep()
            t += dt