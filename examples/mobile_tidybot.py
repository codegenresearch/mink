from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from dm_control.viewer import user_input
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "stanford_tidybot" / "scene.xml"


@dataclass
class KeyCallback:
    fix_base: bool = False
    pause: bool = False

    def __call__(self, key: int) -> None:
        if key == user_input.KEY_ENTER:
            self.fix_base = not self.fix_base
        elif key == user_input.KEY_SPACE:
            self.pause = not self.pause


def inverse_dynamics_controller(
    model, data, qpos_desired, Kp, Kd, dof_ids, actuator_ids
):
    """PD controller + inverse dynamics. Also known as computed torque control."""
    qacc_desired = (
        Kp * (qpos_desired[dof_ids] - data.qpos[dof_ids]) - Kd * data.qvel[dof_ids]
    )
    qacc_prev = data.qacc.copy()
    data.qacc[dof_ids] = qacc_desired
    mujoco.mj_inverse(model, data)
    tau = data.qfrc_inverse.copy()
    data.ctrl[actuator_ids] = tau[actuator_ids]
    # np.clip(tau[actuator_ids], *model.actuator_ctrlrange.T, out=data.ctrl[actuator_ids])
    data.qacc = qacc_prev  # Restore qacc.


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    configuration = mink.Configuration(model)

    # Posture task.
    posture_cost = np.zeros((model.nv,))
    posture_cost[3:] = 1e-3
    posture_task = mink.PostureTask(model, cost=posture_cost)

    # Damping task.
    damping_cost = np.zeros((model.nv,))
    damping_cost[:3] = 100
    damping_task = mink.DampingTask(model, damping_cost)

    # Frame tasks.
    end_effector_task = mink.FrameTask(
        frame_name="pinch_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
    )
    mobile_base_task = mink.FrameTask(
        frame_name="base_link",
        frame_type="body",
        position_cost=[1, 1, 0.0],
        orientation_cost=1,
    )

    tasks = [
        end_effector_task,
        mobile_base_task,
        posture_task,
    ]

    limits = [
        mink.ConfigurationLimit(model),
    ]

    key_callback = KeyCallback()

    model = configuration.model
    data = configuration.data
    base_mid = model.body("base_target").mocapid[0]
    solver = "quadprog"

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        mujoco.mj_resetDataKeyframe(model, data, model.key("test").id)
        configuration.update(data.qpos)
        posture_task.set_target(model.key("home").qpos)

        mink.move_mocap_to_frame(model, data, "pinch_site_target", "pinch_site", "site")
        mink.move_mocap_to_frame(model, data, "base_target", "base_link", "body")

        rate = RateLimiter(frequency=200.0, warn=False)
        dt = rate.period
        t = 0.0
        while viewer.is_running():
            # Update task targets.
            end_effector_task.set_target(
                mink.SE3.from_mocap_name(model, data, "pinch_site_target")
            )

            # Peturb the base target using a sinusoidal function in 1D.
            freq = 0.5
            noise = 0.06 * np.sin(2 * np.pi * freq * t)
            data.mocap_pos[base_mid, 0] = noise
            mobile_base_task.set_target(
                mink.SE3.from_mocap_name(model, data, "base_target")
            )

            if key_callback.fix_base:
                vel = mink.solve_ik(
                    configuration, [*tasks, damping_task], rate.dt, solver, 1e-8
                )
            else:
                vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-8)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
            t += dt
