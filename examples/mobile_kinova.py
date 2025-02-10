from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from dm_control.viewer import user_input
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "stanford_tidybot" / "scene_mobile_kinova.xml"

@dataclass
class KeyCallback:
    fix_base: bool = False
    pause: bool = False

    def __call__(self, key: int) -> None:
        if key == user_input.KEY_ENTER:
            self.fix_base = not self.fix_base
        elif key == user_input.KEY_SPACE:
            self.pause = not self.pause

def main():
    model = mujoco.MjModel.from_xml_path(_XML_PATH.as_posix())
    data = mujoco.MjData(model)

    # Define the joints to control.
    joint_names = [
        "joint_x", "joint_y", "joint_th",  # Base joints
        "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7",  # Arm joints
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    configuration = mink.Configuration(model)

    # Define tasks for the end-effector and posture.
    end_effector_task = mink.FrameTask(
        frame_name="pinch_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    posture_cost = np.zeros((model.nv,))
    posture_cost[2] = 1e-3
    posture_task = mink.PostureTask(model, cost=posture_cost)

    immobile_base_cost = np.zeros((model.nv,))
    immobile_base_cost[:2] = 100
    immobile_base_cost[2] = 1e-3
    damping_task = mink.DampingTask(model, immobile_base_cost)

    tasks = [end_effector_task, posture_task]
    limits = [mink.ConfigurationLimit(model)]

    # Inverse Kinematics settings.
    solver = "quadprog"
    position_threshold = 1e-4
    orientation_threshold = 1e-4
    max_iterations = 20

    key_callback = KeyCallback()

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "pinch_site_target", "pinch_site", "site")

        rate_limiter = RateLimiter(frequency=200.0)
        while viewer.is_running():
            # Update the task target.
            target_pose = mink.SE3.from_mocap_name(model, data, "pinch_site_target")
            end_effector_task.set_target(target_pose)

            # Compute velocity and integrate into the next configuration.
            for _ in range(max_iterations):
                tasks_to_use = [*tasks, damping_task] if key_callback.fix_base else tasks
                velocity = mink.solve_ik(
                    configuration, tasks_to_use, rate_limiter.dt, solver, damping=1e-3
                )
                configuration.integrate_inplace(velocity, rate_limiter.dt)

                # Check if the target is achieved.
                error = end_effector_task.compute_error(configuration)
                position_achieved = np.linalg.norm(error[:3]) <= position_threshold
                orientation_achieved = np.linalg.norm(error[3:]) <= orientation_threshold
                if position_achieved and orientation_achieved:
                    break

            # Update the control signals.
            if not key_callback.pause:
                data.ctrl[actuator_ids] = configuration.q[dof_ids]
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate_limiter.sleep()

if __name__ == "__main__":
    main()