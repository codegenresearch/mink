from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from dm_control.viewer import user_input
from loop_rate_limiters import RateLimiter

import mink

HERE = Path(__file__).parent
XML_FILE_PATH = HERE / "stanford_tidybot" / "scene_mobile_kinova.xml"


@dataclass
class KeyCallback:
    fix_base: bool = False
    pause_simulation: bool = False

    def __call__(self, key: int) -> None:
        if key == user_input.KEY_ENTER:
            self.fix_base = not self.fix_base
        elif key == user_input.KEY_SPACE:
            self.pause_simulation = not self.pause_simulation


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_FILE_PATH))
    data = mujoco.MjData(model)

    # Joints to control: base and arm joints.
    joint_names = [
        "joint_x", "joint_y", "joint_th",  # Base joints.
        "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7",  # Arm joints.
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    configuration = mink.Configuration(model)

    end_effector_task = mink.FrameTask(
        frame_name="pinch_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    # Minimize rotation when moving the base.
    posture_cost = np.zeros(model.nv)
    posture_cost[2] = 1e-3
    posture_task = mink.PostureTask(model, cost=posture_cost)

    # Prevent base rotation, allow minimal xy movement.
    immobile_base_cost = np.zeros(model.nv)
    immobile_base_cost[:2] = 100
    immobile_base_cost[2] = 1e-3
    damping_task = mink.DampingTask(model, immobile_base_cost)

    tasks = [end_effector_task, posture_task]
    limits = [mink.ConfigurationLimit(model)]

    # Inverse kinematics settings.
    ik_solver = "quadprog"
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

        # Reset to home position.
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        # Initialize mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "pinch_site_target", "pinch_site", "site")

        rate_limiter = RateLimiter(frequency=200.0, warn=False)

        while viewer.is_running():
            # Update task target.
            end_effector_pose = mink.SE3.from_mocap_name(model, data, "pinch_site_target")
            end_effector_task.set_target(end_effector_pose)

            # Compute velocity and integrate into the next configuration.
            for _ in range(max_iterations):
                if key_callback.fix_base:
                    velocity = mink.solve_ik(
                        configuration, tasks + [damping_task], rate_limiter.dt, ik_solver, 1e-3
                    )
                else:
                    velocity = mink.solve_ik(
                        configuration, tasks, rate_limiter.dt, ik_solver, 1e-3
                    )
                configuration.integrate_inplace(velocity, rate_limiter.dt)

                # Check for task completion.
                error = end_effector_task.compute_error(configuration)
                position_achieved = np.linalg.norm(error[:3]) <= position_threshold
                orientation_achieved = np.linalg.norm(error[3:]) <= orientation_threshold
                if position_achieved and orientation_achieved:
                    break

            # Update control signals.
            if not key_callback.pause_simulation:
                data.ctrl[actuator_ids] = configuration.q[dof_ids]
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)

            # Sync viewer and sleep to maintain frame rate.
            viewer.sync()
            rate_limiter.sleep()


if __name__ == "__main__":
    main()